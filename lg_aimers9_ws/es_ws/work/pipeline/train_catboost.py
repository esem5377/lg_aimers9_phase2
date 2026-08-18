"""CatBoost 버전 최종 모델 학습 — 리더보드 테스트 제출용.

train.py와 동일한 피처(trackman context 3종 포함, raw pitcher_id/batter_id
제외, team history/cross 그룹 제외 — 둘 다 실험에서 효과 없었음)와 동일한
시간 기반 검증(2019~2023 학습 / 2024 검증)을 사용. tune_v2.py 실험에서
CatBoost 단일 모델이 LightGBM보다 AUC 0.0012 높게 나온 걸 실제로 배포
가능한 형태로 만든다.

8/18 추가: `tune_ensemble_calibrated.py` 실험에서 sigmoid 확률보정만으로
2024 홀드아웃 BSS가 800.75 -> 825.57 (+24.8)로 오르는 걸 확인했다(AUC는
보정이 monotonic 변환이라 그대로 0.55056 -- 순위는 안 바뀌고 확률의
절대적 정확도만 좋아지는데, 대회 실제 채점 지표가 AUC가 아니라 BSS라서
여기서 실질적 이득이 남음). CalibratedClassifierCV 객체를 그대로 pickle
하면 sklearn 버전 의존성 + 8/16에 겪은 pandas DataFrame 크로스버전 unpickle
문제와 같은 리스크가 있으므로, sigmoid 보정(Platt scaling)을 raw
확률 1개를 입력으로 받는 2-파라미터(a, b) 로지스틱 회귀로 직접 재현해서
`feature_meta.json`에 순수 float로 저장한다 -- 추론 시엔
`1 / (1 + exp(-(a * raw_p + b)))`만 계산하면 되므로 버전 의존성이 전혀 없다.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt_scaling(raw_p, y):
    """CalibratedClassifierCV(method='sigmoid')와 동등한 Platt scaling을
    버전 의존성 없는 (a, b) 두 스칼라로 직접 재현한다.
    calibrated_p = sigmoid(a * raw_p + b)
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt_scaling(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model_catboost"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

# work/tune_catboost.py 랜덤서치 trial 7 결과 (auc=0.55056, baseline 0.55005 대비 +0.0005).
BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(
        "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model", "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data...")
    df = load_data()
    print(f" shape={df.shape}")

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    X_all = build_features(df)
    y_all = df[TARGET_COL]
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[valid_mask], y_all[valid_mask]
    print(f" train={X_tr.shape} valid(2024)={X_va.shape}")

    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

    print("Fit (time-split validation)...")
    model_cv = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=200,
        **BEST_PARAMS,
    )
    model_cv.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    va_pred = model_cv.predict_proba(X_va)[:, 1]
    metrics = {
        "valid_season": 2024,
        "auc": roc_auc_score(y_va, va_pred),
        "logloss": log_loss(y_va, va_pred),
        "accuracy@0.5": accuracy_score(y_va, (va_pred >= 0.5).astype(int)),
        "bss_raw": bss_score(va_pred, y_va),
        "best_iteration": model_cv.get_best_iteration(),
    }
    print("Validation metrics (raw, 보정 전):", json.dumps(metrics, indent=2))

    print("\nSigmoid 보정(Platt scaling) 검증 -- 2024 홀드아웃에 직접 fit해 BSS 개선 재확인...")
    a, b = fit_platt_scaling(va_pred, y_va)
    va_pred_calib = apply_platt_scaling(va_pred, a, b)
    # CalibratedClassifierCV(sigmoid) 결과와 수치가 일치하는지 교차 검증
    sk_calib = CalibratedClassifierCV(estimator=FrozenEstimator(model_cv), method="sigmoid")
    sk_calib.fit(X_va, y_va)
    sk_pred_calib = sk_calib.predict_proba(X_va)[:, 1]
    max_abs_diff = float(np.max(np.abs(va_pred_calib - sk_pred_calib)))
    metrics["bss_calibrated"] = bss_score(va_pred_calib, y_va)
    metrics["platt_vs_sklearn_max_abs_diff"] = max_abs_diff
    print(f" 수동 Platt scaling BSS = {metrics['bss_calibrated']:.2f} "
          f"(raw {metrics['bss_raw']:.2f} 대비 {metrics['bss_calibrated'] - metrics['bss_raw']:+.2f})")
    print(f" sklearn CalibratedClassifierCV와 예측값 최대 절대오차 = {max_abs_diff:.2e} (0에 가까울수록 재현 정확)")
    with open(os.path.join(OUT_DIR, "metrics_catboost.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nRefit on full data with best_iteration (보정용 5% carve-out 분리)...")
    best_iter = max(model_cv.get_best_iteration(), 1)
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=42,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}")
    model_final = CatBoostClassifier(
        iterations=best_iter, loss_function="Logloss", random_seed=42,
        cat_features=cat_idx, verbose=False,
        **BEST_PARAMS,
    )
    model_final.fit(X_train_final, y_train_final)

    final_calib_pred_raw = model_final.predict_proba(X_calib)[:, 1]
    a_final, b_final = fit_platt_scaling(final_calib_pred_raw, y_calib)
    final_calib_pred = apply_platt_scaling(final_calib_pred_raw, a_final, b_final)
    print(f" carve-out BSS: raw={bss_score(final_calib_pred_raw, y_calib):.2f}  "
          f"calibrated={bss_score(final_calib_pred, y_calib):.2f}")

    model_final.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2)
    print(f"Saved model to {MODEL_DIR}/catboost.cbm")


if __name__ == "__main__":
    main()
