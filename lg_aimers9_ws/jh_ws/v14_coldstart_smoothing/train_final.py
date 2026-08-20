"""974점(es_ws v7) 레시피 그대로 + asof_* cold-start(표본 0) NaN에
empirical Bayes smoothing 추가. 그 외(CAT_COLS/RAW_ID_COLS/trackman
context/BEST_PARAMS/Platt calibration)는 전부 동일.

8/19 실측 EDA: train.csv에서 결측이 나는 컬럼은 전부 asof_* rate 계열뿐.
  - asof_pitcher_n==0(첫 투구)일 때 pitcher 통산 rate 8개 결측: 792건
  - 직전 1/3/5경기 이력 없을 때(prev1/3/5) rate 6개 결측: 29,185건(가장 큼)
  - asof_batter_n==0(첫 타석)일 때 batter 통산 rate 2개 결측: 830건
표본수(n)가 정확히 0인 행이라 "empirical Bayes 사후추정 = n/(n+k)*개인추정 +
k/(n+k)*prior"에서 n=0이면 그대로 prior(=전체 train 평균)로 수렴한다.
즉 이 케이스는 "전역 평균으로 채우기"와 수학적으로 동일 -- 이 값을
feature_meta.json에 저장해 script.py도 test.csv cold-start에 동일하게
적용한다(test.csv 자체 통계를 쓰지 않으므로 row-wise 독립 추론 원칙 준수).

iterations=2000 -- 이번엔 500 스케일 비교가 아니라 진짜 974 프로덕션
스케일 그대로 확인.
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
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt_scaling(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v14_coldstart_smoothing\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v14_coldstart_smoothing\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

COLDSTART_RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
ITERATIONS = 2000


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_coldstart_priors(df):
    """n=0 cold-start rate 컬럼의 전역 평균(prior) -- empirical Bayes에서
    n=0일 때의 사후추정치와 동일."""
    priors = {}
    for c in COLDSTART_RATE_COLS:
        n_missing = int(df[c].isna().sum())
        prior = float(df[c].mean())  # NaN 자동 제외하고 평균
        priors[c] = prior
        print(f" {c}: n_missing={n_missing}  prior(global_mean)={prior:.4f}")
    return priors


def apply_coldstart_smoothing(df, priors):
    df = df.copy()
    for c, prior in priors.items():
        df[c] = df[c].fillna(prior)
    return df


def build_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data...")
    df = load_data()
    print(f" shape={df.shape}")

    print("Build cold-start priors (전체 train 기준 전역 평균)...")
    priors = build_coldstart_priors(df)
    df = apply_coldstart_smoothing(df, priors)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  batter_id n={len(id_mappings['batter_id'])}")

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[valid_mask], y_all[valid_mask]
    print(f" train={X_tr.shape} valid(2024)={X_va.shape}")

    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

    print("Fit (time-split validation, 참고용 리포트)...")
    model_cv = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC", random_seed=42,
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
    print("Validation metrics (raw):", json.dumps(metrics, indent=2))

    a, b = fit_platt_scaling(va_pred, y_va)
    va_pred_calib = apply_platt_scaling(va_pred, a, b)
    sk_calib = CalibratedClassifierCV(estimator=FrozenEstimator(model_cv), method="sigmoid")
    sk_calib.fit(X_va, y_va)
    sk_pred_calib = sk_calib.predict_proba(X_va)[:, 1]
    max_abs_diff = float(np.max(np.abs(va_pred_calib - sk_pred_calib)))
    metrics["bss_calibrated"] = bss_score(va_pred_calib, y_va)
    metrics["platt_vs_sklearn_max_abs_diff"] = max_abs_diff
    print(f" 보정 후 BSS(2024, 참고용) = {metrics['bss_calibrated']:.2f} "
          f"(raw {metrics['bss_raw']:.2f} 대비 {metrics['bss_calibrated'] - metrics['bss_raw']:+.2f})")
    with open(os.path.join(OUT_DIR, "metrics_v14_coldstart.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

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
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "coldstart_rate_cols": COLDSTART_RATE_COLS,
            "coldstart_priors": priors,
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved model to {MODEL_DIR}\\catboost.cbm")


if __name__ == "__main__":
    main()
