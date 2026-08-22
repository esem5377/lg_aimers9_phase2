"""992점(jh_ws v20_control_risk_score, CatBoost 6시드) 위에 LGBM 배깅을
블렌드하는 프로덕션 후보.

`tune_arch_blend_honest_riskscore.py`로 fold0(->2022)/fold2(->2024)
walk-forward + calib_fit/calib_eval 정직 분리 검증: 두 폴드 모두 거의 동일한
크기로 양수(+3.02/+3.00, 이 프로젝트에서 보기 드물게 안정적인 일치). 가중치는
두 폴드 다 cat 0.75~0.8 / lgb 0.2, **XGBoost는 두 폴드 다 0에 가까움(0.05,
-0.0)** -- v9(986점) 때와 달리 이번 피처셋에서는 XGBoost 기여가 사실상 없어
아예 빼고 CatBoost+LGBM 2-way 블렌드로 단순화(xgboost 버전 설치 이슈였던
전례도 있어 불필요한 의존성 제거 겸함).

CatBoost 6시드는 jh_ws v20 자산을 재학습 없이 그대로 로드. LGBM만 신규
학습(3시드), 가중치는 walk-forward 평균(cat 0.78/lgb 0.22)을 고정 사용
(carve-out에서 재탐색하면 무작위분할 낙관편향 함정 재현 우려 -- v9
빌드에서 이미 확인된 패턴, es_ws/work/pipeline/train_arch_blend_bagged.py
참고).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../es_ws/work
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
JH_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(_BASE)), "jh_ws", "v20_control_risk_score", "model")
OUT_MODEL_DIR = os.path.join(_BASE, "model_riskscore_blend")
OUT_DIR = os.path.join(_BASE, "output")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

LGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.02, num_leaves=63,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, verbosity=-1,
)
NEW_SEEDS = [42, 7, 123]
BLEND_WEIGHTS = {"cat": 0.78, "lgb": 0.22}  # walk-forward honest 평균 (xgb 기여 0에 가까워 제외)
DATA_SPLIT_SEED = 42


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


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(_BASE, "model", "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df


def build_features(df, id_mappings, cat_dtype):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str) if cat_dtype == "str" else X[c].astype(str).astype("category")
    return X


def main():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("Load jh_ws v20 feature_meta (id_mappings, CatBoost 6시드 재사용)...")
    with open(os.path.join(JH_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        jh_meta = json.load(f)
    id_mappings = jh_meta["id_mappings"]
    cat_seeds = jh_meta["seeds"]
    cat_models = []
    for seed in cat_seeds:
        m = CatBoostClassifier()
        m.load_model(os.path.join(JH_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cat_models.append(m)
    print(f" CatBoost {len(cat_models)}개 모델 로드 완료 (seeds={cat_seeds})")

    print("\nLoad train data (+control_risk_score)...")
    df = load_data()
    print(f" shape={df.shape}  ({time.time()-t0:.0f}s)")
    y_all = df[TARGET_COL]

    X_all_str = build_features(df, id_mappings, "str")
    X_all_cat = build_features(df, id_mappings, "category")

    print("\nCalibration carve-out(5%) 분리 (jh_ws 프로덕션과 동일 관례)...")
    idx_train, idx_calib = train_test_split(df.index, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED)
    print(f" train={len(idx_train)}  calib={len(idx_calib)}  ({time.time()-t0:.0f}s)")

    X_tr_cat = X_all_cat.loc[idx_train]
    y_tr = y_all.loc[idx_train]
    X_calib_str, X_calib_cat = X_all_str.loc[idx_calib], X_all_cat.loc[idx_calib]
    y_calib = y_all.loc[idx_calib]

    print("\n=== CatBoost 6시드 배깅 예측 (calib carve-out, 재학습 없음) ===")
    cat_calib_raw = np.mean([m.predict_proba(X_calib_str)[:, 1] for m in cat_models], axis=0)
    print(f" cat bagged raw BSS(calib) = {bss_score(cat_calib_raw, y_calib):.2f}  ({time.time()-t0:.0f}s)")

    print("\n=== LightGBM 신규 3시드 학습 ===")
    lgb_models = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m.fit(X_tr_cat, y_tr, categorical_feature=CAT_COLS)
        lgb_models.append(m)
        m.booster_.save_model(os.path.join(OUT_MODEL_DIR, f"lgb_seed{seed}.txt"))
        print(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    lgb_calib_raw = np.mean([m.predict_proba(X_calib_cat)[:, 1] for m in lgb_models], axis=0)
    print(f" lgb bagged raw BSS(calib) = {bss_score(lgb_calib_raw, y_calib):.2f}  ({time.time()-t0:.0f}s)")

    print("\n=== 블렌드: walk-forward(honest) 고정 가중치 사용 (carve-out 재탐색 안 함) ===")
    print(f" 채택 가중치 = {BLEND_WEIGHTS}")
    blend_raw = BLEND_WEIGHTS["cat"] * cat_calib_raw + BLEND_WEIGHTS["lgb"] * lgb_calib_raw
    a_blend, b_blend = fit_platt_scaling(blend_raw, y_calib)
    blend_calib = apply_platt_scaling(blend_raw, a_blend, b_blend)
    cat_a, cat_b = fit_platt_scaling(cat_calib_raw, y_calib)
    cat_only_calib = apply_platt_scaling(cat_calib_raw, cat_a, cat_b)
    print(f" 전체 calib(5%) 위 비교: CatBoost단독 calibrated={bss_score(cat_only_calib, y_calib):.2f}  "
          f"블렌드 calibrated={bss_score(blend_calib, y_calib):.2f}  "
          f"delta={bss_score(blend_calib, y_calib) - bss_score(cat_only_calib, y_calib):+.2f}")
    print(f" 블렌드: raw={bss_score(blend_raw, y_calib):.2f}  calibrated={bss_score(blend_calib, y_calib):.2f}")

    meta = {
        "cat_cols": CAT_COLS, "raw_id_cols": RAW_ID_COLS, "id_mappings": id_mappings,
        "cat_seeds": cat_seeds, "lgb_seeds": NEW_SEEDS,
        "blend_weights": BLEND_WEIGHTS,
        "calibration": {"method": "platt_sigmoid_on_blend", "a": a_blend, "b": b_blend},
        "columns_str": list(X_all_str.columns), "columns_cat": list(X_all_cat.columns),
        "control_risk_score_feature": True,
        "honest_walkforward_check": {
            "fold0_delta": 3.02, "fold2_delta": 3.00,
            "source": "es_ws/work/experiments/tune_arch_blend_honest_riskscore.py",
        },
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved feature_meta.json + lgb model files to {OUT_MODEL_DIR}")
    print(f"총 소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
