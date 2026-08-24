"""submit_v11_riskscore_blend(993점) 위에 시드 배깅 확장 -- CatBoost 6->9,
LGBM 3->6. 블렌드 가중치/피처셋 변경 없음(v11과 동일 cat 0.78/lgb 0.22,
control_risk_score/_weighted 포함), 순수하게 각 아키텍처 내부 모델 분산만
더 줄이는 시도.

8/22 세션에서 이 확장을 옛(982) 레시피로 먼저 시도했다가 원인 불명으로
3시간 40분+ 실행되는 걸 보고 강제 종료한 전례가 있어(work/pipeline/
train_arch_blend_bagged_v2.py, 완료 못 함), 이번엔 verbose=200 + flush=True로
CatBoost 학습 진행 상황이 로그에 실시간으로 찍히도록 해서 이상 징후를
빨리 포착할 수 있게 했다.

v11과 동일한 calibration carve-out(random_state=42, 5%)에서 apples-to-apples
비교: v11 calibrated BSS = 2070.44.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, Booster as LGBBooster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
JH_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(_BASE)), "jh_ws", "v20_control_risk_score", "model")
V11_MODEL_DIR = os.path.join(_BASE, "model_riskscore_blend")
OUT_MODEL_DIR = os.path.join(_BASE, "model_riskscore_blend_v2")
OUT_DIR = os.path.join(_BASE, "output")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
LGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.02, num_leaves=63,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, verbosity=-1,
)
NEW_SEEDS = [11, 22, 33]
BLEND_WEIGHTS = {"cat": 0.78, "lgb": 0.22}
DATA_SPLIT_SEED = 42


def log(msg):
    print(msg, flush=True)


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

    log("Load v11 feature_meta (id_mappings, 기존 6+3 모델)...")
    with open(os.path.join(V11_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        v11_meta = json.load(f)
    id_mappings = v11_meta["id_mappings"]

    cat_models_old = []
    for seed in v11_meta["cat_seeds"]:
        m = CatBoostClassifier()
        m.load_model(os.path.join(JH_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cat_models_old.append(m)
    lgb_models_old = [LGBBooster(model_file=os.path.join(V11_MODEL_DIR, f"lgb_seed{s}.txt")) for s in v11_meta["lgb_seeds"]]
    log(f" 기존: cat={len(cat_models_old)} lgb={len(lgb_models_old)}")

    log("\nLoad train data (+control_risk_score)...")
    df = load_data()
    log(f" shape={df.shape}  ({time.time()-t0:.0f}s)")
    y_all = df[TARGET_COL]
    X_all_str = build_features(df, id_mappings, "str")
    X_all_cat = build_features(df, id_mappings, "category")
    cat_idx = [X_all_str.columns.get_loc(c) for c in CAT_COLS]

    log("\nCalibration carve-out(5%) 분리 (v11과 동일 random_state=42)...")
    idx_train, idx_calib = train_test_split(df.index, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED)
    log(f" train={len(idx_train)}  calib={len(idx_calib)}  ({time.time()-t0:.0f}s)")

    X_tr_str, X_tr_cat = X_all_str.loc[idx_train], X_all_cat.loc[idx_train]
    y_tr = y_all.loc[idx_train]
    X_calib_str, X_calib_cat = X_all_str.loc[idx_calib], X_all_cat.loc[idx_calib]
    y_calib = y_all.loc[idx_calib]

    log("\n=== 기존 모델 calib carve-out 예측 (재학습 없음) ===")
    cat_raw_old = [m.predict_proba(X_calib_str)[:, 1] for m in cat_models_old]
    lgb_raw_old = [m.predict(X_calib_cat) for m in lgb_models_old]

    log(f"\n=== CatBoost 신규 {len(NEW_SEEDS)}시드 학습 (iterations=2000, verbose=200) ===")
    sys.stdout.flush()
    cat_models_new = []
    for seed in NEW_SEEDS:
        ts = time.time()
        log(f" --- seed={seed} 시작 ({time.time()-t0:.0f}s elapsed) ---")
        m = CatBoostClassifier(iterations=2000, loss_function="Logloss", random_seed=seed,
                                cat_features=cat_idx, verbose=200, **CAT_BEST_PARAMS)
        m.fit(X_tr_str, y_tr)
        cat_models_new.append(m)
        m.save_model(os.path.join(OUT_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        log(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    cat_raw_new = [m.predict_proba(X_calib_str)[:, 1] for m in cat_models_new]

    log(f"\n=== LightGBM 신규 {len(NEW_SEEDS)}시드 학습 ===")
    lgb_models_new = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m.fit(X_tr_cat, y_tr, categorical_feature=CAT_COLS)
        lgb_models_new.append(m)
        m.booster_.save_model(os.path.join(OUT_MODEL_DIR, f"lgb_seed{seed}.txt"))
        log(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    lgb_raw_new = [m.predict_proba(X_calib_cat)[:, 1] for m in lgb_models_new]

    log("\n=== 배깅 확장 비교 (동일 calib carve-out, v11과 apples-to-apples) ===")
    cat_raw_6 = np.mean(cat_raw_old, axis=0)
    cat_raw_9 = np.mean(cat_raw_old + cat_raw_new, axis=0)
    lgb_raw_3 = np.mean(lgb_raw_old, axis=0)
    lgb_raw_6 = np.mean(lgb_raw_old + lgb_raw_new, axis=0)

    def blend_calibrated(cat_r, lgb_r):
        raw = BLEND_WEIGHTS["cat"] * cat_r + BLEND_WEIGHTS["lgb"] * lgb_r
        a, b = fit_platt_scaling(raw, y_calib)
        calib = apply_platt_scaling(raw, a, b)
        return bss_score(raw, y_calib), bss_score(calib, y_calib), a, b

    raw_v11, calib_v11, _, _ = blend_calibrated(cat_raw_6, lgb_raw_3)
    raw_v12, calib_v12, a12, b12 = blend_calibrated(cat_raw_9, lgb_raw_6)
    log(f" v11 재현(6/3): raw={raw_v11:.2f}  calibrated={calib_v11:.2f}  (원래 로그값 2053.51/2070.44와 비교)")
    log(f" v12(9/6):      raw={raw_v12:.2f}  calibrated={calib_v12:.2f}")
    log(f" delta(v12-v11) = {calib_v12 - calib_v11:+.2f}")

    meta = {
        "cat_cols": CAT_COLS, "raw_id_cols": RAW_ID_COLS, "id_mappings": id_mappings,
        "cat_seeds": v11_meta["cat_seeds"] + NEW_SEEDS,
        "lgb_seeds": v11_meta["lgb_seeds"] + NEW_SEEDS,
        "blend_weights": BLEND_WEIGHTS,
        "calibration": {"method": "platt_sigmoid_on_blend", "a": a12, "b": b12},
        "columns_str": list(X_all_str.columns), "columns_cat": list(X_all_cat.columns),
        "control_risk_score_feature": True,
        "v11_vs_v12_same_carveout": {"v11_calibrated": calib_v11, "v12_calibrated": calib_v12, "delta": calib_v12 - calib_v11},
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {OUT_MODEL_DIR}  (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
