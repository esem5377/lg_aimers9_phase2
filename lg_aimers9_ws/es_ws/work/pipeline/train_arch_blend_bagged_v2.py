"""submit_v9_arch_blend(986점) 위에 시드 배깅을 추가 확장 -- CatBoost
6->9시드, LGBM/XGBoost 3->6시드. 새 피처/블렌드가중치 변경 없음(v9와 동일
cat 0.85/lgb 0.1/xgb 0.05), 순수하게 각 아키텍처 내부의 모델 분산만 더
줄이는 시도. jh_ws는 자기 머신 기준(시드당 30분) "한계효용 낮다"고
판단했지만 이 머신은 시드당 CatBoost ~8분/LGBM,XGBoost 1분 미만이라
비용이 훨씬 낮아 시도해볼 가치가 있음.

v9와 동일한 calibration carve-out(random_state=42, 5%)에서 apples-to-apples
비교: v9 calibrated BSS = 2080.01. 이 스크립트는 새 시드들을 기존 6+3+3
모델과 합쳐 재계산한 calibrated BSS를 그 위에서 직접 비교한다.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, Booster as LGBBooster
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
JH_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(_BASE)), "jh_ws", "v18_seed_bagging", "model")
V9_MODEL_DIR = os.path.join(_BASE, "model_arch_blend")
OUT_MODEL_DIR = os.path.join(_BASE, "model_arch_blend_v2")
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
XGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.02, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    enable_categorical=True, n_jobs=-1,
)
NEW_SEEDS = [11, 22, 33]
BLEND_WEIGHTS = {"cat": 0.85, "lgb": 0.1, "xgb": 0.05}  # v9와 동일, walk-forward 채택값
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

    print("Load v9 feature_meta (id_mappings, 기존 6+3+3 모델)...")
    with open(os.path.join(V9_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        v9_meta = json.load(f)
    id_mappings = v9_meta["id_mappings"]

    cat_models_old = []
    for seed in v9_meta["cat_seeds"]:
        m = CatBoostClassifier()
        m.load_model(os.path.join(JH_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cat_models_old.append(m)

    lgb_models_old = [LGBBooster(model_file=os.path.join(V9_MODEL_DIR, f"lgb_seed{s}.txt")) for s in v9_meta["lgb_seeds"]]

    xgb_models_old = []
    for seed in v9_meta["xgb_seeds"]:
        m = XGBClassifier()
        m.load_model(os.path.join(V9_MODEL_DIR, f"xgb_seed{seed}.json"))
        xgb_models_old.append(m)
    print(f" 기존: cat={len(cat_models_old)} lgb={len(lgb_models_old)} xgb={len(xgb_models_old)}")

    print("\nLoad train data...")
    df = load_data()
    print(f" shape={df.shape}  ({time.time()-t0:.0f}s)")
    y_all = df[TARGET_COL]
    X_all_str = build_features(df, id_mappings, "str")
    X_all_cat = build_features(df, id_mappings, "category")
    cat_idx = [X_all_str.columns.get_loc(c) for c in CAT_COLS]

    print("\nCalibration carve-out(5%) 분리 (v9와 동일 random_state=42 -- apples-to-apples 비교용)...")
    idx_train, idx_calib = train_test_split(df.index, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED)
    print(f" train={len(idx_train)}  calib={len(idx_calib)}  ({time.time()-t0:.0f}s)")

    X_tr_str, X_tr_cat = X_all_str.loc[idx_train], X_all_cat.loc[idx_train]
    y_tr = y_all.loc[idx_train]
    X_calib_str, X_calib_cat = X_all_str.loc[idx_calib], X_all_cat.loc[idx_calib]
    y_calib = y_all.loc[idx_calib]

    print("\n=== 기존 모델 calib carve-out 예측 (재학습 없음) ===")
    cat_raw_old = [m.predict_proba(X_calib_str)[:, 1] for m in cat_models_old]
    lgb_raw_old = [m.predict(X_calib_cat) for m in lgb_models_old]
    xgb_raw_old = [m.predict_proba(X_calib_cat)[:, 1] for m in xgb_models_old]

    print(f"\n=== CatBoost 신규 {len(NEW_SEEDS)}시드 학습 (iterations=2000) ===")
    cat_models_new = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = CatBoostClassifier(iterations=2000, loss_function="Logloss", random_seed=seed,
                                cat_features=cat_idx, verbose=False, **CAT_BEST_PARAMS)
        m.fit(X_tr_str, y_tr)
        cat_models_new.append(m)
        m.save_model(os.path.join(OUT_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        print(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    cat_raw_new = [m.predict_proba(X_calib_str)[:, 1] for m in cat_models_new]

    print(f"\n=== LightGBM 신규 {len(NEW_SEEDS)}시드 학습 ===")
    lgb_models_new = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m.fit(X_tr_cat, y_tr, categorical_feature=CAT_COLS)
        lgb_models_new.append(m)
        m.booster_.save_model(os.path.join(OUT_MODEL_DIR, f"lgb_seed{seed}.txt"))
        print(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    lgb_raw_new = [m.predict_proba(X_calib_cat)[:, 1] for m in lgb_models_new]

    print(f"\n=== XGBoost 신규 {len(NEW_SEEDS)}시드 학습 ===")
    xgb_models_new = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(X_tr_cat, y_tr)
        xgb_models_new.append(m)
        m.save_model(os.path.join(OUT_MODEL_DIR, f"xgb_seed{seed}.json"))
        print(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    xgb_raw_new = [m.predict_proba(X_calib_cat)[:, 1] for m in xgb_models_new]

    print("\n=== 배깅 확장 비교 (동일 calib carve-out, v9와 apples-to-apples) ===")
    cat_raw_6 = np.mean(cat_raw_old, axis=0)
    cat_raw_9 = np.mean(cat_raw_old + cat_raw_new, axis=0)
    lgb_raw_3 = np.mean(lgb_raw_old, axis=0)
    lgb_raw_6 = np.mean(lgb_raw_old + lgb_raw_new, axis=0)
    xgb_raw_3 = np.mean(xgb_raw_old, axis=0)
    xgb_raw_6 = np.mean(xgb_raw_old + xgb_raw_new, axis=0)

    def blend_calibrated(cat_r, lgb_r, xgb_r):
        raw = BLEND_WEIGHTS["cat"] * cat_r + BLEND_WEIGHTS["lgb"] * lgb_r + BLEND_WEIGHTS["xgb"] * xgb_r
        a, b = fit_platt_scaling(raw, y_calib)
        calib = apply_platt_scaling(raw, a, b)
        return bss_score(raw, y_calib), bss_score(calib, y_calib), a, b

    raw_v9, calib_v9, _, _ = blend_calibrated(cat_raw_6, lgb_raw_3, xgb_raw_3)
    raw_v10, calib_v10, a10, b10 = blend_calibrated(cat_raw_9, lgb_raw_6, xgb_raw_6)
    print(f" v9 재현(6/3/3): raw={raw_v9:.2f}  calibrated={calib_v9:.2f}  (원래 로그값 2064.70/2080.01과 비교)")
    print(f" v10(9/6/6):     raw={raw_v10:.2f}  calibrated={calib_v10:.2f}")
    print(f" delta(v10-v9) = {calib_v10 - calib_v9:+.2f}")

    meta = {
        "cat_cols": CAT_COLS, "raw_id_cols": RAW_ID_COLS, "id_mappings": id_mappings,
        "cat_seeds": v9_meta["cat_seeds"] + NEW_SEEDS,
        "lgb_seeds": v9_meta["lgb_seeds"] + NEW_SEEDS,
        "xgb_seeds": v9_meta["xgb_seeds"] + NEW_SEEDS,
        "blend_weights": BLEND_WEIGHTS,
        "calibration": {"method": "platt_sigmoid_on_blend", "a": a10, "b": b10},
        "columns_str": list(X_all_str.columns), "columns_cat": list(X_all_cat.columns),
        "v9_vs_v10_same_carveout": {"v9_calibrated": calib_v9, "v10_calibrated": calib_v10, "delta": calib_v10 - calib_v9},
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_MODEL_DIR}  (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
