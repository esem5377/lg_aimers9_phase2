"""tune_arch_blend_walkforward.py 결과(fold0 +5.42, fold2 +2.08, 둘 다 양수)의
낙관편향 재검증 -- 가중치를 eval셋에 직접 fit한 게 아니라, 프로젝트 관례
(tune_isotonic_calibration.py / tune_segment_platt.py)대로 각 fold의
valid_season을 calib_fit/calib_eval로 쪼개 calib_fit에서만 Platt+블렌드
가중치를 고르고 calib_eval에서 평가한다.

모델 학습(각 fold의 CatBoost/LGBM/XGBoost fit, early stopping은 valid_season
전체로 -- 프로젝트 전체에서 계속 써온 관례라 이번에도 유지) 자체는 이전
스크립트와 동일하게 반복한다(재사용 캐시 없음, 5분 안팎이라 재실행 비용 낮음).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
MODEL_DIR = os.path.join(_BASE, "model")
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

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]
CALIB_FIT_SIZE = 0.5


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
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_features(df, id_mappings, cat_dtype):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        if cat_dtype == "str":
            X[c] = X[c].astype(str)
        else:
            X[c] = X[c].astype(str).astype("category")
    return X


def grid_search_blend(preds_calib_fit, y_fit):
    best = {"bss": -1, "w": None}
    step = 0.05
    n_steps = int(round(1 / step))
    for i in range(n_steps + 1):
        w_cat = i * step
        for j in range(n_steps + 1 - i):
            w_lgb = j * step
            w_xgb = 1.0 - w_cat - w_lgb
            if w_xgb < -1e-9:
                continue
            blend = w_cat * preds_calib_fit["cat"] + w_lgb * preds_calib_fit["lgb"] + w_xgb * preds_calib_fit["xgb"]
            bss = bss_score(blend, y_fit)
            if bss > best["bss"]:
                best = {"bss": bss, "w": {"cat": round(w_cat, 2), "lgb": round(w_lgb, 2), "xgb": round(w_xgb, 2)}}
    return best["w"]


def run_fold(fold, df, id_mappings):
    t0 = time.time()
    train_mask = df["season"].isin(fold["train_seasons"])
    valid_mask = df["season"] == fold["valid_season"]
    y_all = df[TARGET_COL]
    y_tr = y_all[train_mask]

    print(f"\n===== {fold['name']} n_train={train_mask.sum()} n_valid={valid_mask.sum()} =====")

    # calib_fit / calib_eval 분리 (valid_season 내부, stratified)
    va_idx = df.index[valid_mask]
    y_va_all = y_all.loc[va_idx]
    idx_fit, idx_eval = train_test_split(
        va_idx, train_size=CALIB_FIT_SIZE, stratify=y_va_all, random_state=42)
    print(f" calib_fit={len(idx_fit)}  calib_eval={len(idx_eval)}")

    # ---- CatBoost ----
    X_all_cb = build_features(df, id_mappings, "str")
    X_tr_cb, X_va_cb = X_all_cb.loc[train_mask], X_all_cb.loc[valid_mask]
    y_va = y_all.loc[valid_mask]
    cat_idx = [X_all_cb.columns.get_loc(c) for c in CAT_COLS]
    print(f" [CatBoost] fit... ({time.time()-t0:.0f}s elapsed)")
    cat_model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **CAT_BEST_PARAMS,
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va))
    cat_raw_all = pd.Series(cat_model.predict_proba(X_va_cb)[:, 1], index=X_va_cb.index)
    print(f"  best_iter={cat_model.get_best_iteration()} auc={roc_auc_score(y_va, cat_raw_all):.5f} ({time.time()-t0:.0f}s)")

    # ---- LightGBM ----
    X_all_lgb = build_features(df, id_mappings, "category")
    X_tr_lgb, X_va_lgb = X_all_lgb.loc[train_mask], X_all_lgb.loc[valid_mask]
    print(f" [LightGBM] fit... ({time.time()-t0:.0f}s elapsed)")
    lgb_model = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(
        X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)],
        categorical_feature=CAT_COLS,
        callbacks=[lgb_early_stopping(100, verbose=False)],
    )
    lgb_raw_all = pd.Series(lgb_model.predict_proba(X_va_lgb)[:, 1], index=X_va_lgb.index)
    print(f"  best_iter={lgb_model.best_iteration_} auc={roc_auc_score(y_va, lgb_raw_all):.5f} ({time.time()-t0:.0f}s)")

    # ---- XGBoost ----
    X_all_xgb = build_features(df, id_mappings, "category")
    X_tr_xgb, X_va_xgb = X_all_xgb.loc[train_mask], X_all_xgb.loc[valid_mask]
    print(f" [XGBoost] fit... ({time.time()-t0:.0f}s elapsed)")
    xgb_model = XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=42, early_stopping_rounds=100,
        eval_metric="auc", n_jobs=-1,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    xgb_raw_all = pd.Series(xgb_model.predict_proba(X_va_xgb)[:, 1], index=X_va_xgb.index)
    print(f"  best_iter={xgb_model.best_iteration} auc={roc_auc_score(y_va, xgb_raw_all):.5f} ({time.time()-t0:.0f}s)")

    raw_all = {"cat": cat_raw_all, "lgb": lgb_raw_all, "xgb": xgb_raw_all}

    # ---- calib_fit: Platt + 블렌드 가중치 선택 ----
    y_fit = y_all.loc[idx_fit]
    ab = {}
    calib_fit_preds = {}
    for name, raw in raw_all.items():
        raw_fit = raw.loc[idx_fit]
        a, b = fit_platt_scaling(raw_fit, y_fit)
        ab[name] = (a, b)
        calib_fit_preds[name] = apply_platt_scaling(raw_fit, a, b)
    best_w = grid_search_blend(calib_fit_preds, y_fit)

    # ---- calib_eval: 진짜 홀드아웃에서 평가 ----
    y_eval = y_all.loc[idx_eval]
    calib_eval_preds = {}
    for name, raw in raw_all.items():
        raw_eval = raw.loc[idx_eval]
        a, b = ab[name]
        calib_eval_preds[name] = apply_platt_scaling(raw_eval, a, b)

    single_bss_eval = {k: bss_score(v, y_eval) for k, v in calib_eval_preds.items()}
    single_best_name = max(single_bss_eval, key=single_bss_eval.get)
    single_best_bss = single_bss_eval[single_best_name]

    blend_eval = (
        best_w["cat"] * calib_eval_preds["cat"]
        + best_w["lgb"] * calib_eval_preds["lgb"]
        + best_w["xgb"] * calib_eval_preds["xgb"]
    )
    blend_bss_eval = bss_score(blend_eval, y_eval)
    delta = blend_bss_eval - single_best_bss
    dt = time.time() - t0

    print(f" calib_fit에서 고른 블렌드 가중치 = {best_w}")
    print(f" calib_eval(진짜 홀드아웃): 단일최고({single_best_name})={single_best_bss:.2f}  블렌드={blend_bss_eval:.2f}  delta={delta:+.2f}")
    print(f" (fold 소요 {dt:.0f}s)")

    return {
        "fold": fold["name"],
        "blend_weights_from_calib_fit": best_w,
        "single_bss_eval": single_bss_eval,
        "single_best_name": single_best_name,
        "single_best_bss_eval": single_best_bss,
        "blend_bss_eval": blend_bss_eval,
        "delta_honest": delta,
        "elapsed_sec": dt,
    }


def main():
    t0 = time.time()
    print("Load data...")
    df = load_data()
    print(f" shape={df.shape}")
    id_mappings = build_id_mappings(df)

    results = [run_fold(f, df, id_mappings) for f in FOLDS]

    print("\n=== 요약 (honest delta = calib_eval에서 블렌드 - 단일최고) ===")
    for r in results:
        print(f"  {r['fold']}: delta={r['delta_honest']:+.2f}  weights={r['blend_weights_from_calib_fit']}")
    both_positive = all(r["delta_honest"] > 0 for r in results)
    print(f"  fold0 & fold2 모두 양수 = {both_positive}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_arch_blend_honest_results.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": results, "both_positive": both_positive, "total_elapsed_sec": time.time() - t0}, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_arch_blend_honest_results.json")


if __name__ == "__main__":
    main()
