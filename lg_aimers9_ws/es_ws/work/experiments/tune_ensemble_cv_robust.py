"""tune_ensemble_calibrated.py / train_ensemble_calibrated.py에서 3모델 블렌드가
CatBoost 단일+보정 대비 보인 효과(+2.16)가 5% carve-out 재검증에서는 반대
부호(-2.46)로 나왔다. 8/16 tune_cv_robust.py와 동일한 문제(단일 홀드아웃
선택 편향)로 의심되어, 같은 해법(3-fold walk-forward)으로 "보정"한다:
폴드 전부에서 방향이 일관돼야만 진짜 신호로 취급.

FOLDS (tune_cv_robust.py와 동일):
  fold0: train 2019~2021 -> valid 2022
  fold1: train 2019~2022 -> valid 2023
  fold2: train 2019~2023 -> valid 2024 (기존 실험과 동일 폴드)

각 폴드에서 CatBoost/LightGBM/XGBoost를 학습 -> Platt 보정 -> 그리드 서치로
블렌드 가중치 최적화한 뒤, "CatBoost 단일+보정" vs "3모델 블렌드+보정"의
BSS를 비교한다. 지표는 BSS (AUC는 보정 효과를 못 잡으므로 비교 기준에서 제외).
"""
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

CB_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
LGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", tree_method="hist",
)

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold1(train<=2022,valid2023)", "train_seasons": [2019, 2020, 2021, 2022], "valid_season": 2023},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def load_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    return df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])


def fit_catboost(X_tr, y_tr, X_va, y_va):
    Xtr, Xva = X_tr.copy(), X_va.copy()
    for c in CAT_COLS:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    cat_idx = [Xtr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, **CB_PARAMS,
    )
    model.fit(Xtr, y_tr, eval_set=(Xva, y_va))
    return model.predict_proba(Xva)[:, 1]


def fit_lgbm(X_tr, y_tr, X_va, y_va):
    Xtr, Xva = X_tr.copy(), X_va.copy()
    for c in CAT_COLS:
        Xtr[c] = Xtr[c].astype("category")
        Xva[c] = Xva[c].astype("category").cat.set_categories(Xtr[c].cat.categories)
    model = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1, **LGB_PARAMS)
    model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model.predict_proba(Xva)[:, 1]


def fit_xgb(X_tr, y_tr, X_va, y_va):
    Xtr, Xva = X_tr.copy(), X_va.copy()
    for c in CAT_COLS:
        Xtr[c] = Xtr[c].astype("category")
        Xva[c] = Xva[c].astype("category").cat.set_categories(Xtr[c].cat.categories)
    model = XGBClassifier(enable_categorical=True, early_stopping_rounds=100, **XGB_PARAMS)
    model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
    return model.predict_proba(Xva)[:, 1]


def optimize_weights_grid(preds, y, step=0.05):
    best = None
    ws = np.arange(0.0, 1.0001, step)
    for w_cb in ws:
        for w_lgb in np.arange(0.0, 1.0001 - w_cb, step):
            w_xgb = max(1.0 - w_cb - w_lgb, 0.0)
            blend = np.clip(preds["cb"] * w_cb + preds["lgb"] * w_lgb + preds["xgb"] * w_xgb, 1e-6, 1 - 1e-6)
            s = bss_score(blend, y)
            if best is None or s > best[0]:
                best = (s, {"cb": float(w_cb), "lgb": float(w_lgb), "xgb": float(w_xgb)})
    return best[1], best[0]


def run_fold(fold, X_all, y_all, df):
    tr_mask = df["season"].isin(fold["train_seasons"])
    va_mask = df["season"] == fold["valid_season"]
    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va = X_all[va_mask], y_all[va_mask]

    t0 = time.time()
    cb_raw = fit_catboost(X_tr, y_tr, X_va, y_va)
    lgb_raw = fit_lgbm(X_tr, y_tr, X_va, y_va)
    xgb_raw = fit_xgb(X_tr, y_tr, X_va, y_va)
    dt = time.time() - t0

    cb_calib = apply_platt(cb_raw, fit_platt(cb_raw, y_va))
    lgb_calib = apply_platt(lgb_raw, fit_platt(lgb_raw, y_va))
    xgb_calib = apply_platt(xgb_raw, fit_platt(xgb_raw, y_va))

    cb_solo_bss = bss_score(cb_calib, y_va)
    weights, blend_bss = optimize_weights_grid({"cb": cb_calib, "lgb": lgb_calib, "xgb": xgb_calib}, y_va)

    print(f"  [{fold['name']}] ({dt:.0f}s) n_train={tr_mask.sum()} n_valid={va_mask.sum()}")
    print(f"    catboost단일+보정 bss={cb_solo_bss:.2f}  "
          f"3모델블렌드+보정 bss={blend_bss:.2f} (weights={weights})  delta={blend_bss - cb_solo_bss:+.2f}")

    return {
        "fold": fold["name"],
        "cb_solo_calibrated_bss": cb_solo_bss,
        "blend_calibrated_bss": blend_bss,
        "blend_weights": weights,
        "delta": blend_bss - cb_solo_bss,
    }


def main():
    print("Load data...")
    df = load_base_df()
    y_all = df[TARGET_COL]
    X_all = build_features(df)
    print(f" shape={df.shape}")

    print("\n3-fold walk-forward: CatBoost단일+보정 vs 3모델블렌드+보정...")
    results = [run_fold(f, X_all, y_all, df) for f in FOLDS]

    deltas = [r["delta"] for r in results]
    all_positive = all(d > 0 for d in deltas)
    all_negative = all(d < 0 for d in deltas)
    print(f"\n=== 요약 ===")
    for r in results:
        print(f"  {r['fold']}: delta={r['delta']:+.2f}")
    print(f"  모든 폴드에서 블렌드가 이김={all_positive}  모든 폴드에서 블렌드가 짐={all_negative}")
    print(f"  mean delta = {sum(deltas) / len(deltas):+.2f}")
    if not all_positive and not all_negative:
        print("  -> 부호가 폴드마다 다름: 노이즈로 판단, 앙상블 추가 이득 신뢰 불가")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_ensemble_cv_robust_results.json"), "w") as f:
        json.dump({
            "folds": results,
            "all_positive": all_positive,
            "all_negative": all_negative,
            "mean_delta": sum(deltas) / len(deltas),
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_ensemble_cv_robust_results.json")


if __name__ == "__main__":
    main()
