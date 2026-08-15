"""실험: LightGBM(5시드 배깅) + CatBoost 블렌드 가중치를 정밀 탐색.

tune_v2.py에서는 50/50 고정 블렌드만 시도해 CatBoost 단독 대비 +0.00014로
노이즈 수준이었다. 이번엔 각 모델을 한 번씩만 학습해두고(둘 다 기존 튜닝에서
찾은 최선 하이퍼파라미터 사용) 블렌드 가중치 w를 0~1 사이 0.02 간격으로
스윕해서 최적 지점을 찾는다. 모델 재학습 없이 예측 확률만 섞으므로 스윕
자체는 빠르다.

- LightGBM: work/tune.py 랜덤서치 1위 파라미터 (auc=0.54907)
- CatBoost: work/tune_catboost.py 랜덤서치 1위 파라미터 (auc=0.55056, 현재 채택)
- 5시드 배깅도 비교 대상에 포함 (LightGBM 단일 vs 배깅).
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

LGB_BEST_PARAMS = dict(
    learning_rate=0.01, num_leaves=63, min_child_samples=500,
    subsample=0.6, colsample_bytree=0.5, reg_lambda=1.0, reg_alpha=0.5,
)
CB_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
LGB_SEEDS = [42, 1, 7, 123, 2024]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def fit_lgb(X_tr, y_tr, X_va, y_va, seed):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category").cat.set_categories(X_tr[c].cat.categories)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=2000, random_state=seed, n_jobs=-1,
        **LGB_BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model.predict_proba(X_va)[:, 1]


def fit_catboost(X_tr, y_tr, X_va, y_va, seed):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=seed,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=200,
        **CB_BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return model.predict_proba(X_va)[:, 1]


def main():
    print("Load data...")
    X_tr, y_tr, X_va, y_va = load_data()
    print(f" train={X_tr.shape} valid={X_va.shape}")

    print("\n=== LightGBM (5-seed bagging) ===")
    lgb_preds = []
    for s in LGB_SEEDS:
        p = fit_lgb(X_tr, y_tr, X_va, y_va, seed=s)
        auc_s = roc_auc_score(y_va, p)
        print(f"[lgb seed={s}] auc={auc_s:.5f}")
        lgb_preds.append(p)
    lgb_single = lgb_preds[0]
    lgb_bag = np.mean(lgb_preds, axis=0)
    auc_lgb_single = roc_auc_score(y_va, lgb_single)
    auc_lgb_bag = roc_auc_score(y_va, lgb_bag)
    print(f"lgb single(seed42) auc={auc_lgb_single:.5f}  lgb 5-seed bag auc={auc_lgb_bag:.5f}")

    print("\n=== CatBoost ===")
    cb_pred = fit_catboost(X_tr, y_tr, X_va, y_va, seed=42)
    auc_cb = roc_auc_score(y_va, cb_pred)
    print(f"catboost auc={auc_cb:.5f}")

    print("\n=== Blend weight sweep (w * catboost + (1-w) * lgb_bag) ===")
    results = []
    for w in np.arange(0.0, 1.0001, 0.02):
        blend = w * cb_pred + (1 - w) * lgb_bag
        auc_b = roc_auc_score(y_va, blend)
        results.append({"weight_catboost": round(float(w), 2), "auc": auc_b})
    res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(res_df.head(15).to_string(index=False))

    best = res_df.iloc[0].to_dict()
    print("\n=== summary ===")
    print(f"lgb single:            {auc_lgb_single:.5f}")
    print(f"lgb 5-seed bag:        {auc_lgb_bag:.5f}")
    print(f"catboost single:       {auc_cb:.5f}  (현재 채택 버전과 동일 세팅)")
    print(f"best blend (w={best['weight_catboost']}): {best['auc']:.5f}")
    print(f"delta vs catboost alone: {best['auc'] - auc_cb:+.5f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(OUT_DIR, "tune_blend_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "tune_blend_summary.json"), "w") as f:
        json.dump({
            "auc_lgb_single": auc_lgb_single,
            "auc_lgb_bag": auc_lgb_bag,
            "auc_catboost": auc_cb,
            "best_blend_weight_catboost": best["weight_catboost"],
            "best_blend_auc": best["auc"],
            "delta_vs_catboost": best["auc"] - auc_cb,
        }, f, indent=2)


if __name__ == "__main__":
    main()
