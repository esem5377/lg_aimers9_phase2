"""두 가지를 한 번에 실험:
  1) trackman 컨텍스트 피처를 카운트x투타좌우 교차 그룹으로 더 세분화하면 도움되는지
  2) LightGBM 멀티시드 + CatBoost 앙상블이 단일 모델보다 나은지

기존 train.py와 동일한 시간 기반 검증(2019~2023 학습 / 2024 검증), 동일 피처
베이스(trackman context 3종 + raw id 제외)를 그대로 쓰고 위 두 가지만 추가로 비교.
"""
import os

import joblib
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}
METRIC_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]


def build_group_table(df, keys, prefix):
    g = df.groupby(keys, dropna=False)
    out = g[METRIC_COLS].mean()
    out.columns = [f"{prefix}_{c}_mean" for c in out.columns]
    pitch_rate = pd.crosstab([df[k] for k in keys], df["pitch_type_group"], normalize="index")
    for pg in PITCH_GROUPS:
        out[f"{prefix}_{pg}_rate"] = pitch_rate.get(pg, 0.0)
    out[f"{prefix}_n"] = g.size()
    return out.reset_index()


def build_cross_table():
    th = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    th["pitcher_hand"] = th["pitcher_hand"].map(HAND_MAP)
    th["batter_hand"] = th["batter_hand"].map(HAND_MAP)
    keys = ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"]
    table = build_group_table(th, keys, "tk_cross")
    print(f"cross table rows={len(table)}  n stats: min={table['tk_cross_n'].min()} "
          f"median={table['tk_cross_n'].median()} max={table['tk_cross_n'].max()}")
    return {"keys": keys, "table": table}


def load_base(with_cross=False, cross_spec=None):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    if with_cross:
        df = df.merge(cross_spec["table"], on=cross_spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def fit_lgb(X_tr, y_tr, X_va, y_va, seed):
    params = dict(
        objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=63,
        max_depth=-1, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, model.predict_proba(X_va)[:, 1]


def fit_catboost(X_tr, y_tr, X_va, y_va, cat_cols, seed):
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="AUC", random_seed=seed,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
    )
    model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va))
    return model, model.predict_proba(X_va_cb)[:, 1]


def main():
    print("=== Part 1: trackman cross-group feature ===")
    cross_spec = build_cross_table()

    df_base = load_base(with_cross=False)
    train_mask = df_base["season"] <= 2023
    valid_mask = df_base["season"] == 2024
    y_all = df_base[TARGET_COL]

    X_base = build_features(df_base)
    model_base, pred_base = fit_lgb(
        X_base[train_mask], y_all[train_mask], X_base[valid_mask], y_all[valid_mask], seed=42)
    auc_base = roc_auc_score(y_all[valid_mask], pred_base)
    print(f"[no cross] auc={auc_base:.5f}")

    df_cross = load_base(with_cross=True, cross_spec=cross_spec)
    X_cross = build_features(df_cross)
    model_cross, pred_cross = fit_lgb(
        X_cross[train_mask], y_all[train_mask], X_cross[valid_mask], y_all[valid_mask], seed=42)
    auc_cross = roc_auc_score(y_all[valid_mask], pred_cross)
    print(f"[with count x hand cross] auc={auc_cross:.5f}")

    use_cross = auc_cross > auc_base
    print(f"-> cross group {'HELPS' if use_cross else 'does NOT help'}, using "
          f"{'cross' if use_cross else 'base'} feature set for part 2")

    print("\n=== Part 2: ensembling ===")
    X_ens = X_cross if use_cross else X_base
    X_tr, y_tr = X_ens[train_mask], y_all[train_mask]
    X_va, y_va = X_ens[valid_mask], y_all[valid_mask]

    seed_preds = []
    seeds = [42, 1, 7, 123, 2024]
    for s in seeds:
        _, p = fit_lgb(X_tr, y_tr, X_va, y_va, seed=s)
        auc_s = roc_auc_score(y_va, p)
        print(f"[lgb seed={s}] auc={auc_s:.5f}")
        seed_preds.append(p)

    import numpy as np
    bag_pred = np.mean(seed_preds, axis=0)
    auc_bag = roc_auc_score(y_va, bag_pred)
    print(f"[lgb 5-seed bagging] auc={auc_bag:.5f}")

    print("\nFitting CatBoost...")
    _, cb_pred = fit_catboost(X_tr, y_tr, X_va, y_va, CAT_COLS, seed=42)
    auc_cb = roc_auc_score(y_va, cb_pred)
    print(f"[catboost] auc={auc_cb:.5f}")

    blend_pred = 0.5 * bag_pred + 0.5 * cb_pred
    auc_blend = roc_auc_score(y_va, blend_pred)
    print(f"[lgb-bag + catboost 50/50 blend] auc={auc_blend:.5f}")

    print("\n=== summary ===")
    print(f"single lgb (seed42, best feature set): {(auc_cross if use_cross else auc_base):.5f}")
    print(f"lgb 5-seed bagging:                    {auc_bag:.5f}")
    print(f"catboost single:                       {auc_cb:.5f}")
    print(f"lgb-bag + catboost blend:               {auc_blend:.5f}")


if __name__ == "__main__":
    main()
