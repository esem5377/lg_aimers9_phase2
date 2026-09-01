"""CatBoost 손실함수를 Logloss(cross-entropy) -> RMSE로 바꿔서 fold0/fold2 walk-forward
비교. 대회 채점 지표가 Brier Score(=확률 예측과 0/1 라벨의 MSE)인데 지금까지 전부
Logloss로 학습해왔음 -- 손실함수와 평가지표가 처음부터 어긋나 있었을 가능성. RMSE 손실은
CatBoostRegressor로 학습(타겟 0/1을 그대로 회귀), Platt으로 보정 후 BSS 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/rmse_loss_check.json"

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

CB_SEED = 42
CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
CAT_ITERATIONS = 2000

FOLDS = [
    {"name": "fold0", "train_max": 2021, "eval": 2022},
    {"name": "fold2", "train_max": 2023, "eval": 2024},
]


def log(msg):
    print(msg, flush=True)


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


def build_id_mappings(df):
    return {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in RAW_ID_COLS}


def build_catboost_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def train_logloss(seed, X_train, y_train, cat_idx):
    t0 = time.time()
    m = CatBoostClassifier(
        iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=seed,
        cat_features=cat_idx, verbose=False, task_type="GPU", devices="0",
        **CAT_BEST_PARAMS,
    )
    m.fit(X_train, y_train)
    log(f"    Logloss seed={seed} done ({time.time()-t0:.0f}s)")
    return m


def train_rmse(seed, X_train, y_train, cat_idx):
    t0 = time.time()
    m = CatBoostRegressor(
        iterations=CAT_ITERATIONS, loss_function="RMSE", random_seed=seed,
        cat_features=cat_idx, verbose=False, task_type="GPU", devices="0",
        **CAT_BEST_PARAMS,
    )
    m.fit(X_train, y_train)
    log(f"    RMSE seed={seed} done ({time.time()-t0:.0f}s)")
    return m


def calib_and_score(raw_calib, y_calib, raw_eval, y_eval):
    a, b = fit_platt(raw_calib, y_calib)
    eval_pred = apply_platt(raw_eval, a, b)
    return bss_score(eval_pred, y_eval), roc_auc_score(y_eval, raw_eval)


def run_fold(df, fold):
    train_max, eval_season, name = fold["train_max"], fold["eval"], fold["name"]
    train_df = df[df["season"] <= train_max]
    eval_df = df[df["season"] == eval_season]
    log(f"\n### {name}: train(<={train_max})={len(train_df)} eval({eval_season})={len(eval_df)}")

    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)
    train_sub = train_sub.reset_index(drop=True)
    calib = calib.reset_index(drop=True)
    y_calib = calib[TARGET_COL].values
    y_eval = eval_df[TARGET_COL].values

    id_mappings = build_id_mappings(train_sub)
    X_train_cb = build_catboost_features(train_sub, id_mappings)
    X_calib_cb = build_catboost_features(calib, id_mappings)
    X_eval_cb = build_catboost_features(eval_df, id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]
    y_train_cb = train_sub[TARGET_COL]

    log(f"  --- Logloss baseline ---")
    m_ll = train_logloss(CB_SEED, X_train_cb, y_train_cb, cat_idx)
    ll_calib = m_ll.predict_proba(X_calib_cb)[:, 1]
    ll_eval = m_ll.predict_proba(X_eval_cb)[:, 1]
    bss_ll, auc_ll = calib_and_score(ll_calib, y_calib, ll_eval, y_eval)
    log(f"  Logloss: auc={auc_ll:.4f} bss_calib={bss_ll:.2f}")

    log(f"  --- RMSE ---")
    m_rmse = train_rmse(CB_SEED, X_train_cb, y_train_cb, cat_idx)
    rmse_calib = m_rmse.predict(X_calib_cb)
    rmse_eval = m_rmse.predict(X_eval_cb)
    bss_rmse, auc_rmse = calib_and_score(rmse_calib, y_calib, rmse_eval, y_eval)
    log(f"  RMSE: auc={auc_rmse:.4f} bss_calib={bss_rmse:.2f}  delta_vs_logloss={bss_rmse-bss_ll:+.2f}")

    log(f"  --- 2-way mean(Logloss + RMSE, each Platt-calibrated first) ---")
    a_ll, b_ll = fit_platt(ll_calib, y_calib)
    a_rmse, b_rmse = fit_platt(rmse_calib, y_calib)
    combined_calib = np.mean([apply_platt(ll_calib, a_ll, b_ll), apply_platt(rmse_calib, a_rmse, b_rmse)], axis=0)
    combined_eval = np.mean([apply_platt(ll_eval, a_ll, b_ll), apply_platt(rmse_eval, a_rmse, b_rmse)], axis=0)
    bss_comb = bss_score(combined_eval, y_eval)
    log(f"  combined: bss={bss_comb:.2f}  delta_vs_logloss={bss_comb-bss_ll:+.2f}")

    return {
        "fold": name, "train_max": train_max, "eval_season": eval_season,
        "logloss": {"auc": auc_ll, "bss_calib": bss_ll},
        "rmse": {"auc": auc_rmse, "bss_calib": bss_rmse, "delta_vs_logloss": bss_rmse - bss_ll},
        "combined": {"bss_calib": bss_comb, "delta_vs_logloss": bss_comb - bss_ll},
    }


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df = add_risk_score_drop_ingredients(df)
    log(f"shape={df.shape}")

    results = [run_fold(df, fold) for fold in FOLDS]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nSaved: {OUT_PATH}")
    log("\n=== SUMMARY ===")
    for r in results:
        log(f"{r['fold']}: RMSE_delta={r['rmse']['delta_vs_logloss']:+.2f}  combined_delta={r['combined']['delta_vs_logloss']:+.2f}")


if __name__ == "__main__":
    main()
