"""diversity_screen_walkforward_1seed.py의 rsm=0.8 결과(+12.67)가 진짜 rsm 효과인지,
아니면 GPU->CPU 전환 자체(CatBoost GPU/CPU 구현 차이) 때문인지 분리하기 위한 대조군.
동일 fold(train<=2023->eval=2024), 동일 seed=42, CPU에서 rsm 없이 학습해 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/rsm_confound_check.json"

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
CAT_ITERATIONS = 2000


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


def calib_and_score(raw_calib, y_calib, raw_eval, y_eval):
    a, b = fit_platt(raw_calib, y_calib)
    eval_pred = apply_platt(raw_eval, a, b)
    return bss_score(eval_pred, y_eval), roc_auc_score(y_eval, raw_eval)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df = add_risk_score_drop_ingredients(df)

    train_df = df[df["season"] <= 2023]
    eval_df = df[df["season"] == 2024]

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

    log("=== CatBoost 1seed, CPU, no rsm (confound check vs GPU baseline 824.09 / rsm=0.8 835.62) ===")
    t0 = time.time()
    m = CatBoostClassifier(
        iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=SEED,
        cat_features=cat_idx, verbose=False, task_type="CPU", thread_count=-1,
        **CAT_BEST_PARAMS,
    )
    m.fit(X_train_cb, y_train_cb)
    log(f"  done ({time.time()-t0:.0f}s)")

    calib_p = m.predict_proba(X_calib_cb)[:, 1]
    eval_p = m.predict_proba(X_eval_cb)[:, 1]
    bss, auc = calib_and_score(calib_p, y_calib, eval_p, y_eval)
    log(f"  CPU no-rsm: auc={auc:.4f} bss_calib={bss:.2f}")

    result = {"cpu_norsm": {"auc": auc, "bss_calib": bss},
              "gpu_norsm_baseline": 824.09, "cpu_rsm08": 835.62}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
