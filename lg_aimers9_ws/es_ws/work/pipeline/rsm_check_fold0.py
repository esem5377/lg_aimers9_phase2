"""diversity_screen_walkforward_1seed.py의 Task B(rsm)를 fold0(train<=2021 -> eval=2022)로
재검증. fold2(train<=2023 -> eval=2024)에서만 확인된 rsm=0.8 이득(+11.53 vs no-rsm,
CPU/GPU 교란 배제 후 +9.54)이 다른 fold에서도 재현되는지 확인 -- v13이 fold2 단독
검증만으로 제출했다가 -14로 반증된 것과 같은 함정을 피하기 위함.
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
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/rsm_check_fold0.json"

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
CAT_TASK_TYPE = "GPU"
RSM = 0.8

TRAIN_SEASON_MAX = 2021
EVAL_SEASON = 2022


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


def train_one(seed, X_train, y_train, cat_idx, extra_params=None):
    params = dict(CAT_BEST_PARAMS)
    task_type = CAT_TASK_TYPE
    if extra_params:
        params.update(extra_params)
        if "rsm" in extra_params:
            task_type = "CPU"
    kwargs = dict(
        iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=seed,
        cat_features=cat_idx, verbose=False, task_type=task_type, **params,
    )
    if task_type == "GPU":
        kwargs["devices"] = "0"
    else:
        kwargs["thread_count"] = -1
    t0 = time.time()
    m = CatBoostClassifier(**kwargs)
    m.fit(X_train, y_train)
    log(f"    seed={seed} extra={extra_params} task_type={task_type} done ({time.time()-t0:.0f}s)")
    return m


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
    log(f"shape={df.shape}")

    train_df = df[df["season"] <= TRAIN_SEASON_MAX]
    eval_df = df[df["season"] == EVAL_SEASON]
    log(f"train(<={TRAIN_SEASON_MAX})={len(train_df)} eval({EVAL_SEASON})={len(eval_df)}")

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

    result = {"train_season_max": TRAIN_SEASON_MAX, "eval_season": EVAL_SEASON}

    log(f"\n=== fold0: CatBoost 1seed no-rsm baseline ===")
    m_norsm = train_one(CB_SEED, X_train_cb, y_train_cb, cat_idx)
    final_calib = m_norsm.predict_proba(X_calib_cb)[:, 1]
    final_eval = m_norsm.predict_proba(X_eval_cb)[:, 1]
    bss_baseline, auc_baseline = calib_and_score(final_calib, y_calib, final_eval, y_eval)
    log(f"  baseline(no-rsm): auc={auc_baseline:.4f} bss_calib={bss_baseline:.2f}")

    log(f"\n=== fold0: CatBoost 1seed rsm={RSM} ===")
    m_rsm = train_one(CB_SEED, X_train_cb, y_train_cb, cat_idx, extra_params={"rsm": RSM})
    rsm_calib = m_rsm.predict_proba(X_calib_cb)[:, 1]
    rsm_eval = m_rsm.predict_proba(X_eval_cb)[:, 1]
    bss_rsm, auc_rsm = calib_and_score(rsm_calib, y_calib, rsm_eval, y_eval)
    log(f"  rsm-only: auc={auc_rsm:.4f} bss_calib={bss_rsm:.2f}  delta_vs_baseline={bss_rsm-bss_baseline:+.2f}")

    combined_calib = np.mean([final_calib, rsm_calib], axis=0)
    combined_eval = np.mean([final_eval, rsm_eval], axis=0)
    bss_combined, auc_combined = calib_and_score(combined_calib, y_calib, combined_eval, y_eval)
    log(f"  combined(no-rsm + rsm, 2-way mean): auc={auc_combined:.4f} bss_calib={bss_combined:.2f}  delta_vs_baseline={bss_combined-bss_baseline:+.2f}")

    result["baseline"] = {"auc": auc_baseline, "bss_calib": bss_baseline}
    result["rsm_only"] = {"auc": auc_rsm, "bss_calib": bss_rsm, "delta_vs_baseline": bss_rsm - bss_baseline}
    result["combined_2way"] = {"auc": auc_combined, "bss_calib": bss_combined, "delta_vs_baseline": bss_combined - bss_baseline}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
