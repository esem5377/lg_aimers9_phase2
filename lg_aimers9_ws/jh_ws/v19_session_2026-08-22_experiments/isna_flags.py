"""catboost_pitch_preprocessing_guide.md 1절 "_isna 결측 여부 플래그" 검증.
974 레시피 그대로, 결측이 발생하는 asof_* 컬럼 16개(8/19 실측 확인:
pitcher 통산 rate 8개=792건, prev1/3/5 rate 6개=29,185건(최다),
batter 통산 rate 2개=830건) 각각에 대해 `{col}_isna`(0/1) 플래그를
추가해 baseline과 비교.

가설: CatBoost는 NaN을 `nan_mode="Min"`으로 이미 "다른 값보다 작은
센티널"로 자동 구분해서 split할 수 있으므로(8/19 확인), 이 플래그가 주는
정보가 원본 컬럼에 이미 다 들어있어 중복/무의미할 가능성이 높음 -- 그래도
직접 검증되지 않았던 항목이라 확인.
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

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "isna_flags_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
ITERATIONS = 1000
SEED = 42

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

ISNA_SOURCE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


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


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_isna_flags(df):
    df = df.copy()
    for c in ISNA_SOURCE_COLS:
        df[f"{c}_isna"] = df[c].isna().astype(int)
    return df


def build_id_mappings(train_df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(train_df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def run_variant(tag, train_df, eval_df, use_isna):
    if use_isna:
        train_df = add_isna_flags(train_df)
        eval_df = add_isna_flags(eval_df)

    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL]

    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    X_train_sub = build_features(train_sub_df, id_mappings)
    y_train_sub = train_sub_df[TARGET_COL]
    X_calib = build_features(calib_df, id_mappings)
    y_calib = calib_df[TARGET_COL]
    X_eval = build_features(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL]

    cat_idx = [X_train_sub.columns.get_loc(c) for c in CAT_COLS]

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **BEST_PARAMS,
    )
    model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = model.predict_proba(X_eval)[:, 1]
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag, "n_features": X_train_sub.shape[1],
        "best_iteration": model.get_best_iteration(),
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    print(
        f"  [{tag}] n_features={result['n_features']} auc={result['auc']:.4f} "
        f"bss_calib={result['bss_calibrated']:.2f} ({elapsed:.1f}s)",
        flush=True,
    )
    return result


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}
    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    for fold_name, (train_df, eval_df) in fold_specs.items():
        print(f"\n=== {fold_name} ===", flush=True)
        baseline_r = run_variant(f"{fold_name}/baseline", train_df, eval_df, use_isna=False)
        isna_r = run_variant(f"{fold_name}/+isna_flags", train_df, eval_df, use_isna=True)
        all_results[fold_name] = {
            "baseline": baseline_r,
            "with_isna_flags": isna_r,
            "delta_calibrated": isna_r["bss_calibrated"] - baseline_r["bss_calibrated"],
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    all_positive = all(v["delta_calibrated"] > 0 for v in all_results.values())
    summary = {k: v["delta_calibrated"] for k, v in all_results.items()}
    summary["all_axes_positive"] = all_positive
    all_results["summary"] = summary

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (baseline -> +isna_flags, calibrated BSS) ===", flush=True)
    for k, v in all_results.items():
        if k == "summary":
            continue
        print(f"  {k}: {v['baseline']['bss_calibrated']:.2f} -> {v['with_isna_flags']['bss_calibrated']:.2f}  (delta {v['delta_calibrated']:+.2f})", flush=True)
    print(f"\n  ALL AXES POSITIVE: {all_positive}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
