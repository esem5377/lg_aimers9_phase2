"""calibration carve-out 비율(5%->10%) 검증 -- 시드 배깅(이미 실제 제출로
검증됨: fold0=2390.84, fold2=836.60, 5% carve-out)과 동일한 방식이되
carve-out만 10%로 늘려 비교한다. 데이터가 140만행이라 10%를 떼도 학습에
지장 없고, Platt 보정이 더 큰/안정적인 표본으로 fit되는 효과만 분리해서 본다.

채택 기준: 기존 5% 수치(fold0=2390.84, fold2=836.60)보다 두 축 모두 나아야 함.
로컬(iterations=1000, 3시드, fold0/fold2)로 먼저 확인.
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
RESULT_PATH = os.path.join(OUT_DIR, "carveout_ratio_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
ITERATIONS = 1000
DATA_SPLIT_SEED = 42
SEEDS = [42, 7, 123]
CARVEOUT_FRAC = 0.10  # 기존 0.05 -> 0.10

KNOWN_5PCT = {
    "fold0_2022": 2390.8372673780764,
    "fold2_2024": 836.6047680560929,
}

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

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


def run_context(tag, train_df, eval_df):
    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL]

    train_sub_df, calib_df = train_test_split(
        train_df, test_size=CARVEOUT_FRAC, stratify=y_train_full, random_state=DATA_SPLIT_SEED,
    )
    X_train_sub = build_features(train_sub_df, id_mappings)
    y_train_sub = train_sub_df[TARGET_COL]
    X_calib = build_features(calib_df, id_mappings)
    y_calib = calib_df[TARGET_COL]
    X_eval = build_features(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL]

    cat_idx = [X_train_sub.columns.get_loc(c) for c in CAT_COLS]

    calib_raw_preds = []
    eval_raw_preds = []
    seed_results = []

    for seed in SEEDS:
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
            random_seed=seed, cat_features=cat_idx, early_stopping_rounds=100,
            verbose=False, thread_count=-1, **BEST_PARAMS,
        )
        model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
        elapsed = time.time() - t0

        calib_raw = model.predict_proba(X_calib)[:, 1]
        eval_raw = model.predict_proba(X_eval)[:, 1]
        calib_raw_preds.append(calib_raw)
        eval_raw_preds.append(eval_raw)

        a, b = fit_platt(calib_raw, y_calib)
        eval_calib = apply_platt(eval_raw, a, b)
        seed_results.append({
            "seed": seed,
            "best_iteration": model.get_best_iteration(),
            "auc": roc_auc_score(y_eval, eval_raw),
            "bss_calibrated": bss_score(eval_calib, y_eval),
            "elapsed_sec": elapsed,
        })
        print(f"  [{tag}/seed{seed}] bss_calib={seed_results[-1]['bss_calibrated']:.2f} ({elapsed:.1f}s)", flush=True)

    calib_raw_bagged = np.mean(calib_raw_preds, axis=0)
    eval_raw_bagged = np.mean(eval_raw_preds, axis=0)
    a_bag, b_bag = fit_platt(calib_raw_bagged, y_calib)
    eval_calib_bagged = apply_platt(eval_raw_bagged, a_bag, b_bag)

    bagged_bss = bss_score(eval_calib_bagged, y_eval)
    print(f"  [{tag}/BAGGED_10PCT] bss_calib={bagged_bss:.2f}", flush=True)

    return {"tag": tag, "seeds": seed_results, "bagged_bss_calibrated": bagged_bss}


def main():
    print(f"Load train data + trackman context... (carveout_frac={CARVEOUT_FRAC})", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}

    print("\n=== fold0 (train<=2021 / eval==2022) ===", flush=True)
    all_results["fold0_2022"] = run_context(
        "fold0_2022", df[df["season"] <= 2021], df[df["season"] == 2022]
    )
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== fold2 (train<=2023 / eval==2024) ===", flush=True)
    all_results["fold2_2024"] = run_context(
        "fold2_2024", df[df["season"] <= 2023], df[df["season"] == 2024]
    )
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = {}
    for fold in ["fold0_2022", "fold2_2024"]:
        known = KNOWN_5PCT[fold]
        new = all_results[fold]["bagged_bss_calibrated"]
        summary[fold] = {"known_5pct": known, "carveout_10pct": new, "delta": new - known}
    summary["both_axes_positive"] = all(v["delta"] > 0 for v in summary.values())
    all_results["summary"] = summary

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (5% carve-out -> 10% carve-out, calibrated BSS) ===", flush=True)
    for k, v in summary.items():
        if k == "both_axes_positive":
            continue
        print(f"  {k}: {v['known_5pct']:.2f} -> {v['carveout_10pct']:.2f}  (delta {v['delta']:+.2f})", flush=True)
    print(f"\n  BOTH AXES POSITIVE: {summary['both_axes_positive']}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
