"""raw pitcher_id/batter_id(수치형, jh_ws 방식) + 보정의 효과를 3-fold
walk-forward로 재검증. `tune_rawid_numeric_calibrated.py`(단일 2024
홀드아웃)에서는 raw id 포함이 보정 후 BSS +9.48이었는데, `tune_ensemble_cv_robust.py`
에서 단일 홀드아웃 델타(+2.15)가 3-fold 평균(+1.13, 폴드 간 0~+2.15로 편차)보다
과장돼 있었던 전례가 있어 동일한 방법으로 검증한다.

FOLDS (tune_cv_robust.py / tune_ensemble_cv_robust.py와 동일):
  fold0: train 2019~2021 -> valid 2022
  fold1: train 2019~2022 -> valid 2023
  fold2: train 2019~2023 -> valid 2024

각 폴드에서 "raw id 제외 + 보정" vs "raw id 포함(수치형) + 보정"의 BSS를 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

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

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
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


def build_features(df, with_raw_id):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    if not with_raw_id:
        X = X.drop(columns=["pitcher_id", "batter_id"])
    else:
        for c in ["pitcher_id", "batter_id"]:
            uniq = sorted(X[c].astype(str).unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            X[c] = X[c].astype(str).map(mapping).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def fit_catboost_calibrated(X_tr, y_tr, X_va, y_va):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, **BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    raw_pred = model.predict_proba(X_va)[:, 1]
    calib_pred = apply_platt(raw_pred, fit_platt(raw_pred, y_va))
    return bss_score(raw_pred, y_va), bss_score(calib_pred, y_va)


def run_fold(fold, df):
    tr_mask = df["season"].isin(fold["train_seasons"])
    va_mask = df["season"] == fold["valid_season"]
    y_tr, y_va = df.loc[tr_mask, TARGET_COL], df.loc[va_mask, TARGET_COL]

    t0 = time.time()
    X_no_id = build_features(df, with_raw_id=False)
    bss_raw_no_id, bss_calib_no_id = fit_catboost_calibrated(
        X_no_id[tr_mask], y_tr, X_no_id[va_mask], y_va)

    X_with_id = build_features(df, with_raw_id=True)
    bss_raw_with_id, bss_calib_with_id = fit_catboost_calibrated(
        X_with_id[tr_mask], y_tr, X_with_id[va_mask], y_va)
    dt = time.time() - t0

    delta_calib = bss_calib_with_id - bss_calib_no_id
    print(f"  [{fold['name']}] ({dt:.0f}s) n_train={tr_mask.sum()} n_valid={va_mask.sum()}")
    print(f"    no_id:   raw={bss_raw_no_id:.2f}  calib={bss_calib_no_id:.2f}")
    print(f"    with_id: raw={bss_raw_with_id:.2f}  calib={bss_calib_with_id:.2f}")
    print(f"    delta(calibrated, with_id - no_id) = {delta_calib:+.2f}")

    return {
        "fold": fold["name"],
        "bss_raw_no_id": bss_raw_no_id, "bss_calib_no_id": bss_calib_no_id,
        "bss_raw_with_id": bss_raw_with_id, "bss_calib_with_id": bss_calib_with_id,
        "delta_calibrated": delta_calib,
    }


def main():
    print("Load data...")
    df = load_base_df()
    print(f" shape={df.shape}")

    print("\n3-fold walk-forward: raw id 제외+보정 vs raw id 포함+보정...")
    results = [run_fold(f, df) for f in FOLDS]

    deltas = [r["delta_calibrated"] for r in results]
    all_positive = all(d > 0 for d in deltas)
    all_negative = all(d < 0 for d in deltas)
    print(f"\n=== 요약 ===")
    for r in results:
        print(f"  {r['fold']}: delta={r['delta_calibrated']:+.2f}")
    print(f"  모든 폴드에서 raw id 포함이 이김={all_positive}  모든 폴드에서 짐={all_negative}")
    print(f"  mean delta = {sum(deltas) / len(deltas):+.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_rawid_cv_robust_results.json"), "w") as f:
        json.dump({
            "folds": results,
            "all_positive": all_positive,
            "all_negative": all_negative,
            "mean_delta": sum(deltas) / len(deltas),
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_rawid_cv_robust_results.json")


if __name__ == "__main__":
    main()
