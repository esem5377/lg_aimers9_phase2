"""raw pitcher_id/batter_id(수치형, label-encoded, jh_ws 방식)를 "보정된"
파이프라인 위에서 재검증.

`tune_catboost_rawid_numeric.py`(8/18)에서 이 방식으로 AUC +0.00033(경계선
노이즈)을 확인했지만, 그건 보정 적용 *전* 지표였다. 오늘 확률보정 실험에서
확인했듯 sigmoid 보정은 AUC(순위)엔 안 잡히고 BSS(대회 실제 채점 지표)에만
잡히는 효과를 낼 수 있다 — 즉 raw id의 AUC상 이득이 작아 보여도, 보정 이후
BSS 기준으로는 유의미한 차이를 만들 수 있는지 다시 봐야 한다.

train_catboost.py(현재 프로덕션, raw id 제외+보정, 로컬 BSS 825.57, 리더보드
959 실측)와 동일 피처·동일 시간분할·동일 보정 방식(Platt scaling)에 raw
pitcher_id/batter_id(수치형, cat_features 미포함)만 추가해서 BSS를 직접 비교.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

TARGET_COL = "control_success"
ID_COL = "row_id"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]  # pitcher_id/batter_id는 수치형(label-encoded)으로만 추가, cat_features 아님

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

BASELINE_BSS_RAW = 800.75         # train_catboost.py 2024 홀드아웃, raw id 제외, 보정 전
BASELINE_BSS_CALIB = 825.57       # 〃, 보정 후 (리더보드 889->959로 실측 확인됨)


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


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in ["pitcher_id", "batter_id"]:
        uniq = sorted(X[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        X[c] = X[c].astype(str).map(mapping).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def main():
    print("Load data (pitcher_id/batter_id 수치형 포함, jh_ws 방식)...")
    X_tr, y_tr, X_va, y_va = load_data()
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    print(f" train={X_tr.shape} valid={X_va.shape}")

    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC",
        random_seed=42, cat_features=cat_idx,
        early_stopping_rounds=100, verbose=200,
        **BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    raw_pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, raw_pred)
    bss_raw = bss_score(raw_pred, y_va)
    best_iter = model.get_best_iteration()

    ab = fit_platt(raw_pred, y_va)
    calib_pred = apply_platt(raw_pred, ab)
    bss_calib = bss_score(calib_pred, y_va)

    print(f"\nauc(raw id 포함, 보정 전)  = {auc:.5f}")
    print(f"bss(raw id 포함, 보정 전)  = {bss_raw:.2f}  (baseline raw id 제외 = {BASELINE_BSS_RAW:.2f}, delta={bss_raw - BASELINE_BSS_RAW:+.2f})")
    print(f"bss(raw id 포함, 보정 후)  = {bss_calib:.2f}  (baseline raw id 제외+보정 = {BASELINE_BSS_CALIB:.2f}, delta={bss_calib - BASELINE_BSS_CALIB:+.2f})")
    print(f"best_iteration = {best_iter}")

    imp = model.get_feature_importance(prettified=True)
    print("\n=== Top 15 feature importance ===")
    print(imp.head(15).to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_rawid_numeric_calibrated_result.json"), "w") as f:
        json.dump({
            "auc_with_raw_id": auc,
            "bss_raw_id_uncalibrated": bss_raw,
            "bss_raw_id_calibrated": bss_calib,
            "baseline_bss_no_raw_id_uncalibrated": BASELINE_BSS_RAW,
            "baseline_bss_no_raw_id_calibrated": BASELINE_BSS_CALIB,
            "delta_bss_uncalibrated": bss_raw - BASELINE_BSS_RAW,
            "delta_bss_calibrated": bss_calib - BASELINE_BSS_CALIB,
            "platt_a": ab[0], "platt_b": ab[1],
            "best_iteration": int(best_iter),
        }, f, indent=2)


if __name__ == "__main__":
    main()
