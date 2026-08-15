"""실험: CatBoost에 raw pitcher_id/batter_id를 범주형으로 다시 포함시켜 봄.

LightGBM에서는 이 두 컬럼이 과적합을 일으켜(work worklog #3) 제거했지만,
CatBoost는 ordered target statistics 방식으로 고카디널리티 범주형을
다르게 다루므로 과적합 없이 신호를 살릴 수 있는지 확인한다.
train_catboost.py와 동일 피처(trackman context 포함)·동일 시간 분할·
동일 하이퍼파라미터(BEST_PARAMS)를 쓰고, cat_features에
pitcher_id/batter_id만 추가한 버전으로 검증 AUC를 비교한다.
"""
import json
import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/output"

TARGET_COL = "control_success"
ID_COL = "row_id"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "pitcher_id", "batter_id",  # raw id 재도입 (기존엔 DROP_COLS였음)
]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

BASELINE_AUC = 0.55056  # train_catboost.py 현재 채택 버전 (raw id 제외)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def main():
    print("Load data (raw pitcher_id/batter_id 포함)...")
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

    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    best_iter = model.get_best_iteration()

    print(f"\nauc(raw id 포함) = {auc:.5f}  best_iteration={best_iter}")
    print(f"auc(raw id 제외, 현재 채택 버전) = {BASELINE_AUC:.5f}")
    print(f"delta = {auc - BASELINE_AUC:+.5f}")

    imp = model.get_feature_importance(prettified=True)
    print("\n=== Top 15 feature importance ===")
    print(imp.head(15).to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_catboost_rawid_result.json"), "w") as f:
        json.dump({
            "auc_with_raw_id": auc,
            "auc_baseline_no_raw_id": BASELINE_AUC,
            "delta": auc - BASELINE_AUC,
            "best_iteration": int(best_iter),
        }, f, indent=2)


if __name__ == "__main__":
    main()
