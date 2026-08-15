"""CatBoost 버전 최종 모델 학습 — 리더보드 테스트 제출용.

train.py와 동일한 피처(trackman context 3종 포함, raw pitcher_id/batter_id
제외, team history/cross 그룹 제외 — 둘 다 실험에서 효과 없었음)와 동일한
시간 기반 검증(2019~2023 학습 / 2024 검증)을 사용. tune_v2.py 실험에서
CatBoost 단일 모델이 LightGBM보다 AUC 0.0012 높게 나온 걸 실제로 배포
가능한 형태로 만든다.
"""
import json
import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model_catboost"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

# work/tune_catboost.py 랜덤서치 trial 7 결과 (auc=0.55056, baseline 0.55005 대비 +0.0005).
BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(
        "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model", "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data...")
    df = load_data()
    print(f" shape={df.shape}")

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    X_all = build_features(df)
    y_all = df[TARGET_COL]
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[valid_mask], y_all[valid_mask]
    print(f" train={X_tr.shape} valid(2024)={X_va.shape}")

    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

    print("Fit (time-split validation)...")
    model_cv = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=200,
        **BEST_PARAMS,
    )
    model_cv.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    va_pred = model_cv.predict_proba(X_va)[:, 1]
    metrics = {
        "valid_season": 2024,
        "auc": roc_auc_score(y_va, va_pred),
        "logloss": log_loss(y_va, va_pred),
        "accuracy@0.5": accuracy_score(y_va, (va_pred >= 0.5).astype(int)),
        "best_iteration": model_cv.get_best_iteration(),
    }
    print("Validation metrics:", json.dumps(metrics, indent=2))
    with open(os.path.join(OUT_DIR, "metrics_catboost.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nRefit on full data with best_iteration...")
    best_iter = max(model_cv.get_best_iteration(), 1)
    model_final = CatBoostClassifier(
        iterations=best_iter, loss_function="Logloss", random_seed=42,
        cat_features=cat_idx, verbose=False,
        **BEST_PARAMS,
    )
    model_final.fit(X_all, y_all)

    model_final.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w") as f:
        json.dump({"columns": list(X_all.columns), "cat_cols": CAT_COLS}, f, indent=2)
    print(f"Saved model to {MODEL_DIR}/catboost.cbm")


if __name__ == "__main__":
    main()
