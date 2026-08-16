"""LightGBM 하이퍼파라미터 랜덤서치.

train.py와 동일한 피처(trackman context 포함, raw pitcher_id/batter_id 제외,
team history 제외)와 동일한 시간 기반 검증(2019~2023 학습 / 2024 검증)을
그대로 써서 하이퍼파라미터만 바꿔가며 AUC를 비교한다. 데이터/피처는 한 번만
만들어두고 재사용해 트라이얼마다 로드 오버헤드가 없게 한다.
"""
import json
import os
import random
import time

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

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
DROP_COLS = ["pitcher_id", "batter_id"]

SEARCH_SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [50, 100, 200, 500, 1000],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.5, 1.0],
}

BASELINE = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
)

N_TRIALS = 25
SEED = 42


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def sample_params(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def fit_eval(params, X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=2000, random_state=SEED, n_jobs=-1,
        **params,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, pred), model.best_iteration_


def main():
    rng = random.Random(SEED)
    print("Load data...")
    X_tr, y_tr, X_va, y_va = load_data()
    print(f" train={X_tr.shape} valid={X_va.shape}")

    results = []

    print("\n[baseline]")
    t0 = time.time()
    auc, best_iter = fit_eval(BASELINE, X_tr, y_tr, X_va, y_va)
    print(f" auc={auc:.5f} best_iter={best_iter} ({time.time()-t0:.1f}s) params={BASELINE}")
    results.append({"trial": "baseline", "auc": auc, "best_iteration": best_iter, **BASELINE})

    tried = {tuple(sorted(BASELINE.items()))}
    for i in range(N_TRIALS):
        params = sample_params(rng)
        key = tuple(sorted(params.items()))
        if key in tried:
            continue
        tried.add(key)

        t0 = time.time()
        try:
            auc, best_iter = fit_eval(params, X_tr, y_tr, X_va, y_va)
        except Exception as e:
            print(f"[trial {i}] FAILED: {e}")
            continue
        dt = time.time() - t0
        print(f"[trial {i}] auc={auc:.5f} best_iter={best_iter} ({dt:.1f}s) params={params}")
        results.append({"trial": i, "auc": auc, "best_iteration": best_iter, **params})

    res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(OUT_DIR, "tune_results.csv"), index=False)

    print("\n=== Top 10 ===")
    print(res_df.head(10).to_string(index=False))

    best = res_df.iloc[0].to_dict()
    print("\nBest params:", json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
