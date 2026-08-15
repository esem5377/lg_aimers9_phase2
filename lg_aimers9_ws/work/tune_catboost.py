"""CatBoost 하이퍼파라미터 랜덤서치.

train_catboost.py와 동일 피처(trackman context 3종 포함, raw id 제외)와
동일한 시간 기반 검증(2019~2023 학습 / 2024 검증)으로 데이터를 한 번만
로드해두고 파라미터만 바꿔가며 비교한다.
"""
import json
import os
import random
import time

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/output"

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
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 10, 20],
    "bagging_temperature": [0, 0.5, 1, 2],
    "random_strength": [0.5, 1, 2, 5],
    "border_count": [32, 64, 128, 254],
}

BASELINE = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    bagging_temperature=1, random_strength=1, border_count=254,
)

N_TRIALS = 15
SEED = 42

# 2026-08-15: 세션 끊김으로 중단된 실행에서 이미 얻은 결과 (로그에서 복원).
# random.Random(SEED)는 재실행 시 동일 순서로 샘플링하므로, rng를 이만큼
# 그대로 소진시키고 결과는 재사용해 trial 0~6을 다시 학습하지 않는다.
COMPLETED_TRIALS = 7
COMPLETED_RESULTS = [
    {"trial": "baseline", "auc": 0.55005, "best_iteration": 348, **BASELINE},
    {"trial": 0, "auc": 0.54935, "best_iteration": 1792, "learning_rate": 0.01, "depth": 4, "l2_leaf_reg": 5, "bagging_temperature": 0.5, "random_strength": 1, "border_count": 64},
    {"trial": 1, "auc": 0.54947, "best_iteration": 1987, "learning_rate": 0.01, "depth": 4, "l2_leaf_reg": 20, "bagging_temperature": 2, "random_strength": 0.5, "border_count": 32},
    {"trial": 2, "auc": 0.55013, "best_iteration": 1240, "learning_rate": 0.01, "depth": 6, "l2_leaf_reg": 3, "bagging_temperature": 0, "random_strength": 1, "border_count": 254},
    {"trial": 3, "auc": 0.54974, "best_iteration": 402, "learning_rate": 0.02, "depth": 10, "l2_leaf_reg": 20, "bagging_temperature": 1, "random_strength": 0.5, "border_count": 64},
    {"trial": 4, "auc": 0.54983, "best_iteration": 158, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5, "bagging_temperature": 0.5, "random_strength": 1, "border_count": 128},
    {"trial": 5, "auc": 0.54028, "best_iteration": 58, "learning_rate": 0.01, "depth": 4, "l2_leaf_reg": 10, "bagging_temperature": 0, "random_strength": 2, "border_count": 128},
    {"trial": 6, "auc": 0.55031, "best_iteration": 109, "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 1, "bagging_temperature": 2, "random_strength": 0.5, "border_count": 254},
]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def sample_params(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def fit_eval(params, X_tr, y_tr, X_va, y_va, cat_idx):
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx,
        early_stopping_rounds=100, verbose=False,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, pred), model.get_best_iteration()


def main():
    rng = random.Random(SEED)
    print("Load data...")
    X_tr, y_tr, X_va, y_va = load_data()
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    print(f" train={X_tr.shape} valid={X_va.shape}")

    results = list(COMPLETED_RESULTS)
    print(f"\n[resume] reusing {len(COMPLETED_RESULTS)} cached results (baseline + trial 0~{COMPLETED_TRIALS - 1})")
    for r in COMPLETED_RESULTS:
        print(f" trial={r['trial']} auc={r['auc']:.5f} best_iter={r['best_iteration']} (cached)")

    tried = {tuple(sorted(BASELINE.items()))}
    for i in range(COMPLETED_TRIALS):
        params = sample_params(rng)
        tried.add(tuple(sorted(params.items())))

    for i in range(COMPLETED_TRIALS, N_TRIALS):
        params = sample_params(rng)
        key = tuple(sorted(params.items()))
        if key in tried:
            continue
        tried.add(key)

        t0 = time.time()
        try:
            auc, best_iter = fit_eval(params, X_tr, y_tr, X_va, y_va, cat_idx)
        except Exception as e:
            print(f"[trial {i}] FAILED: {e}")
            continue
        dt = time.time() - t0
        print(f"[trial {i}] auc={auc:.5f} best_iter={best_iter} ({dt:.1f}s) params={params}")
        results.append({"trial": i, "auc": auc, "best_iteration": best_iter, **params})

    res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(OUT_DIR, "tune_catboost_results.csv"), index=False)

    print("\n=== Top 10 ===")
    print(res_df.head(10).to_string(index=False))

    best = res_df.iloc[0].to_dict()
    print("\nBest params:", json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
