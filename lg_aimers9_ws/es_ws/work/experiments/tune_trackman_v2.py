"""trackman_context_v2(확장판: extension/rel_height/rel_side/zone_speed + std)이
기존 trackman_context(mean 4종만)보다 나은지 3-fold walk-forward CV로 검증.

섹션 2(work worklog)의 트랙맨 교차 피처 실패(2024 단일 홀드아웃에서만 좋고
리더보드는 하락)를 반복하지 않기 위해, 처음부터 tune_cv_robust.py와 동일한
3-fold walk-forward(2022/2023/2024 각각 valid)로 검증하고 폴드 전체에서
일관되게 이겨야만 "진짜 신호"로 인정한다. CatBoost 파라미터는 v4 채택값
(tuned_params) 고정, 트랙맨 컨텍스트 버전만 바꿔서 비교.
"""
import json
import os
import time

import joblib
import pandas as pd
from catboost import CatBoostClassifier
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

TUNED_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold1(train<=2022,valid2023)", "train_seasons": [2019, 2020, 2021, 2022], "valid_season": 2023},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]


def load_base_df():
    return pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")


def merge_context(df, context_file):
    context = joblib.load(os.path.join(MODEL_DIR, context_file))
    out = df.copy()
    for spec in context.values():
        out = out.merge(spec["table"], on=spec["keys"], how="left")
    return out


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def fit_eval_fold(X_tr, y_tr, X_va, y_va, params):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, pred), model.get_best_iteration()


def run_config(name, df, params):
    print(f"\n{'=' * 60}\n[CONFIG] {name}\n{'=' * 60}", flush=True)
    y_all = df[TARGET_COL]
    X_all = build_features(df)
    print(f"  n_features={X_all.shape[1]}", flush=True)

    fold_aucs = []
    for fold in FOLDS:
        tr_mask = df["season"].isin(fold["train_seasons"])
        va_mask = df["season"] == fold["valid_season"]
        t0 = time.time()
        auc, best_iter = fit_eval_fold(
            X_all[tr_mask], y_all[tr_mask], X_all[va_mask], y_all[va_mask], params)
        dt = time.time() - t0
        print(f"  [{fold['name']}] auc={auc:.5f} best_iter={best_iter} ({dt:.1f}s) "
              f"n_train={tr_mask.sum()} n_valid={va_mask.sum()}", flush=True)
        fold_aucs.append(auc)

    mean_auc = sum(fold_aucs) / len(fold_aucs)
    print(f"  -> mean_auc={mean_auc:.5f}  per-fold={['%.5f' % a for a in fold_aucs]}", flush=True)
    return {"name": name, "fold_aucs": fold_aucs, "mean_auc": mean_auc}


def main():
    t_start = time.time()
    print("Load base train.csv...", flush=True)
    df_base = load_base_df()
    print(f" shape={df_base.shape}", flush=True)

    print("Merge trackman_context (baseline, mean 4종)...", flush=True)
    df_v1 = merge_context(df_base, "trackman_context.pkl")
    print("Merge trackman_context_v2 (확장: +extension/rel_height/rel_side/zone_speed, +std)...", flush=True)
    df_v2 = merge_context(df_base, "trackman_context_v2.pkl")

    results = []
    results.append(run_config("v1_baseline (기존 trackman context, mean 4종)", df_v1, TUNED_PARAMS))
    results.append(run_config("v2_extended (확장 트랙맨 물리량 + std)", df_v2, TUNED_PARAMS))

    print(f"\n{'=' * 60}\n=== FINAL SUMMARY (elapsed {time.time() - t_start:.0f}s) ===\n{'=' * 60}", flush=True)
    base = results[0]
    for r in results:
        delta = r["mean_auc"] - base["mean_auc"]
        fold_deltas = [a - b for a, b in zip(r["fold_aucs"], base["fold_aucs"])]
        consistent = all(d > 0 for d in fold_deltas) if r is not base else True
        print(f"{r['name']:55s} mean_auc={r['mean_auc']:.5f}  delta_vs_v1={delta:+.5f}  "
              f"fold_deltas={['%+.5f' % d for d in fold_deltas]}  모든폴드에서_v1보다_좋음={consistent}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_trackman_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_trackman_v2_results.json", flush=True)


if __name__ == "__main__":
    main()
