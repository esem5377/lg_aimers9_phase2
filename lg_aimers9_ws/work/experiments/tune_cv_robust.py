"""자동 탐색 세션 (2026-08-16, ~1시간 예산): 단일 홀드아웃 선택 편향 문제를
해결하기 위해 3-fold walk-forward CV로 재검증하면서 새 후보 몇 개를 테스트.

work worklog #12(교차 피처 v5)에서 2024 단일 홀드아웃 기준으로 좋아 보였던
피처가 리더보드에서는 오히려 하락(889->880)한 사건이 있었다. 원인으로
"2024 검증셋에서만 우연히 좋았던 선택 편향"을 지목했는데, 이번엔 그
가설 자체와 새 후보들을 여러 시즌에 걸친 walk-forward CV로 한 번에
검증한다: 폴드 전부에서 방향이 일관돼야만 "진짜 신호"로 취급.

FOLDS:
  fold0: train 2019~2021 -> valid 2022
  fold1: train 2019~2022 -> valid 2023
  fold2: train 2019~2023 -> valid 2024 (기존 단일 홀드아웃과 동일)

CONFIGS:
  baseline_params : CatBoost 기본값 계열 파라미터(최초 채택, lr=0.03/depth=8/l2=3)
  tuned_params     : work/tune_catboost.py에서 찾은 현재 채택 파라미터(v4, trial 7)
                     -> baseline 대비 폴드 전체에서 정말 이기는지 재검증
  tuned+ctr_deep   : tuned_params에 max_ctr_complexity=6 추가
                     (CatBoost의 leak-safe한 ordered CTR 자동 조합을 더 깊게 허용
                     -> 예전에 실패한 수동 trackman 교차피처와 달리 in-fold
                     ordered target statistics라 선택 편향 위험이 훨씬 적음)
  tuned+interact   : tuned_params + train.csv 컬럼만으로 만든 안전한 파생 피처 3종
                     (skill_diff, count_pressure, risp) 추가
                     -> 전부 해당 행 자신의 입력 변수만 쓰는 단순 연산이라
                     대회 규칙(행 독립 예측) 위반 없음.
"""
import json
import os
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

BASELINE_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    bagging_temperature=1, random_strength=1, border_count=254,
)
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
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_safe_interactions(df):
    df = df.copy()
    df["ix_skill_diff"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    df["ix_count_pressure"] = df["strikes_before"] - df["balls_before"]
    df["ix_risp"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def fit_eval_fold(X_tr, y_tr, X_va, y_va, params, extra_cb_kwargs=None):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    kwargs = dict(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **params,
    )
    if extra_cb_kwargs:
        kwargs.update(extra_cb_kwargs)
    model = CatBoostClassifier(**kwargs)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, pred), model.get_best_iteration()


def run_config(name, df_base, y_all, params, extra_cb_kwargs=None, use_interactions=False):
    print(f"\n{'=' * 60}\n[CONFIG] {name}\n{'=' * 60}")
    df = add_safe_interactions(df_base) if use_interactions else df_base
    X_all = build_features(df)

    fold_aucs = []
    for fold in FOLDS:
        tr_mask = df["season"].isin(fold["train_seasons"])
        va_mask = df["season"] == fold["valid_season"]
        t0 = time.time()
        auc, best_iter = fit_eval_fold(
            X_all[tr_mask], y_all[tr_mask], X_all[va_mask], y_all[va_mask],
            params, extra_cb_kwargs)
        dt = time.time() - t0
        print(f"  [{fold['name']}] auc={auc:.5f} best_iter={best_iter} ({dt:.1f}s) "
              f"n_train={tr_mask.sum()} n_valid={va_mask.sum()}")
        fold_aucs.append(auc)

    mean_auc = sum(fold_aucs) / len(fold_aucs)
    print(f"  -> mean_auc={mean_auc:.5f}  per-fold={['%.5f' % a for a in fold_aucs]}")
    return {"name": name, "fold_aucs": fold_aucs, "mean_auc": mean_auc}


def main():
    t_start = time.time()
    print("Load base train data (trackman context v4, no rejected cross feats)...")
    df_base = load_base_df()
    y_all = df_base[TARGET_COL]
    print(f" shape={df_base.shape}")

    results = []
    results.append(run_config("baseline_params (원조 CatBoost 기본값)", df_base, y_all, BASELINE_PARAMS))
    results.append(run_config("tuned_params (v4 채택, trial 7)", df_base, y_all, TUNED_PARAMS))
    results.append(run_config(
        "tuned+ctr_deep (max_ctr_complexity=6)", df_base, y_all, TUNED_PARAMS,
        extra_cb_kwargs={"max_ctr_complexity": 6}))
    results.append(run_config(
        "tuned+interact (skill_diff/count_pressure/risp)", df_base, y_all, TUNED_PARAMS,
        use_interactions=True))

    print(f"\n{'=' * 60}\n=== FINAL SUMMARY (elapsed {time.time() - t_start:.0f}s) ===\n{'=' * 60}")
    base_mean = results[0]["mean_auc"]
    for r in sorted(results, key=lambda x: -x["mean_auc"]):
        delta = r["mean_auc"] - base_mean
        consistent = all((a - b) > 0 for a, b in zip(r["fold_aucs"], results[0]["fold_aucs"])) if r["name"] != results[0]["name"] else True
        print(f"{r['name']:50s} mean_auc={r['mean_auc']:.5f}  delta_vs_baseline_params={delta:+.5f}  "
              f"모든 폴드에서 baseline_params보다 좋음={consistent}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_cv_robust_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_cv_robust_results.json")


if __name__ == "__main__":
    main()
