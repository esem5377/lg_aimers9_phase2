"""asof_* 비율 피처의 콜드스타트(표본 수 적은 케이스) 베이지안 스무딩 실험.

smoothed = (n * rate + k * prior) / (n + k)
- prior: 학습 데이터 전체에서 n으로 가중평균한 그 컬럼의 평균값 (train만 사용, 리크 없음)
- k: 스무딩 강도(가상 표본 수). n=0이면 완전히 prior로, n이 크면 원래 rate에 수렴.

원본 raw rate 컬럼은 그대로 두고 smoothed 컬럼을 추가로 붙여 LightGBM이
필요한 쪽을 알아서 쓰게 한다. train.py와 동일 피처(trackman context 포함,
raw id 제외)·동일 시간 분할로 k별 AUC를 비교.
"""
import os

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

RATE_N_PAIRS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]


def compute_priors(df):
    priors = {}
    for rate_col, n_col in RATE_N_PAIRS:
        w = df[n_col].fillna(0)
        r = df[rate_col].fillna(0)
        priors[rate_col] = float((w * r).sum() / w.sum())
    return priors


def add_smoothed(df, priors, k):
    for rate_col, n_col in RATE_N_PAIRS:
        n = df[n_col].fillna(0)
        r = df[rate_col].fillna(0)
        prior = priors[rate_col]
        df[f"{rate_col}_smooth_k{k}"] = (n * r + k * prior) / (n + k)
    return df


def load_base():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, extra_cols):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def fit_eval(X_tr, y_tr, X_va, y_va):
    params = dict(
        objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=63,
        max_depth=-1, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, pred), model.best_iteration_


def main():
    print("Load base data (+ trackman context)...")
    df = load_base()
    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024

    priors = compute_priors(df[train_mask])
    print("Priors (train only):", {k: round(v, 4) for k, v in priors.items()})

    results = []

    # baseline: 스무딩 없음
    X_all = build_features(df, [])
    y_all = df[TARGET_COL]
    auc, bi = fit_eval(X_all[train_mask], y_all[train_mask], X_all[valid_mask], y_all[valid_mask])
    print(f"[baseline, no smoothing] auc={auc:.5f} best_iter={bi}")
    results.append(("baseline", auc, bi))

    for k in [5, 10, 20, 50, 100, 300]:
        df_k = add_smoothed(df.copy(), priors, k)
        X_all = build_features(df_k, None)
        auc, bi = fit_eval(X_all[train_mask], y_all[train_mask], X_all[valid_mask], y_all[valid_mask])
        print(f"[k={k}] auc={auc:.5f} best_iter={bi}")
        results.append((f"k={k}", auc, bi))

    print("\n=== summary ===")
    for name, auc, bi in sorted(results, key=lambda x: -x[1]):
        print(f"{name:25s} auc={auc:.5f} best_iter={bi}")


if __name__ == "__main__":
    main()
