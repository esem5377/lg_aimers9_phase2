"""control_risk_score 추가 + 원재료(reverse/middle/ball_rate) 제거 레시피
(70피처, risk_score_drop_ingredients.py의 drop_ingredients와 동일 피처셋,
실제 제출 987점/v22 1seed로 검증됨)에서, calibration 방식만 비교:
  A) 기존 Platt scaling: 2파라미터 sigmoid를 LogisticRegression(MLE, logloss
     최소화)으로 fit -- 지금까지 프로덕션 전체가 쓴 방식.
  E) Brier-최적화 Platt: 똑같이 2파라미터 sigmoid(a,b)이지만 logloss가 아니라
     실제 채점 지표인 Brier Score(=squared error)를 직접 최소화하도록 fit
     (scipy.optimize.minimize, L-BFGS-B).

Isotonic(더 유연한 비모수 보정, 두 번 기각: -66.79/-11~-28)과 기존 Platt(MLE)
사이의 중간 지점 -- 파라미터 수는 Platt과 동일(2개)이라 과적합 위험은
낮지만, 목적함수를 실제 채점 지표에 맞춘 변형. 모델은 fold당 1번만 학습하고
같은 raw 예측 위에서 두 calibration만 비교(효율).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "brier_optimal_platt_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
ITERATIONS = 1000
SEED = 42

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_mle(raw_p, y):
    """기존 방식: logloss(MLE) 최소화."""
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def fit_platt_brier(raw_p, y, init_ab):
    """신규: Brier Score(squared error) 직접 최소화. MLE 결과를 초기값으로 사용."""
    raw_p = np.asarray(raw_p)
    y = np.asarray(y)

    def loss(ab):
        a, b = ab
        p = sigmoid(a * raw_p + b)
        return np.mean((p - y) ** 2)

    res = minimize(loss, x0=np.array(init_ab), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def apply_calib(raw_p, a, b):
    return sigmoid(a * np.asarray(raw_p) + b)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    df = df.drop(columns=INGREDIENT_COLS)
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


def run_fold(tag, train_df, eval_df):
    train_df = add_risk_score_drop_ingredients(train_df)
    eval_df = add_risk_score_drop_ingredients(eval_df)

    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL]

    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    X_train_sub = build_features(train_sub_df, id_mappings)
    y_train_sub = train_sub_df[TARGET_COL]
    X_calib = build_features(calib_df, id_mappings)
    y_calib = calib_df[TARGET_COL].values
    X_eval = build_features(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL].values

    cat_idx = [X_train_sub.columns.get_loc(c) for c in CAT_COLS]

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **BEST_PARAMS,
    )
    model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0

    calib_raw = model.predict_proba(X_calib)[:, 1]
    eval_raw = model.predict_proba(X_eval)[:, 1]

    # A) 기존 Platt(MLE)
    a_mle, b_mle = fit_platt_mle(calib_raw, y_calib)
    eval_platt = apply_calib(eval_raw, a_mle, b_mle)

    # E) Brier-최적화(MLE 결과를 초기값으로 최적화)
    a_brier, b_brier = fit_platt_brier(calib_raw, y_calib, init_ab=(a_mle, b_mle))
    eval_brier = apply_calib(eval_raw, a_brier, b_brier)

    result = {
        "tag": tag, "n_features": X_train_sub.shape[1],
        "best_iteration": model.get_best_iteration(),
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "platt_mle": {"a": a_mle, "b": b_mle, "bss_calibrated": bss_score(eval_platt, y_eval)},
        "platt_brier": {"a": a_brier, "b": b_brier, "bss_calibrated": bss_score(eval_brier, y_eval)},
        "elapsed_sec": elapsed,
    }
    result["delta_brier_vs_mle"] = result["platt_brier"]["bss_calibrated"] - result["platt_mle"]["bss_calibrated"]
    print(
        f"  [{tag}] auc={result['auc']:.4f} bss_raw={result['bss_raw']:.2f}  "
        f"MLE={result['platt_mle']['bss_calibrated']:.2f}  "
        f"Brier={result['platt_brier']['bss_calibrated']:.2f}  "
        f"delta={result['delta_brier_vs_mle']:+.2f} ({elapsed:.1f}s)",
        flush=True,
    )
    return result


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}
    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    for fold_name, (train_df, eval_df) in fold_specs.items():
        print(f"\n=== {fold_name} ===", flush=True)
        r = run_fold(fold_name, train_df, eval_df)
        all_results[fold_name] = r
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = {fn: all_results[fn]["delta_brier_vs_mle"] for fn in fold_specs}
    summary["all_axes_positive"] = all(v > 0 for v in summary.values())
    all_results["summary"] = summary
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (Brier-optimized Platt vs MLE Platt, calibrated BSS delta) ===", flush=True)
    print(f"  {summary}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
