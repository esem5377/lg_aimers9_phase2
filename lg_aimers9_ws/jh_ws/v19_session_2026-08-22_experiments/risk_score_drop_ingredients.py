"""control_risk_score의 원재료(asof_pitcher_reverse_rate/middle_rate/ball_rate)를
남겨둔 채 파생 컬럼만 추가한 게 지금까지의 방식(71->73피처, 992점 기여 확인됨).
"원재료를 빼고 파생만 남기면 어떨까"만 확인 -- baseline/with_ingredients는
control_risk_score.py에서 동일 시드/동일 fold로 이미 계산됐으므로 재학습하지
않고 control_risk_score_results.json에서 재사용, drop_ingredients만 신규 학습.

  A) baseline: 원재료 있음, risk_score 없음 (71피처) -- 재사용
  B) with_ingredients(992 방식): 원재료 있음 + risk_score 2개 (73피처) -- 재사용
  C) drop_ingredients: 원재료 3개 제거 + risk_score 2개만 (70피처) -- 신규 학습

982 레시피(CAT_COLS 네이티브/RAW_ID_COLS/trackman context/BEST_PARAMS) 그대로,
fold0(->2022)/fold2(->2024), 동일 SEED=42, 동일 calibration 방식.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "risk_score_drop_ingredients_results.json")
PRIOR_RESULT_PATH = os.path.join(OUT_DIR, "control_risk_score_results.json")

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

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
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


def run_drop_ingredients(tag, train_df, eval_df):
    train_df = add_risk_score(train_df).drop(columns=INGREDIENT_COLS)
    eval_df = add_risk_score(eval_df).drop(columns=INGREDIENT_COLS)

    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL]

    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    X_train_sub = build_features(train_sub_df, id_mappings)
    y_train_sub = train_sub_df[TARGET_COL]
    X_calib = build_features(calib_df, id_mappings)
    y_calib = calib_df[TARGET_COL]
    X_eval = build_features(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL]

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
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = model.predict_proba(X_eval)[:, 1]
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag, "variant": "drop_ingredients", "n_features": X_train_sub.shape[1],
        "best_iteration": model.get_best_iteration(),
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    print(
        f"  [{tag}] n_features={result['n_features']} auc={result['auc']:.4f} "
        f"bss_calib={result['bss_calibrated']:.2f} ({elapsed:.1f}s)",
        flush=True,
    )
    return result


def main():
    with open(PRIOR_RESULT_PATH, encoding="utf-8") as f:
        prior = json.load(f)

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
        drop_r = run_drop_ingredients(f"{fold_name}/drop_ingredients", train_df, eval_df)
        baseline_bss = prior[fold_name]["baseline"]["bss_calibrated"]
        with_ing_bss = prior[fold_name]["with_risk_score"]["bss_calibrated"]
        all_results[fold_name] = {
            "baseline_reused": prior[fold_name]["baseline"],
            "with_ingredients_reused": prior[fold_name]["with_risk_score"],
            "drop_ingredients": drop_r,
            "delta_drop_vs_baseline": drop_r["bss_calibrated"] - baseline_bss,
            "delta_drop_vs_with_ingredients": drop_r["bss_calibrated"] - with_ing_bss,
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = {
        fn: {
            "delta_drop_vs_baseline": all_results[fn]["delta_drop_vs_baseline"],
            "delta_drop_vs_with_ingredients": all_results[fn]["delta_drop_vs_with_ingredients"],
        }
        for fn in fold_specs
    }
    all_results["summary"] = summary
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (drop_ingredients vs baseline / vs with_ingredients) ===", flush=True)
    for fn, s in summary.items():
        print(f"  {fn}: {s}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
