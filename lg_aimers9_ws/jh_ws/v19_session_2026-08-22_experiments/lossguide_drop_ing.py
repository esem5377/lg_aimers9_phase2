"""control_risk_score 추가 + 원재료(reverse/middle/ball_rate) 제거 레시피
(70피처, risk_score_drop_ingredients.py의 drop_ingredients와 동일 피처셋,
실제 제출로도 987점 확인됨/v22 1seed)를 base로 두고, CatBoost의 트리 성장
방식만 대칭(Symmetric, 기존)에서 비대칭(Lossguide)으로 교체 검증.

이 프로젝트에서 지금까지 한 번도 안 써본 구조적 레버 -- feature 추가/제거가
아니라 트리 성장 알고리즘 자체를 바꾸는 시도. Lossguide는 LightGBM과 비슷한
leaf-wise 성장 방식(가장 손실을 많이 줄이는 리프를 우선 분할)이라, 기존
depth 기반 대칭 트리와 다른 종류의 표현력을 가짐.

Symmetric baseline(risk_score_drop_ingredients.py의 drop_ingredients)은
이미 계산됐으므로 재사용, Lossguide만 신규 학습.
  drop_ingredients(Symmetric, 기존): fold0=2368.23, fold2=848.32

Lossguide 파라미터: 기존 BEST_PARAMS(learning_rate/l2_leaf_reg/bagging_temperature/
random_strength/border_count)는 그대로 유지하고 grow_policy만 교체 +
max_leaves=64(대칭 depth=8의 최대 256리프보다 작게 잡아 과적합 방지,
Lossguide 기본 관례인 leaf 수 직접 제어 방식을 따름), depth 파라미터는
Lossguide에서 사용 안 하므로 제거.
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
RESULT_PATH = os.path.join(OUT_DIR, "lossguide_drop_ing_results.json")
PRIOR_RESULT_PATH = os.path.join(OUT_DIR, "risk_score_drop_ingredients_results.json")

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

LOSSGUIDE_PARAMS = dict(
    learning_rate=0.01, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
    grow_policy="Lossguide", max_leaves=64,
)


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


def run_lossguide(tag, train_df, eval_df):
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
    y_calib = calib_df[TARGET_COL]
    X_eval = build_features(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL]

    cat_idx = [X_train_sub.columns.get_loc(c) for c in CAT_COLS]

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **LOSSGUIDE_PARAMS,
    )
    model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = model.predict_proba(X_eval)[:, 1]
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag, "variant": "lossguide", "n_features": X_train_sub.shape[1],
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
        r = run_lossguide(f"{fold_name}/lossguide", train_df, eval_df)
        symmetric_bss = prior[fold_name]["drop_ingredients"]["bss_calibrated"]
        all_results[fold_name] = {
            "symmetric_reused": prior[fold_name]["drop_ingredients"],
            "lossguide": r,
            "delta_vs_symmetric": r["bss_calibrated"] - symmetric_bss,
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = {fn: all_results[fn]["delta_vs_symmetric"] for fn in fold_specs}
    summary["all_axes_positive"] = all(v > 0 for v in summary.values())
    all_results["summary"] = summary
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (Lossguide vs Symmetric, drop_ingredients base) ===", flush=True)
    print(f"  {summary}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
