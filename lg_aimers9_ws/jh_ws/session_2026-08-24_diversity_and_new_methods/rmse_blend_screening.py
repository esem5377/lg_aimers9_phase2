"""987 레시피 CatBoost 분류기(sigmoid calib) + CatBoostRegressor(RMSE 직접회귀)
블렌드 스크리닝. hackathon_fresh v3(8/18, 별도 레시피) "D조합"의 재현이지만
이번엔 현재 production 레시피(987, control_risk_score 포함)로 fold0/fold2에서
검증.

분류기는 재학습 없이 이전 세션 캐시(catboost_cache_fold0_2022.pkl /
catboost_cache_fold2_2024.pkl, bss_calibrated 2368.23/848.32로 987 baseline과
정확히 일치 확인됨)를 재사용. RMSE 회귀 모델만 두 폴드 새로 학습.

블렌드 가중치(alpha, classifier 비중)는 calib_df(각 fold의 학습 파티션 내부
5% carve-out) 자체로 그리드서치해서 선택(v26 blend_weight_search.py와 동일
원칙), eval_df(계절 홀드아웃)는 최종 보고에만 사용 -- eval 라벨은 alpha
선택에 전혀 관여 안 함.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
CACHE_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\24e954c1-d480-4a70-9d75-dbfa46ca88d3\scratchpad"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "rmse_blend_screening_results.json")

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


def train_regressor(X_train, y_train, X_calib, y_calib):
    cat_idx = [X_train.columns.get_loc(c) for c in CAT_COLS]
    t0 = time.time()
    model = CatBoostRegressor(
        iterations=ITERATIONS, loss_function="RMSE", eval_metric="RMSE",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **BEST_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0
    print(f"  regressor best_iteration={model.get_best_iteration()} ({elapsed:.1f}s)", flush=True)
    return model, elapsed


def run_fold(fold_name, train_df, eval_df, cache_path):
    train_df = add_risk_score(train_df).drop(columns=INGREDIENT_COLS)
    eval_df = add_risk_score(eval_df).drop(columns=INGREDIENT_COLS)

    y_train_full = train_df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    id_mappings = build_id_mappings(train_sub_df)

    print(f"  분류기 캐시 로드: {cache_path}", flush=True)
    cls_meta, cls_eval_raw, cls_calib_raw = joblib.load(cache_path)
    print(f"  [classifier cached] auc={cls_meta['auc']:.4f} bss_calib={cls_meta['bss_calibrated']:.2f}", flush=True)

    X_train_r = build_features(train_sub_df, id_mappings)
    y_train_r = train_sub_df[TARGET_COL]
    X_calib_r = build_features(calib_df, id_mappings)
    y_calib_r = calib_df[TARGET_COL]
    X_eval_r = build_features(eval_df, id_mappings)
    y_eval_r = eval_df[TARGET_COL]

    print("  RMSE 회귀 학습...", flush=True)
    reg_model, reg_elapsed = train_regressor(X_train_r, y_train_r, X_calib_r, y_calib_r)
    reg_calib_raw = reg_model.predict(X_calib_r)
    reg_eval_raw = reg_model.predict(X_eval_r)

    reg_auc = roc_auc_score(y_eval_r, reg_eval_raw)
    a_r, b_r = fit_platt(reg_calib_raw, y_calib_r)
    reg_eval_calib = apply_platt(reg_eval_raw, a_r, b_r)
    reg_bss = bss_score(reg_eval_calib, y_eval_r)
    print(f"  [regressor solo] auc={reg_auc:.4f} bss_calib={reg_bss:.2f}", flush=True)

    print("  alpha 그리드서치(calib_df 자체로 선택, eval 라벨 미사용)...", flush=True)
    alphas = [round(x * 0.1, 1) for x in range(0, 11)]
    calib_scores = []
    for alpha in alphas:
        blend_calib_raw = alpha * cls_calib_raw + (1 - alpha) * reg_calib_raw
        a_b, b_b = fit_platt(blend_calib_raw, y_calib_r)
        blend_calib_pred = apply_platt(blend_calib_raw, a_b, b_b)
        score = bss_score(blend_calib_pred, y_calib_r)
        calib_scores.append((alpha, score, a_b, b_b))
    best_alpha, best_calib_score, best_a, best_b = max(calib_scores, key=lambda t: t[1])
    print(f"  best_alpha(calib 기준)={best_alpha} (calib_bss={best_calib_score:.2f})", flush=True)

    blend_eval_raw = best_alpha * cls_eval_raw + (1 - best_alpha) * reg_eval_raw
    blend_eval_pred = apply_platt(blend_eval_raw, best_a, best_b)
    blend_eval_bss = bss_score(blend_eval_pred, y_eval_r)

    return {
        "classifier_baseline_bss": cls_meta["bss_calibrated"],
        "regressor_solo_bss": reg_bss,
        "regressor_solo_auc": reg_auc,
        "regressor_elapsed_sec": reg_elapsed,
        "best_alpha": best_alpha,
        "blend_eval_bss": blend_eval_bss,
        "delta_vs_classifier": blend_eval_bss - cls_meta["bss_calibrated"],
        "alpha_grid_on_calib": [(a, s) for a, s, _, _ in calib_scores],
    }


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022], os.path.join(CACHE_DIR, "catboost_cache_fold0_2022.pkl")),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024], os.path.join(CACHE_DIR, "catboost_cache_fold2_2024.pkl")),
    }

    all_results = {}
    for fold_name, (train_df, eval_df, cache_path) in fold_specs.items():
        print(f"\n=== {fold_name} ===", flush=True)
        r = run_fold(fold_name, train_df, eval_df, cache_path)
        all_results[fold_name] = r
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (classifier baseline -> RMSE blend, calibrated BSS) ===", flush=True)
    for k, v in all_results.items():
        print(
            f"  {k}: {v['classifier_baseline_bss']:.2f} -> {v['blend_eval_bss']:.2f} "
            f"(alpha={v['best_alpha']}, delta {v['delta_vs_classifier']:+.2f})",
            flush=True,
        )
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
