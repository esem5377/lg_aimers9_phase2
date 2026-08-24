"""타겟 디트렌딩 + 잔차 부스팅. EDA에서 확인된 사실(control_success 평균이
2019->2024 6년 연속 단조 하락, 약 -7.9%p)에 대해, 8/23 `season_trend_prior`
(트렌드를 "피처"로 추가, 실패 -146.8)와 반대 방향으로 접근:

  raw_score(row) = trend_logit(season) + residual_score(row)

trend_logit(season)은 season 단일 변수로 학습한 로지스틱회귀의 decision
function(=a*season+b, 선형이라 2025 같은 미지의 season에도 자연스럽게
외삽됨). CatBoost의 네이티브 `baseline` 파라미터(부스팅이 이 값 위에 잔차를
더하는 offset 메커니즘, GAM의 offset과 동일 원리)로 residual_score만 트리가
학습하게 만듦 -- 트리는 "패턴이 있는 잔차"만 담당하고, 외삽이 필요한 연속적
레벨 이동은 로지스틱회귀 하나가 전담.

987 레시피 그대로(season 피처도 그대로 유지, baseline은 추가 메커니즘),
CatBoost 분류기 캐시(재학습 없이 비교 기준)와 fold0/fold2로 solo 성능 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
CACHE_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\24e954c1-d480-4a70-9d75-dbfa46ca88d3\scratchpad"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "trend_baseline_screening_results.json")

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


def run_fold(fold_name, train_df, eval_df, cache_path):
    train_df = add_risk_score(train_df).drop(columns=INGREDIENT_COLS)
    eval_df = add_risk_score(eval_df).drop(columns=INGREDIENT_COLS)

    y_train_full = train_df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    print(f"  분류기 캐시 로드(비교 기준): {cache_path}", flush=True)
    cls_meta, cls_eval_raw, cls_calib_raw = joblib.load(cache_path)
    print(f"  [classifier cached] auc={cls_meta['auc']:.4f} bss_calib={cls_meta['bss_calibrated']:.2f}", flush=True)

    y_train_sub = train_sub_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]

    print("  season 단독 트렌드(로지스틱회귀, train_sub만으로 fit)...", flush=True)
    SEASON_ORIGIN = 2019  # season 값을 원본(2019~2024) 그대로 쓰면 lbfgs가
    # C=1e10과 결합해 수렴 실패(계수가 0에 수렴하는 버그 발견/수정) -- 중앙화 필요
    trend_lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
    season_train_c = (train_sub_df[["season"]].values - SEASON_ORIGIN).astype(float)
    trend_lr.fit(season_train_c, y_train_sub)
    a_trend = float(trend_lr.coef_[0][0])
    b_trend = float(trend_lr.intercept_[0])
    print(f"  trend_logit(season) = {a_trend:.5f}*(season-{SEASON_ORIGIN}) + {b_trend:.5f}", flush=True)

    baseline_train = trend_lr.decision_function((train_sub_df[["season"]].values - SEASON_ORIGIN).astype(float))
    baseline_calib = trend_lr.decision_function((calib_df[["season"]].values - SEASON_ORIGIN).astype(float))
    baseline_eval = trend_lr.decision_function((eval_df[["season"]].values - SEASON_ORIGIN).astype(float))
    print(f"  season 범위: train={sorted(train_sub_df['season'].unique())}  eval={sorted(eval_df['season'].unique())}", flush=True)
    print(f"  trend_logit 예시: eval season -> baseline 평균={baseline_eval.mean():.4f} (train 평균={baseline_train.mean():.4f})", flush=True)

    id_mappings = build_id_mappings(train_sub_df)
    X_train = build_features(train_sub_df, id_mappings)
    X_calib = build_features(calib_df, id_mappings)
    X_eval = build_features(eval_df, id_mappings)
    cat_idx = [X_train.columns.get_loc(c) for c in CAT_COLS]

    train_pool = Pool(X_train, y_train_sub, cat_features=cat_idx, baseline=baseline_train)
    calib_pool = Pool(X_calib, y_calib, cat_features=cat_idx, baseline=baseline_calib)
    eval_pool_noY = Pool(X_eval, cat_features=cat_idx, baseline=baseline_eval)
    calib_pool_noY = Pool(X_calib, cat_features=cat_idx, baseline=baseline_calib)

    print("  CatBoost(잔차, baseline=trend) 학습...", flush=True)
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **BEST_PARAMS,
    )
    model.fit(train_pool, eval_set=calib_pool)
    elapsed = time.time() - t0
    print(f"  best_iteration={model.get_best_iteration()} ({elapsed:.1f}s)", flush=True)

    calib_raw = model.predict_proba(calib_pool_noY)[:, 1]
    eval_raw = model.predict_proba(eval_pool_noY)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "classifier_baseline_bss": cls_meta["bss_calibrated"],
        "trend_a": a_trend, "trend_b": b_trend,
        "trend_residual_auc": roc_auc_score(y_eval, eval_raw),
        "trend_residual_bss_raw": bss_score(eval_raw, y_eval),
        "trend_residual_bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    result["delta_vs_classifier"] = result["trend_residual_bss_calibrated"] - result["classifier_baseline_bss"]
    print(
        f"  [trend+residual] auc={result['trend_residual_auc']:.4f} "
        f"bss_calib={result['trend_residual_bss_calibrated']:.2f} "
        f"(delta {result['delta_vs_classifier']:+.2f})",
        flush=True,
    )
    return result


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

    print("\n=== SUMMARY (classifier baseline -> trend+residual, calibrated BSS) ===", flush=True)
    for k, v in all_results.items():
        print(
            f"  {k}: {v['classifier_baseline_bss']:.2f} -> {v['trend_residual_bss_calibrated']:.2f} "
            f"(delta {v['delta_vs_classifier']:+.2f})",
            flush=True,
        )
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
