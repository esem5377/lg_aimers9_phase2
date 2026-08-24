"""987 레시피 CatBoost 분류기(캐시 재사용) + ExtraTrees(배깅 계열) 블렌드
스크리닝. CatBoost(순차 부스팅, ordered boosting)와 달리 각 트리가 독립적으로
부트스트랩+랜덤 분할점으로 학습되는 배깅 모델이라 오류 생성 메커니즘이
근본적으로 다름. RandomForest보다 ExtraTrees를 택한 이유: 분할점을 완전
랜덤으로 고르는(최적 분할 탐색 안 함) 만큼 학습이 빨라 그리드서치에 유리하고,
개별 트리 간 상관을 더 줄여 CatBoost와 다르게 틀릴 가능성도 이론적으로 더 큼.

sklearn 트리는 CatBoost처럼 categorical/NaN을 네이티브로 못 다루므로:
  - CAT_COLS(저카디널리티 7개): one-hot 인코딩(train_sub 기준 컬럼 고정)
  - RAW_ID_COLS(pitcher_id/batter_id, 고카디널리티): 정수 라벨 인코딩(OOV=-1)
  - 수치형: train_sub 중앙값으로 결측 대체 + _isna 플래그

튜닝: ExtraTreesClassifier(bootstrap=True, oob_score=True)의 OOB 예측 확률
(oob_decision_function_)로 별도 held-out 없이 그리드서치 -- 재학습 없이
학습 데이터 자체에서 unbiased 검증 신호를 얻는 방식.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
CACHE_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\24e954c1-d480-4a70-9d75-dbfa46ca88d3\scratchpad"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "extratrees_blend_screening_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

GRID = [
    {"max_depth": 15, "min_samples_leaf": 50},
    {"max_depth": 15, "min_samples_leaf": 200},
    {"max_depth": None, "min_samples_leaf": 50},
    {"max_depth": None, "min_samples_leaf": 200},
]
N_ESTIMATORS_SEARCH = 200
N_ESTIMATORS_FINAL = 400


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


def build_rf_prep(train_sub_df):
    """train_sub 기준으로 one-hot 컬럼셋 + 수치형 중앙값을 고정."""
    id_mappings = build_id_mappings(train_sub_df)
    dummies = pd.get_dummies(train_sub_df[CAT_COLS].astype(str), columns=CAT_COLS)
    dummy_columns = dummies.columns.tolist()
    numeric_cols = [
        c for c in train_sub_df.columns
        if c not in [ID_COL, TARGET_COL] + CAT_COLS + RAW_ID_COLS
    ]
    medians = train_sub_df[numeric_cols].median()
    return {
        "id_mappings": id_mappings, "dummy_columns": dummy_columns,
        "numeric_cols": numeric_cols, "medians": medians,
    }


def build_rf_features(df, prep):
    dummies = pd.get_dummies(df[CAT_COLS].astype(str), columns=CAT_COLS)
    dummies = dummies.reindex(columns=prep["dummy_columns"], fill_value=0)

    ids = pd.DataFrame(index=df.index)
    for c in RAW_ID_COLS:
        ids[c] = df[c].astype(str).map(prep["id_mappings"][c]).fillna(-1).astype(int)

    X_num = df[prep["numeric_cols"]].copy()
    isna_flags = X_num.isna().astype(np.float32)
    isna_flags.columns = [f"{c}__isna" for c in prep["numeric_cols"]]
    X_num = X_num.fillna(prep["medians"])

    X = pd.concat(
        [dummies.reset_index(drop=True), ids.reset_index(drop=True),
         X_num.reset_index(drop=True), isna_flags.reset_index(drop=True)],
        axis=1,
    )
    return X.values.astype(np.float32)


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

    print(f"  분류기 캐시 로드: {cache_path}", flush=True)
    cls_meta, cls_eval_raw, cls_calib_raw = joblib.load(cache_path)
    print(f"  [classifier cached] auc={cls_meta['auc']:.4f} bss_calib={cls_meta['bss_calibrated']:.2f}", flush=True)

    y_train_sub = train_sub_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]

    print("  피처 전처리(one-hot + id 라벨 + 수치형 중앙값대체)...", flush=True)
    prep = build_rf_prep(train_sub_df)
    X_train = build_rf_features(train_sub_df, prep)
    X_calib = build_rf_features(calib_df, prep)
    X_eval = build_rf_features(eval_df, prep)
    print(f"  n_features={X_train.shape[1]}", flush=True)

    print(f"  ExtraTrees 그리드서치({len(GRID)}개 조합, OOB 기준, n_estimators={N_ESTIMATORS_SEARCH})...", flush=True)
    t0 = time.time()
    best = None
    for params in GRID:
        et = ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS_SEARCH, bootstrap=True, oob_score=True,
            n_jobs=-1, random_state=SEED, **params,
        )
        et.fit(X_train, y_train_sub)
        oob_proba = et.oob_decision_function_[:, 1]
        oob_bss = bss_score(oob_proba, y_train_sub)
        print(f"    {params} -> oob_bss={oob_bss:.2f}", flush=True)
        if best is None or oob_bss > best[0]:
            best = (oob_bss, params)
    best_oob_bss, best_params = best
    print(f"  best_params={best_params} (oob_bss={best_oob_bss:.2f}, {time.time()-t0:.1f}s)", flush=True)

    print(f"  최종 모델 재학습(n_estimators={N_ESTIMATORS_FINAL})...", flush=True)
    t1 = time.time()
    et_final = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS_FINAL, bootstrap=True, oob_score=True,
        n_jobs=-1, random_state=SEED, **best_params,
    )
    et_final.fit(X_train, y_train_sub)
    final_elapsed = time.time() - t1
    print(f"  최종 모델 학습 완료 ({final_elapsed:.1f}s)", flush=True)

    et_calib_raw = et_final.predict_proba(X_calib)[:, 1]
    et_eval_raw = et_final.predict_proba(X_eval)[:, 1]

    et_auc = roc_auc_score(y_eval, et_eval_raw)
    a_e, b_e = fit_platt(et_calib_raw, y_calib)
    et_eval_calib = apply_platt(et_eval_raw, a_e, b_e)
    et_bss = bss_score(et_eval_calib, y_eval)
    print(f"  [extratrees solo] auc={et_auc:.4f} bss_calib={et_bss:.2f}", flush=True)

    print("  alpha 그리드서치(classifier vs extratrees, calib_df 기준)...", flush=True)
    alphas = [round(x * 0.1, 1) for x in range(0, 11)]
    calib_scores = []
    for alpha in alphas:
        blend_calib_raw = alpha * cls_calib_raw + (1 - alpha) * et_calib_raw
        a_b, b_b = fit_platt(blend_calib_raw, y_calib)
        blend_calib_pred = apply_platt(blend_calib_raw, a_b, b_b)
        score = bss_score(blend_calib_pred, y_calib)
        calib_scores.append((alpha, score, a_b, b_b))
    best_alpha, best_calib_score, best_a, best_b = max(calib_scores, key=lambda t: t[1])
    print(f"  best_alpha(calib 기준)={best_alpha} (calib_bss={best_calib_score:.2f})", flush=True)

    blend_eval_raw = best_alpha * cls_eval_raw + (1 - best_alpha) * et_eval_raw
    blend_eval_pred = apply_platt(blend_eval_raw, best_a, best_b)
    blend_eval_bss = bss_score(blend_eval_pred, y_eval)

    return {
        "classifier_baseline_bss": cls_meta["bss_calibrated"],
        "extratrees_solo_bss": et_bss,
        "extratrees_solo_auc": et_auc,
        "best_params": best_params,
        "best_alpha": best_alpha,
        "blend_eval_bss": blend_eval_bss,
        "delta_vs_classifier": blend_eval_bss - cls_meta["bss_calibrated"],
        "search_elapsed_sec": time.time() - t0,
        "final_train_elapsed_sec": final_elapsed,
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

    print("\n=== SUMMARY (classifier baseline -> extratrees blend, calibrated BSS) ===", flush=True)
    for k, v in all_results.items():
        print(
            f"  {k}: {v['classifier_baseline_bss']:.2f} -> {v['blend_eval_bss']:.2f} "
            f"(alpha={v['best_alpha']}, params={v['best_params']}, delta {v['delta_vs_classifier']:+.2f})",
            flush=True,
        )
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
