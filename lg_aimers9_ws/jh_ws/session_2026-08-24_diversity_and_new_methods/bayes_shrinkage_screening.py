"""987 레시피 CatBoost 분류기(캐시 재사용) + 베이지안 shrinkage(Beta-Binomial
부분 풀링) 통계 모델 블렌드 스크리닝.

트리도 kernel regression도 아닌 세 번째 축: 공식 제공 asof_pitcher_n/
asof_pitcher_success_rate, asof_batter_n/asof_batter_success_rate(이미
point-in-time으로 안전하게 사전계산된 컬럼)에 순수 통계 공식만 적용.

shrunk_rate = (n*rate + kappa*global_mean) / (n + kappa)
  -> 표본(n)이 적을수록 global_mean 쪽으로 강하게 당겨짐(regression to the
     mean), 표본이 많을수록 실측 rate를 그대로 신뢰. 트리 분기도 커널 유사도
     평균도 아닌 순수 폐쇄형 공식이라 학습이랄 게 없음(kappa만 그리드서치).

투수/타자 shrunk rate 두 개를 작은 로지스틱회귀(피처 2개)로 결합해 최종
확률 생성. CatBoost 분류기와 fold0/fold2로 블렌드해 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
CACHE_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\24e954c1-d480-4a70-9d75-dbfa46ca88d3\scratchpad"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "bayes_shrinkage_screening_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

KAPPA_GRID = [5, 20, 50, 100, 300, 1000]


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


def shrink_rate(n, rate, global_mean, kappa):
    n = n.fillna(0.0)
    rate = rate.fillna(0.0)
    return (n * rate + kappa * global_mean) / (n + kappa)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_shrunk_features(df, global_mean, kappa_p, kappa_b):
    sp = shrink_rate(df["asof_pitcher_n"], df["asof_pitcher_success_rate"], global_mean, kappa_p)
    sb = shrink_rate(df["asof_batter_n"], df["asof_batter_success_rate"], global_mean, kappa_b)
    return np.column_stack([logit(sp), logit(sb)])


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
    global_mean = float(y_train_sub.mean())

    print(f"  kappa 그리드서치({len(KAPPA_GRID)}x{len(KAPPA_GRID)}, calib_df 기준)...", flush=True)
    t0 = time.time()
    best = None
    for kp in KAPPA_GRID:
        for kb in KAPPA_GRID:
            X_train_s = build_shrunk_features(train_sub_df, global_mean, kp, kb)
            lr = LogisticRegression(C=1e10, solver="lbfgs")
            lr.fit(X_train_s, y_train_sub)

            X_calib_s = build_shrunk_features(calib_df, global_mean, kp, kb)
            calib_raw = lr.predict_proba(X_calib_s)[:, 1]
            score = bss_score(calib_raw, y_calib)
            if best is None or score > best[0]:
                best = (score, kp, kb, lr)
    best_score, best_kp, best_kb, best_lr = best
    print(f"  best kappa_pitcher={best_kp} kappa_batter={best_kb} (calib_bss={best_score:.2f}, {time.time()-t0:.1f}s)", flush=True)

    X_calib_s = build_shrunk_features(calib_df, global_mean, best_kp, best_kb)
    X_eval_s = build_shrunk_features(eval_df, global_mean, best_kp, best_kb)
    shrink_calib_raw = best_lr.predict_proba(X_calib_s)[:, 1]
    shrink_eval_raw = best_lr.predict_proba(X_eval_s)[:, 1]

    shrink_auc = roc_auc_score(y_eval, shrink_eval_raw)
    a_s, b_s = fit_platt(shrink_calib_raw, y_calib)
    shrink_eval_calib = apply_platt(shrink_eval_raw, a_s, b_s)
    shrink_bss = bss_score(shrink_eval_calib, y_eval)
    print(f"  [shrinkage solo] auc={shrink_auc:.4f} bss_calib={shrink_bss:.2f}", flush=True)

    print("  alpha 그리드서치(classifier vs shrinkage, calib_df 기준)...", flush=True)
    alphas = [round(x * 0.1, 1) for x in range(0, 11)]
    calib_scores = []
    for alpha in alphas:
        blend_calib_raw = alpha * cls_calib_raw + (1 - alpha) * shrink_calib_raw
        a_b, b_b = fit_platt(blend_calib_raw, y_calib)
        blend_calib_pred = apply_platt(blend_calib_raw, a_b, b_b)
        score = bss_score(blend_calib_pred, y_calib)
        calib_scores.append((alpha, score, a_b, b_b))
    best_alpha, best_calib_score, best_a, best_b = max(calib_scores, key=lambda t: t[1])
    print(f"  best_alpha(calib 기준)={best_alpha} (calib_bss={best_calib_score:.2f})", flush=True)

    blend_eval_raw = best_alpha * cls_eval_raw + (1 - best_alpha) * shrink_eval_raw
    blend_eval_pred = apply_platt(blend_eval_raw, best_a, best_b)
    blend_eval_bss = bss_score(blend_eval_pred, y_eval)

    return {
        "classifier_baseline_bss": cls_meta["bss_calibrated"],
        "shrinkage_solo_bss": shrink_bss,
        "shrinkage_solo_auc": shrink_auc,
        "best_kappa_pitcher": best_kp,
        "best_kappa_batter": best_kb,
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

    print("\n=== SUMMARY (classifier baseline -> shrinkage blend, calibrated BSS) ===", flush=True)
    for k, v in all_results.items():
        print(
            f"  {k}: {v['classifier_baseline_bss']:.2f} -> {v['blend_eval_bss']:.2f} "
            f"(alpha={v['best_alpha']}, kappa_p={v['best_kappa_pitcher']}, kappa_b={v['best_kappa_batter']}, "
            f"delta {v['delta_vs_classifier']:+.2f})",
            flush=True,
        )
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
