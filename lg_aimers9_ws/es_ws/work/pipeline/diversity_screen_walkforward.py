"""8/25 세션 -- 두 가지를 시간 기준 walk-forward(train<=2023 -> eval=2024)로 재검증.

1. "소폭 delta 그룹" 중 checkpoint_avg.py(median 블렌드 / 체크포인트 스냅샷 평균)만
   재검증. isna_flags.py/carveout_ratio.py는 원래부터 fold0/fold2 season split으로
   평가했어서 재검증 불필요(FULL_PROJECT_LOG 8/25 확인). checkpoint_avg.py는 랜덤
   5% carve-out을 평가셋으로 재사용(calibration fit과 동일 셋)한 오염된 방법이라
   원래 결과(-1.56~-9.80)를 신뢰할 수 없음 -- v15가 겪은 것과 같은 함정.

2. "CatBoost/retrieval과 다른 귀납편향" 새 축 후보로 CatBoost 네이티브 rsm(random
   subspace method, 트리마다 컬럼을 무작위 부분집합만 사용)을 시드 배깅과 별도
   축으로 추가했을 때 디코릴레이션 효과가 있는지 확인. FM/로지스틱/ExtraTrees처럼
   "완전히 다른 모델 계열"은 전부 fold2(2024)에서 붕괴(bss=0)했던 반복 패턴이 있어,
   이번엔 이미 검증된 CatBoost 골격 안에서 분산원만 하나 더 추가하는 보수적인 접근.

베이스 레시피는 v13과 동일(987, control_risk_score 원재료 drop 유지). CB6(rsm 없음)은
v13_walkforward_check.py와 동일 설정으로 새로 학습(모델을 저장하지 않는 스크립트라 재사용 불가).
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

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/diversity_screen_walkforward.json"

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

CB_SEEDS = [42, 7, 123, 1, 99, 777]
CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
CAT_ITERATIONS = 2000
CAT_TASK_TYPE = "GPU"
CHECKPOINTS = [1800, 1850, 1900, 1950, 2000]
RSM = 0.8


def log(msg):
    print(msg, flush=True)


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


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


def build_id_mappings(df):
    return {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in RAW_ID_COLS}


def build_catboost_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def train_one(seed, X_train, y_train, cat_idx, extra_params=None):
    params = dict(CAT_BEST_PARAMS)
    if extra_params:
        params.update(extra_params)
    t0 = time.time()
    m = CatBoostClassifier(
        iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=seed,
        cat_features=cat_idx, verbose=False, task_type=CAT_TASK_TYPE, devices="0",
        **params,
    )
    m.fit(X_train, y_train)
    log(f"    seed={seed} done ({time.time()-t0:.0f}s)")
    return m


def calib_and_score(raw_calib, y_calib, raw_eval, y_eval):
    a, b = fit_platt(raw_calib, y_calib)
    eval_pred = apply_platt(raw_eval, a, b)
    return bss_score(eval_pred, y_eval), roc_auc_score(y_eval, raw_eval)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df = add_risk_score_drop_ingredients(df)
    log(f"shape={df.shape}")

    train_df = df[df["season"] <= 2023]
    eval_df = df[df["season"] == 2024]
    log(f"train={len(train_df)} eval(2024)={len(eval_df)}")

    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)
    train_sub = train_sub.reset_index(drop=True)
    calib = calib.reset_index(drop=True)
    y_calib = calib[TARGET_COL].values
    y_eval = eval_df[TARGET_COL].values

    id_mappings = build_id_mappings(train_sub)
    X_train_cb = build_catboost_features(train_sub, id_mappings)
    X_calib_cb = build_catboost_features(calib, id_mappings)
    X_eval_cb = build_catboost_features(eval_df, id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]
    y_train_cb = train_sub[TARGET_COL]

    result = {}

    # === Task A: CB6(no rsm) -- baseline / median / checkpoint-avg, on eval=2024 ===
    log("\n=== Task A: CatBoost 6seed baseline (no rsm), checkpoints saved ===")
    calib_ckpt = {c: [] for c in CHECKPOINTS}
    eval_ckpt = {c: [] for c in CHECKPOINTS}
    for seed in CB_SEEDS:
        m = train_one(seed, X_train_cb, y_train_cb, cat_idx)
        for c in CHECKPOINTS:
            calib_ckpt[c].append(m.predict_proba(X_calib_cb, ntree_end=c)[:, 1])
            eval_ckpt[c].append(m.predict_proba(X_eval_cb, ntree_end=c)[:, 1])

    final_calib = np.array(calib_ckpt[2000])
    final_eval = np.array(eval_ckpt[2000])

    baseline_mean_calib = final_calib.mean(axis=0)
    baseline_mean_eval = final_eval.mean(axis=0)
    bss_baseline, auc_baseline = calib_and_score(baseline_mean_calib, y_calib, baseline_mean_eval, y_eval)
    log(f"  baseline(6seed final-only, mean): auc={auc_baseline:.4f} bss_calib={bss_baseline:.2f}")

    median_calib = np.median(final_calib, axis=0)
    median_eval = np.median(final_eval, axis=0)
    bss_median, auc_median = calib_and_score(median_calib, y_calib, median_eval, y_eval)
    log(f"  median(6seed final-only): auc={auc_median:.4f} bss_calib={bss_median:.2f}  delta={bss_median-bss_baseline:+.2f}")

    all_ckpt_calib = np.concatenate([np.array(calib_ckpt[c]) for c in CHECKPOINTS], axis=0)
    all_ckpt_eval = np.concatenate([np.array(eval_ckpt[c]) for c in CHECKPOINTS], axis=0)
    ckpt_mean_calib = all_ckpt_calib.mean(axis=0)
    ckpt_mean_eval = all_ckpt_eval.mean(axis=0)
    bss_ckptavg, auc_ckptavg = calib_and_score(ckpt_mean_calib, y_calib, ckpt_mean_eval, y_eval)
    log(f"  checkpoint_avg(6seed x {len(CHECKPOINTS)}ckpt): auc={auc_ckptavg:.4f} bss_calib={bss_ckptavg:.2f}  delta={bss_ckptavg-bss_baseline:+.2f}")

    result["task_a_checkpoint_median"] = {
        "baseline_6seed_final_mean": {"auc": auc_baseline, "bss_calib": bss_baseline},
        "median_6seed_final": {"auc": auc_median, "bss_calib": bss_median, "delta_vs_baseline": bss_median - bss_baseline},
        "checkpoint_avg_6seed_x5ckpt": {"auc": auc_ckptavg, "bss_calib": bss_ckptavg, "delta_vs_baseline": bss_ckptavg - bss_baseline},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # === Task B: CB6 with rsm=0.8 (column subsampling) -- new diversity axis ===
    log(f"\n=== Task B: CatBoost 6seed with rsm={RSM} (new diversity axis) ===")
    rsm_calib_preds, rsm_eval_preds = [], []
    for seed in CB_SEEDS:
        m = train_one(seed, X_train_cb, y_train_cb, cat_idx, extra_params={"rsm": RSM})
        rsm_calib_preds.append(m.predict_proba(X_calib_cb)[:, 1])
        rsm_eval_preds.append(m.predict_proba(X_eval_cb)[:, 1])

    rsm_calib_mean = np.mean(rsm_calib_preds, axis=0)
    rsm_eval_mean = np.mean(rsm_eval_preds, axis=0)
    bss_rsm, auc_rsm = calib_and_score(rsm_calib_mean, y_calib, rsm_eval_mean, y_eval)
    log(f"  rsm-only(6seed, rsm={RSM}): auc={auc_rsm:.4f} bss_calib={bss_rsm:.2f}  (vs no-rsm baseline {bss_baseline:.2f})")

    # correlation between no-rsm and rsm ensembles' raw eval predictions (diversity check)
    corr = float(np.corrcoef(baseline_mean_eval, rsm_eval_mean)[0, 1])
    log(f"  corr(no-rsm mean, rsm mean) on eval raw preds = {corr:.4f}")

    combined_calib = np.concatenate([final_calib, np.array(rsm_calib_preds)], axis=0).mean(axis=0)
    combined_eval = np.concatenate([final_eval, np.array(rsm_eval_preds)], axis=0).mean(axis=0)
    bss_combined, auc_combined = calib_and_score(combined_calib, y_calib, combined_eval, y_eval)
    log(f"  combined(6 no-rsm + 6 rsm, 12-way mean): auc={auc_combined:.4f} bss_calib={bss_combined:.2f}  delta_vs_baseline={bss_combined-bss_baseline:+.2f}")

    result["task_b_rsm_diversity"] = {
        "rsm_value": RSM,
        "rsm_only_6seed": {"auc": auc_rsm, "bss_calib": bss_rsm},
        "corr_norsm_vs_rsm_eval_preds": corr,
        "combined_12way": {"auc": auc_combined, "bss_calib": bss_combined, "delta_vs_baseline": bss_combined - bss_baseline},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
