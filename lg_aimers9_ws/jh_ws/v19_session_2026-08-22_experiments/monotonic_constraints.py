"""Monotonic constraints 검증 -- 974 레시피 그대로, 방향성이 명확한
asof_* 피처 7개에 도메인 지식 기반 단조 제약을 건다. 정규화 강화(기각)/
시드 배깅(+4대, 판단보류)에 이은 이번 세션 세 번째 새 축.

사전 확인(전체 train.csv 상관계수, 이 스크립트 실행 전 별도로 확인함):
  양(+1): asof_pitcher_success_rate(0.084), asof_pitcher_prev1_game_success_rate(0.062),
          asof_pitcher_prev3_game_success_rate(0.078), asof_pitcher_prev5_game_success_rate(0.082),
          asof_batter_success_rate(0.059)
  음(-1): asof_pitcher_reverse_rate(-0.080), asof_pitcher_middle_rate(-0.036),
          asof_batter_middle_rate(-0.034)
  (참고: balls_before/strikes_before/outs_before/li는 상관계수가 -0.01~0.004로
  거의 0이라 방향이 불분명 -- 제약 대상에서 제외. asof_pitcher_ball_rate/
  strike_rate도 상관 거의 0이라 제외.)

근거: 위 7개는 전부 "투수/타자의 과거 제구 관련 이력"이라 방향이 도메인
상식과 상관계수 둘 다에서 명확함(과거 성공률이 높을수록 현재도 성공할
가능성이 높다, reverse/middle rate가 높을수록 낮다). 이건 새 정보를
추가하는 피처 엔지니어링이 아니라 **이미 쓰고 있는 기존 피처에 방향성
제약만 추가**하는 것이라, 여러 번 실패했던 "새 situational 피처 추가"류와
실패 메커니즘이 다름 -- 오히려 트리가 노이즈로 잘못된 방향의 split을
학습하는 것을 막는 정규화에 가까움.

채택 기준: fold0/fold2 두 축 모두 baseline보다 개선되어야 함(GroupKFold는
사용자 요청으로 이번 세션 다른 실험들과 동일하게 스킵, 시간 관계상).
정직성 원칙(calibration은 TRAIN 파티션 내부 carve-out에만 fit) 동일 준수.
iterations=1000, 974 레시피(CAT_COLS 네이티브/RAW_ID_COLS/trackman context) 유지.
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
RESULT_PATH = os.path.join(OUT_DIR, "monotonic_constraints_results.json")

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

MONOTONE_DIRS = {
    "asof_pitcher_success_rate": 1,
    "asof_pitcher_prev1_game_success_rate": 1,
    "asof_pitcher_prev3_game_success_rate": 1,
    "asof_pitcher_prev5_game_success_rate": 1,
    "asof_batter_success_rate": 1,
    "asof_pitcher_reverse_rate": -1,
    "asof_pitcher_middle_rate": -1,
    "asof_batter_middle_rate": -1,
}


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


def run_context(tag, train_df, eval_df, use_monotone):
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

    extra = {}
    if use_monotone:
        mono = {col: MONOTONE_DIRS.get(col, 0) for col in X_train_sub.columns}
        extra["monotone_constraints"] = mono

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **BEST_PARAMS, **extra,
    )
    model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = model.predict_proba(X_eval)[:, 1]
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag, "n_train": len(train_df), "n_eval": len(eval_df),
        "best_iteration": model.get_best_iteration(),
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    print(
        f"  [{tag}] auc={result['auc']:.4f} bss_raw={result['bss_raw']:.2f} "
        f"bss_calib={result['bss_calibrated']:.2f} ({elapsed:.1f}s)",
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
        baseline_r = run_context(f"{fold_name}/baseline", train_df, eval_df, use_monotone=False)
        mono_r = run_context(f"{fold_name}/monotonic", train_df, eval_df, use_monotone=True)
        all_results[fold_name] = {
            "baseline": baseline_r,
            "monotonic": mono_r,
            "delta_calibrated": mono_r["bss_calibrated"] - baseline_r["bss_calibrated"],
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    all_positive = all(v["delta_calibrated"] > 0 for v in all_results.values())
    summary = {k: v["delta_calibrated"] for k, v in all_results.items()}
    summary["all_axes_positive"] = all_positive
    all_results["summary"] = summary

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (baseline -> monotonic, calibrated BSS delta) ===", flush=True)
    for k, v in all_results.items():
        if k == "summary":
            continue
        print(f"  {k}: {v['baseline']['bss_calibrated']:.2f} -> {v['monotonic']['bss_calibrated']:.2f}  (delta {v['delta_calibrated']:+.2f})", flush=True)
    print(f"\n  ALL AXES POSITIVE: {all_positive}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
