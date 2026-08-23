"""987점(v22 drop_ingredients) 레시피에서 "약하면서 불안정한" asof_* 컬럼
5개를 추가로 제거한 버전, 단일 시드(42)로 빠르게 실제 제출 검증.

배경(2026-08-23, 사용자 제안): asof_* 19개 컬럼 전체의 연도별(2019~2024)
상관계수 range(max-min)를 스캔한 결과, 전체상관이 매우 약하면서(|corr|<0.02)
연도 간 부호까지 뒤집히는 컬럼 5개를 확인:
  asof_pitcher_n(표본수, overall_corr=-0.012, range=0.048, sign_flip)
  asof_batter_n(표본수, overall_corr=-0.036, range=0.061, sign_flip)
  asof_pitcher_pitchmix_n(표본수, overall_corr=-0.012, range=0.048, sign_flip)
  asof_pitcher_strike_rate(overall_corr=0.004, range=0.059, sign_flip --
    8/23 control_quality_score 실패의 원인이었던 그 컬럼)
  asof_pitcher_fastball_rate(overall_corr=-0.0002, range=0.034, sign_flip)
강하지만 불안정한 컬럼(asof_batter_success_rate range=0.089,
asof_pitcher_reverse_rate)은 8/23 quality_score 실패 전례(강한 신호 제거 시
-28)로 건드리지 않음 -- "약한 신호만 골라 제거"가 이번 실험의 핵심 가설.

로컬 fold0/fold2 스크리닝(drop_unstable_weak.py) 결과 fold0 -11.98
(2368.23->2356.25)로 소폭 음수 -- 이 프로젝트에서 소폭 delta는 로컬만으로
기각하지 않고 실제 제출로 확인하는 관례(8/22 control_risk_score가 로컬
fold0 -3.74/fold2 +7.04였는데 실제로는 +10이었던 전례)에 따라 fold2 결과
대기 없이 바로 실제 제출 진행(사용자 명시적 요청).

CAT_COLS(네이티브)/RAW_ID_COLS(label-encoded)/trackman context/BEST_PARAMS/
calibration carve-out(5%, seed=42) 방식은 v18/v20/v22와 동일.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt_scaling(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt_scaling(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v24_drop_weak_unstable_1seed\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v24_drop_weak_unstable_1seed\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]
WEAK_UNSTABLE_COLS = [
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
    "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
ITERATIONS = 2000
SEED = 42
DATA_SPLIT_SEED = 42


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score_drop_cols(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    df = df.drop(columns=INGREDIENT_COLS + WEAK_UNSTABLE_COLS)
    return df


def build_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data (전체) + control_risk_score(*2) 추가, 원재료3+약한불안정5 = 8개 제거...", flush=True)
    df = load_data()
    df = add_risk_score_drop_cols(df)
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]
    print(f" n_features={X_all.shape[1]} (v22의 70 - 약한불안정5 = 65)", flush=True)

    print("\nCalibration carve-out(5%) 분리...", flush=True)
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}", flush=True)

    print(f"\n=== seed={SEED} 학습 (iterations={ITERATIONS}) ===", flush=True)
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED,
        cat_features=cat_idx, verbose=200,
        **BEST_PARAMS,
    )
    model.fit(X_train_final, y_train_final)
    elapsed = time.time() - t0
    print(f" 학습 완료 ({elapsed:.1f}s)", flush=True)

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a_final, b_final = fit_platt_scaling(calib_raw, y_calib)
    calib_pred = apply_platt_scaling(calib_raw, a_final, b_final)

    metrics = {
        "seed": SEED,
        "elapsed_sec": elapsed,
        "carveout_bss_raw": bss_score(calib_raw, y_calib),
        "carveout_bss_calibrated": bss_score(calib_pred, y_calib),
    }
    print(f" carve-out BSS: raw={metrics['carveout_bss_raw']:.2f}  "
          f"calibrated={metrics['carveout_bss_calibrated']:.2f}", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v24_1seed.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model_path = os.path.join(MODEL_DIR, f"catboost_seed{SEED}.cbm")
    model.save_model(model_path)
    print(f" saved: {model_path}", flush=True)

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "seeds": [SEED],
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved 1 model + feature_meta.json to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
