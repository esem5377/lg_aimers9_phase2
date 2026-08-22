"""992점(v20 control_risk_score) 레시피 + matchup_skill_gap/recent_form_gap
("both" 변형) 추가, 단일 시드(42)로 빠르게 검증용 제출.

배경: 두 피처를 fold0/fold2 walk-forward로 개별 검증했을 때 matchup은 두
폴드 다 음수(-7.23/-0.93), recent_form은 fold2에서 -8.43으로 나빴지만,
둘을 합친 "both"는 fold0 +5.30/fold2 -1.05로 가장 균형 잡힌(그나마 나은)
조합이었음(2026-08-22, matchup_recent_gap.py). 확신 있는 신호는 아니라서
6시드 풀 재학습(3시간) 대신 시드 1개(42)로 빠르게(iterations=2000, 약
30분) 만들어 실제 제출로 검증. 비교 대상은 현재 팀 최고 992점(v20,
6시드+control_risk_score) -- 다만 이 실험은 시드 1개뿐이라 순수하게
"두 피처의 효과"만 격리된 비교는 아니고(시드 배깅 자체의 기여분(982->992
경로에서 6시드가 3시드 대비 +3 정도)도 같이 빠짐), 빠른 방향 확인용임을
감안할 것.

CAT_COLS(네이티브)/RAW_ID_COLS(label-encoded)/trackman context/BEST_PARAMS/
ITERATIONS=2000/calibration carve-out(5%, seed=42)는 v18/v20과 동일.
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
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v21_matchup_recent_gap_1seed\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v21_matchup_recent_gap_1seed\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
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
ITERATIONS = 2000
SEED = 42
DATA_SPLIT_SEED = 42


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_all_features(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    df["matchup_skill_gap"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    df["recent_form_gap"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
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

    print("Load train data (전체) + control_risk_score(*2) + matchup_skill_gap + recent_form_gap 추가...", flush=True)
    df = load_data()
    df = add_all_features(df)
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]
    print(f" n_features={X_all.shape[1]} (v20 대비 +2: matchup_skill_gap, recent_form_gap)", flush=True)

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
    print(f" (참고: v20 6시드 carve-out calibrated=2068.46)", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v21_1seed.json"), "w", encoding="utf-8") as f:
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
