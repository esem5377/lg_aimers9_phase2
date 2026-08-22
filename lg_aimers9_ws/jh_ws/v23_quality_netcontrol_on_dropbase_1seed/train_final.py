"""987점 베이스(v22 drop_ingredients: control_risk_score 추가 + 원재료
reverse/middle/ball_rate 제거, 70피처) 위에 control_quality_score(=
success_rate+strike_rate)와 pitcher_net_control(=success_rate-
control_risk_score)을 "both"로 추가. 이번엔 quality/net_control의 원재료
(success_rate/strike_rate)는 제거하지 않고 그대로 유지 -- 8/23 로컬
검증에서 원재료 제거 조합은 두 폴드 다 확실히 나빴던 반면(quality_on_
drop_ing_base.py, fold0 -7.94/fold2 -13.50), 원재료 유지 조합은 아직
987점 베이스 위에서 테스트한 적이 없어 실제 제출로 확인.

참고로 992 베이스(risk_score 원재료 유지) 위에서의 "both, 원재료 유지"
로컬 결과는 fold0 -8.72/fold2 +6.00으로 애매했음(quality_net_control.py).
단일 시드(42)로 빠르게 실제 제출 검증, 로컬 fold0/fold2는 생략.

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
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v23_quality_netcontrol_on_dropbase_1seed\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v23_quality_netcontrol_on_dropbase_1seed\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
RISK_INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

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


def add_features(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    df["control_quality_score"] = df["asof_pitcher_success_rate"] + df["asof_pitcher_strike_rate"]
    df["pitcher_net_control"] = df["asof_pitcher_success_rate"] - df["control_risk_score"]
    # risk_score 원재료만 제거(987점 베이스), quality/net_control 원재료(success_rate/strike_rate)는 유지
    df = df.drop(columns=RISK_INGREDIENT_COLS)
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

    print("Load train data (전체) + control_risk_score(원재료 제거) + quality/net_control(원재료 유지) 추가...", flush=True)
    df = load_data()
    df = add_features(df)
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]
    print(f" n_features={X_all.shape[1]} (987베이스 70 + quality/net_control 2 = 72)", flush=True)

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
    print(f" (참고: v22 987점 1시드 베이스 carve-out calibrated는 별도 기록 없음, v20 6시드 2068.46/v21 1시드 참고용)", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v23_1seed.json"), "w", encoding="utf-8") as f:
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
