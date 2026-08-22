"""982점(v18 6시드 배깅) 레시피 + control_risk_score 피처 추가.
CAT_COLS(네이티브)/RAW_ID_COLS(label-encoded)/trackman context/BEST_PARAMS/
SEEDS(42,7,123,1,99,777)/ITERATIONS=2000/calibration carve-out(5%, seed=42)
전부 v18과 동일. 유일한 변경은 기존 asof_pitcher_reverse_rate/middle_rate/
ball_rate를 재조합한 2개 컬럼을 피처로 추가하는 것:
  control_risk_score = reverse_rate + middle_rate + ball_rate
  control_risk_score_weighted = 0.4*reverse + 0.3*middle + 0.3*ball

2026-08-22 세션 로컬 검증(v19_session.../control_risk_score.py, iterations=1000,
season walk-forward): fold0(->2022) delta -3.74, fold2(->2024) delta +7.04 --
방향이 엇갈리는 소폭 신호라 로컬 판정만으론 확정 불가, 실제 제출로 검증.

새 피처가 들어가 기존 6개 모델을 재사용할 수 없음(피처셋이 다름) --
"기존 재사용+신규만 학습" 방식이었던 train_add_seeds.py와 달리 이번엔
6시드 전부 처음부터 학습한다(시드당 약 30분, 총 약 3시간).
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
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v20_control_risk_score\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v20_control_risk_score\output"

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
SEEDS = [42, 7, 123, 1, 99, 777]
DATA_SPLIT_SEED = 42  # v18과 동일 -- 다만 컬럼 구성이 다르므로 carve-out 인덱스 자체는 달라짐


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
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

    print("Load train data (전체, season 제한 없음) + control_risk_score 피처 추가...", flush=True)
    df = load_data()
    df = add_risk_score(df)
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]
    print(f" n_features={X_all.shape[1]} (v18 대비 +2: control_risk_score, control_risk_score_weighted)", flush=True)

    print("\nCalibration carve-out(5%) 분리 (모든 시드가 동일 split 공유)...", flush=True)
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}", flush=True)

    calib_raw_preds = []
    seed_metrics = []
    for seed in SEEDS:
        print(f"\n=== seed={seed} 학습 (iterations={ITERATIONS}) ===", flush=True)
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=ITERATIONS, loss_function="Logloss", random_seed=seed,
            cat_features=cat_idx, verbose=200,
            **BEST_PARAMS,
        )
        model.fit(X_train_final, y_train_final)
        elapsed = time.time() - t0
        print(f" seed={seed} 학습 완료 ({elapsed:.1f}s)", flush=True)

        calib_raw = model.predict_proba(X_calib)[:, 1]
        calib_raw_preds.append(calib_raw)
        seed_metrics.append({
            "seed": seed,
            "carveout_bss_raw": bss_score(calib_raw, y_calib),
            "elapsed_sec": elapsed,
        })

        model_path = os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm")
        model.save_model(model_path)
        print(f" saved: {model_path}", flush=True)

        # 중간에 죽어도 재개 가능하도록 매 시드마다 meta 스냅샷 저장
        with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "columns": list(X_all.columns),
                "cat_cols": CAT_COLS,
                "raw_id_cols": RAW_ID_COLS,
                "id_mappings": id_mappings,
                "seeds": [m["seed"] for m in seed_metrics],
                "calibration": None,
            }, f, indent=2, ensure_ascii=False)

    print("\n=== 배깅: 6시드 raw 확률 평균 -> Platt 1회 fit ===", flush=True)
    calib_raw_bagged = np.mean(calib_raw_preds, axis=0)
    a_final, b_final = fit_platt_scaling(calib_raw_bagged, y_calib)
    calib_pred_bagged = apply_platt_scaling(calib_raw_bagged, a_final, b_final)

    metrics = {
        "seeds": seed_metrics,
        "carveout_bss_raw_bagged": bss_score(calib_raw_bagged, y_calib),
        "carveout_bss_calibrated_bagged": bss_score(calib_pred_bagged, y_calib),
    }
    print(f" carve-out BSS(배깅): raw={metrics['carveout_bss_raw_bagged']:.2f}  "
          f"calibrated={metrics['carveout_bss_calibrated_bagged']:.2f}", flush=True)
    print(f" (참고: v18 6시드 carve-out calibrated=2069.21)", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v20_control_risk_score.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "seeds": SEEDS,
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(SEEDS)} models + feature_meta.json to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
