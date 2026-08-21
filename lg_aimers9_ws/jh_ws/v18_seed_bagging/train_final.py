"""974점(es_ws v7) 레시피 그대로 + 시드 배깅(seed-averaging bagging) 추가.
CAT_COLS(네이티브)/RAW_ID_COLS(label-encoded)/trackman context/BEST_PARAMS는
전부 동일. 유일한 변경은 완전히 동일한 레시피를 랜덤시드 3개(42/7/123)로
각각 학습해 raw 확률을 평균낸 뒤, 그 평균 위에 Platt calibration을 1회
fit하는 것 -- 새 정보/피처를 전혀 추가하지 않고 순수하게 분산만 줄이는
가장 보수적인 앙상블.

2026-08-21 세션 로컬 검증(scratchpad seed_bagging.py, iterations=1000,
season walk-forward): fold0(->2022) 단일시드 평균 2386.50(std=1.85) ->
배깅 2390.84(+4.34), fold2(->2024) 832.63(std=2.28) -> 836.60(+3.98).
두 축 모두 일관되게 양수. GroupKFold(미본 투수) 축은 사용자 요청으로
이번엔 확인하지 않고(시간 관계상 1시간40분 소요 예상) 바로 프로덕션
빌드로 진행함 -- 검증 커버리지가 fold0/fold2 두 축뿐이라는 걸 감안할 것.

시간 절약을 위해 이번엔 기존 프로덕션들의 "season<=2023 학습/2024 검증"
진단 단계를 생략하고(이미 위 실험으로 충분히 확인됨), 바로 전체 데이터
refit(ITERATIONS=2000 고정, best_iteration 진단 없이 프로덕션 스케일
그대로) + calibration carve-out 분리로 넘어간다.
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
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v18_seed_bagging\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v18_seed_bagging\output"

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
SEEDS = [42, 7, 123]
DATA_SPLIT_SEED = 42  # calibration carve-out 분리는 고정, 모델 시드만 변수


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
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

    print("Load train data (전체, season 제한 없음 -- 974 프로덕션과 동일)...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

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

    print("\n=== 배깅: 3시드 raw 확률 평균 -> Platt 1회 fit ===", flush=True)
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
    with open(os.path.join(OUT_DIR, "metrics_v18_seed_bagging.json"), "w", encoding="utf-8") as f:
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
