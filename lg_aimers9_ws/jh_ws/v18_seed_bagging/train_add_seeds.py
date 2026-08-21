"""기존 3시드(42/7/123, 979점 프로덕션)에 새 시드 3개를 추가 학습해
6시드 배깅으로 확장. 기존 모델은 재학습하지 않고 재사용 -- 동일한
데이터 로드/전처리/calibration split(random_state=42 고정)이라 결정론적으로
동일한 X_all/y_all/X_calib/y_calib이 재현됨을 이용해, 기존 3개 모델의
raw 예측만 다시 계산(수 초, 재학습 30분과 비교해 사실상 공짜)하고 새
시드 3개만 실제로 학습(iterations=2000, 전체데이터, 시드당 약 30분)한다.

배경: 3시드 버전(v18)이 로컬 fold0/fold2 예측(+4.34/+3.98)과 거의 정확히
일치하는 실제 리더보드 +4.1(974.9->979)을 만들어냄 -- 이 프로젝트에서
드물게 로컬 신호가 크기까지 맞아떨어진 사례. 시드 배깅은 새 정보를
추가하지 않는 순수 분산감소라 이 프로젝트의 반복 실패 메커니즘(새 정보의
계절별 과적합)이 성립하지 않는다는 해석을 뒷받침 -- 시드를 늘리는 것도
같은 메커니즘이라 상대적으로 안전한 후속 실험으로 판단해 진행.
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
DATA_SPLIT_SEED = 42  # v18 train_final.py와 동일 -- 동일 split 재현용

EXISTING_SEEDS = [42, 7, 123]
NEW_SEEDS = [1, 99, 777]


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
    print("Load train data (v18 train_final.py와 동일 절차 -- split 재현 목적)...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        existing_meta = json.load(f)
    assert id_mappings == existing_meta["id_mappings"], "id_mappings가 기존 저장값과 다름 -- split 재현 실패 위험"
    print(" id_mappings 기존 저장값과 일치 확인 (재현성 OK)", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}", flush=True)

    calib_raw_preds = []

    print("\n=== 기존 3개 모델 재사용: raw 예측만 재계산 ===", flush=True)
    for seed in EXISTING_SEEDS:
        model = CatBoostClassifier()
        model.load_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))
        calib_raw = model.predict_proba(X_calib)[:, 1]
        calib_raw_preds.append(calib_raw)
        print(f" seed={seed}: 로드+예측 완료 (재학습 안 함)", flush=True)

    seed_metrics = []
    for seed in NEW_SEEDS:
        print(f"\n=== 신규 seed={seed} 학습 (iterations={ITERATIONS}) ===", flush=True)
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

    all_seeds = EXISTING_SEEDS + NEW_SEEDS
    print(f"\n=== 배깅: {len(all_seeds)}시드 raw 확률 평균 -> Platt 1회 fit ===", flush=True)
    calib_raw_bagged = np.mean(calib_raw_preds, axis=0)
    a_final, b_final = fit_platt_scaling(calib_raw_bagged, y_calib)
    calib_pred_bagged = apply_platt_scaling(calib_raw_bagged, a_final, b_final)

    metrics = {
        "all_seeds": all_seeds,
        "new_seed_metrics": seed_metrics,
        "carveout_bss_raw_bagged": bss_score(calib_raw_bagged, y_calib),
        "carveout_bss_calibrated_bagged": bss_score(calib_pred_bagged, y_calib),
    }
    print(f" carve-out BSS(6시드 배깅): raw={metrics['carveout_bss_raw_bagged']:.2f}  "
          f"calibrated={metrics['carveout_bss_calibrated_bagged']:.2f}", flush=True)
    print(f" (참고: 3시드 버전 calibrated=2065.91)", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v18_6seed.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "seeds": all_seeds,
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: feature_meta.json 갱신 (seeds={all_seeds}), 6개 모델 파일 model/에 존재", flush=True)


if __name__ == "__main__":
    main()
