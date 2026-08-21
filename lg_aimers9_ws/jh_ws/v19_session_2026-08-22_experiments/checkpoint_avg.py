"""체크포인트(스냅샷) 평균 -- 재학습 없이, 이미 저장된 982점 프로덕션의
6개 모델(model/catboost_seed{42,7,123,1,99,777}.cbm, 전부 iterations=2000
학습 완료)을 그대로 로드해서, 마지막 트리(2000번째)만 쓰는 대신 뒤쪽
체크포인트 여러 개(1800/1850/1900/1950/2000번째 트리)의 예측까지 같이
평균낸다. 시드 배깅과 동일한 원리(새 정보 없이 순수 노이즈만 줄임)를
"트리 개수" 축으로 확장한 것 -- 딥러닝의 SWA(가중치 평균)와 유사.

평가는 프로덕션과 동일한 calibration carve-out(X_calib/y_calib, 전체
데이터의 5%, random_state=42로 결정론적 재현)에서 진행. 이 carve-out은
Platt 보정을 fit하는 데도 쓰이는 셋이라 "완전히 독립적인 holdout"은
아니지만, 3->6시드 확장 판단(2065.91->2069.21) 때도 동일한 방식으로
비교해 실제 제출로 검증됐던 방법론이라 여기서도 상대비교 용도로 신뢰함.

병행 검증: 6개 모델(마지막 체크포인트만) 단순평균(mean) vs 중앙값(median)도
같이 비교.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v18_seed_bagging\model"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "checkpoint_avg_results.json")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
SEEDS = [42, 7, 123, 1, 99, 777]
DATA_SPLIT_SEED = 42
CHECKPOINTS = [1800, 1850, 1900, 1950, 2000]


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
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def main():
    print("Load feature_meta (id_mappings 재사용, 재계산 안 함)...", flush=True)
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    id_mappings = meta["id_mappings"]

    print("Load train data + trackman context (calib split 재현용)...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    _, X_calib, _, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    X_calib = X_calib[meta["columns"]]
    print(f" calib carve-out={X_calib.shape}", flush=True)

    print("\nLoad 6 saved models (재학습 없음)...", flush=True)
    models = {}
    for seed in SEEDS:
        m = CatBoostClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))
        models[seed] = m
        print(f" loaded seed={seed}", flush=True)

    print("\n=== baseline: 6모델 x 마지막 체크포인트(2000)만, 단순평균 ===", flush=True)
    final_preds = {seed: m.predict_proba(X_calib)[:, 1] for seed, m in models.items()}
    baseline_mean = np.mean(list(final_preds.values()), axis=0)
    a, b = fit_platt(baseline_mean, y_calib)
    baseline_calib = apply_platt(baseline_mean, a, b)
    baseline_bss = bss_score(baseline_calib, y_calib)
    print(f" baseline(6seed, final only) calibrated BSS = {baseline_bss:.2f}"
          f"  (참고: 프로덕션 기록값 2069.21)", flush=True)

    print("\n=== median 블렌드: 6모델 x 마지막 체크포인트만, 중앙값 ===", flush=True)
    baseline_median = np.median(list(final_preds.values()), axis=0)
    a_med, b_med = fit_platt(baseline_median, y_calib)
    median_calib = apply_platt(baseline_median, a_med, b_med)
    median_bss = bss_score(median_calib, y_calib)
    print(f" median(6seed, final only) calibrated BSS = {median_bss:.2f}"
          f"  (delta vs mean {median_bss - baseline_bss:+.2f})", flush=True)

    print(f"\n=== 체크포인트 평균: 6모델 x {CHECKPOINTS} 체크포인트, 전부 평균 ===", flush=True)
    all_ckpt_preds = []
    for seed, m in models.items():
        for ntree in CHECKPOINTS:
            p = m.predict_proba(X_calib, ntree_end=ntree)[:, 1]
            all_ckpt_preds.append(p)
        print(f" seed={seed}: {len(CHECKPOINTS)}개 체크포인트 예측 완료", flush=True)
    ckpt_mean = np.mean(all_ckpt_preds, axis=0)
    a_ck, b_ck = fit_platt(ckpt_mean, y_calib)
    ckpt_calib = apply_platt(ckpt_mean, a_ck, b_ck)
    ckpt_bss = bss_score(ckpt_calib, y_calib)
    print(f" checkpoint_avg(6seed x {len(CHECKPOINTS)}ckpt = {len(all_ckpt_preds)}-way) "
          f"calibrated BSS = {ckpt_bss:.2f}  (delta vs baseline {ckpt_bss - baseline_bss:+.2f})", flush=True)

    result = {
        "baseline_6seed_final_only": baseline_bss,
        "median_6seed_final_only": median_bss,
        "checkpoint_avg_6seed_x_5ckpt": ckpt_bss,
        "delta_median_vs_mean": median_bss - baseline_bss,
        "delta_checkpoint_avg_vs_baseline": ckpt_bss - baseline_bss,
        "checkpoints_used": CHECKPOINTS,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
