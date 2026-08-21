"""시드 배깅(seed-averaging bagging) 검증 -- 정규화 강화 가설(기각)에 이어
이번 세션 두 번째로 시도하는 새 축. 이 프로젝트의 다른 모든 "앙상블" 시도
(LGBM/XGBoost 이종 모델, D=분류기+회귀기 손실함수 다양성)와 달리, 이번엔
*완전히 동일한* 974 레시피를 랜덤 시드만 바꿔 N번 학습해 확률을 평균낸다
-- bias는 그대로 두고 variance만 줄이는 가장 보수적인 앙상블 형태.

동기: 이 프로젝트에서 반복 관찰된 "로컬 개선폭과 실제 결과가 역상관"하는
패턴은 단일 시드 모델의 학습 노이즈에 로컬 검증이 낚였을 가능성을 시사함.
시드 배깅은 새 정보/피처를 전혀 추가하지 않고 그 노이즈 자체를 줄이는
원리적 해법이라, 지금까지 기각된 "새 정보 추가형" 시도들과 실패 메커니즘이
다르다 -- 실패하더라도 최소한 이 프로젝트의 핵심 미스터리에 대해 답을 준다
(노이즈가 실제로 크다면 배깅이 분산을 줄여 로컬-실제 괴리도 줄여야 함).

설계:
  - 데이터 split(계절 walk-forward의 train/eval 경계, calibration carve-out
    비율)은 세 축(fold0/fold2/GroupKFold) 각각에서 고정(random_state=42로
    고정) -- 시드 배깅 비교에서 "모델 랜덤시드"만 변수가 되도록 격리.
  - SEEDS = [42, 7, 123] (3개), 각 축마다 3개 모델을 독립 학습.
  - 개별 시드 성능(단일 시드 그대로 썼을 때, 이게 곧 지금까지의 production
    방식과 동일)과, 배깅(raw 확률 평균 -> 그 위에 Platt 1회 fit)의 성능을
    비교. 채택 기준: 배깅이 3개 개별 시드의 평균보다 나아야 하고, 두 축
    (fold0/fold2) 전부에서 일관되게 개선되어야 한다. GroupKFold(미본 투수
    축)는 사용자 요청으로 이번 실행에서는 시간 관계상 스킵함 -- 통과해도
    실제 제출 전 별도 확인 권장(두 축만으론 불충분했던 전례 있음, freq974).
  - Platt calibration은 항상 TRAIN 파티션 내부 carve-out에만 fit (정직성
    원칙 준수, eval 정답 미열람).

로컬 저전력 노트북 속도 고려해 iterations=1000
(reg_strength_hypothesis.py와 동일 스케일이라 그 결과와 직접 비교 가능:
baseline calibrated BSS -- fold0=2386.49, fold2=832.41).
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
RESULT_PATH = os.path.join(OUT_DIR, "seed_bagging_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
GROUP_COL = "pitcher_id"
GROUPKFOLD_SPLITS = 3
ITERATIONS = 1000
DATA_SPLIT_SEED = 42  # calib carve-out / groupkfold 그룹 분할은 이 시드로 고정 (모델 시드와 분리)
SEEDS = [42, 7, 123]

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


def run_context(tag, train_df, eval_df):
    """한 평가 컨텍스트(fold0/fold2/groupkfold_i)에 대해 SEEDS 개수만큼
    모델을 학습하고, 개별 시드 성능 + 배깅 성능을 함께 계산한다."""
    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL]

    # data split은 고정 (모델 시드와 분리)
    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=DATA_SPLIT_SEED,
    )
    X_train_sub = build_features(train_sub_df, id_mappings)
    y_train_sub = train_sub_df[TARGET_COL]
    X_calib = build_features(calib_df, id_mappings)
    y_calib = calib_df[TARGET_COL]
    X_eval = build_features(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL]

    cat_idx = [X_train_sub.columns.get_loc(c) for c in CAT_COLS]

    seed_results = []
    calib_raw_preds = []
    eval_raw_preds = []

    for seed in SEEDS:
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
            random_seed=seed, cat_features=cat_idx, early_stopping_rounds=100,
            verbose=False, thread_count=-1, **BEST_PARAMS,
        )
        model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
        elapsed = time.time() - t0

        calib_raw = model.predict_proba(X_calib)[:, 1]
        eval_raw = model.predict_proba(X_eval)[:, 1]
        calib_raw_preds.append(calib_raw)
        eval_raw_preds.append(eval_raw)

        a, b = fit_platt(calib_raw, y_calib)
        eval_calib = apply_platt(eval_raw, a, b)

        seed_result = {
            "seed": seed,
            "best_iteration": model.get_best_iteration(),
            "auc": roc_auc_score(y_eval, eval_raw),
            "bss_raw": bss_score(eval_raw, y_eval),
            "bss_calibrated": bss_score(eval_calib, y_eval),
            "elapsed_sec": elapsed,
        }
        seed_results.append(seed_result)
        print(
            f"  [{tag}/seed{seed}] auc={seed_result['auc']:.4f} "
            f"bss_raw={seed_result['bss_raw']:.2f} bss_calib={seed_result['bss_calibrated']:.2f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    # --- 배깅: raw 확률 평균 -> 그 위에 Platt 1회 fit ---
    calib_raw_bagged = np.mean(calib_raw_preds, axis=0)
    eval_raw_bagged = np.mean(eval_raw_preds, axis=0)
    a_bag, b_bag = fit_platt(calib_raw_bagged, y_calib)
    eval_calib_bagged = apply_platt(eval_raw_bagged, a_bag, b_bag)

    bagged_result = {
        "n_seeds": len(SEEDS),
        "auc": roc_auc_score(y_eval, eval_raw_bagged),
        "bss_raw": bss_score(eval_raw_bagged, y_eval),
        "bss_calibrated": bss_score(eval_calib_bagged, y_eval),
    }
    mean_single_seed_bss = float(np.mean([r["bss_calibrated"] for r in seed_results]))
    std_single_seed_bss = float(np.std([r["bss_calibrated"] for r in seed_results]))

    print(
        f"  [{tag}/BAGGED] auc={bagged_result['auc']:.4f} "
        f"bss_raw={bagged_result['bss_raw']:.2f} bss_calib={bagged_result['bss_calibrated']:.2f}  "
        f"(single-seed mean={mean_single_seed_bss:.2f} +/- {std_single_seed_bss:.2f})",
        flush=True,
    )

    return {
        "tag": tag,
        "n_train": len(train_df), "n_eval": len(eval_df),
        "seeds": seed_results,
        "bagged": bagged_result,
        "mean_single_seed_bss_calibrated": mean_single_seed_bss,
        "std_single_seed_bss_calibrated": std_single_seed_bss,
        "bagging_delta_vs_mean_single_seed": bagged_result["bss_calibrated"] - mean_single_seed_bss,
    }


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}

    print("\n=== fold0 (train<=2021 / eval==2022) ===", flush=True)
    fold0_train = df[df["season"] <= 2021]
    fold0_eval = df[df["season"] == 2022]
    all_results["fold0_2022"] = run_context("fold0_2022", fold0_train, fold0_eval)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== fold2 (train<=2023 / eval==2024) ===", flush=True)
    fold2_train = df[df["season"] <= 2023]
    fold2_eval = df[df["season"] == 2024]
    all_results["fold2_2024"] = run_context("fold2_2024", fold2_train, fold2_eval)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # --- 요약 (GroupKFold는 시간 관계상 스킵, fold0/fold2 두 축만) ---
    summary = {
        "fold0_2022": {
            "single_seed_mean": all_results["fold0_2022"]["mean_single_seed_bss_calibrated"],
            "bagged": all_results["fold0_2022"]["bagged"]["bss_calibrated"],
            "delta": all_results["fold0_2022"]["bagging_delta_vs_mean_single_seed"],
        },
        "fold2_2024": {
            "single_seed_mean": all_results["fold2_2024"]["mean_single_seed_bss_calibrated"],
            "bagged": all_results["fold2_2024"]["bagged"]["bss_calibrated"],
            "delta": all_results["fold2_2024"]["bagging_delta_vs_mean_single_seed"],
        },
    }
    all_positive = all(v["delta"] > 0 for v in summary.values())
    summary["all_axes_positive"] = all_positive
    all_results["summary"] = summary

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (single-seed mean -> bagged, calibrated BSS) ===", flush=True)
    for k, v in summary.items():
        if k == "all_axes_positive":
            continue
        print(f"  {k}: {v['single_seed_mean']:.2f} -> {v['bagged']:.2f}  (delta {v['delta']:+.2f})", flush=True)
    print(f"\n  ALL AXES POSITIVE: {all_positive}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
