"""정규화 강화 가설 검증 (2026-08-21 세션에서 논의만 되고 미착수였던 항목).

가설: "regime shift(연도 간 분포 변화) 상황에서는 강하게 정규화된(더 얕고,
더 강한 l2/bagging/random noise) 모델이 실제 미래(2025 test)에 더 잘
버틸 것이다." 이 프로젝트에서 반증된 다른 실험들과 달리, 이건 특정 피처를
추가하는 게 아니라 모델 자체의 분산을 줄이는 방향이라 성격이 다르다.

채택 기준 (이 프로젝트에서 학습한 교훈 반영, 기존보다 엄격화):
  - fold0/fold2(계절 walk-forward)뿐 아니라 GroupKFold(미본 투수 축)까지
    포함한 "세 축 모두"에서 baseline보다 개선되어야 채택 후보로 고려한다.
    (8/20 freq974 실패: fold0/fold2 이중 일치도 실제 제출에서 반증됐음 ->
    두 축만으로는 불충분하다는 게 이미 확인된 교훈이라 세 번째 축을 추가함)
  - 세 축 중 하나라도 악화되면 기각.

레시피는 es_ws 974 production(train_catboost.py)과 100% 동일하게 유지:
  CAT_COLS(네이티브 cat_features) / RAW_ID_COLS(label-encoded 수치형) /
  trackman_context.pkl 병합. 하이퍼파라미터만 두 가지 비교:
    BASELINE      = 기존 974 BEST_PARAMS (depth=8, l2=20, bagging_temp=1, random_strength=1)
    REG_STRONG    = depth=5, l2=50, bagging_temperature=2.5, random_strength=5.0

정직성 원칙(이 프로젝트에서 반복 확인된 leak 3종 전부 회피):
  1) id_mappings는 각 fold의 TRAIN 파티션에서만 생성 (validation/eval에만
     있는 값은 -1 OOV로 실제 배포와 동일하게 처리).
  2) Platt calibration은 TRAIN 파티션 내부 5% carve-out에만 fit, eval 정답
     은 calibration 단계에서 전혀 안 봄.
  3) feature selection/importance 필터링 없음.

로컬 저전력 노트북(i5-1135G7) 속도 고려해 iterations=1000(이 프로젝트의
다른 ablation들과 동일 관례), GroupKFold는 3-fold로 축소(5-fold 대비 실행
시간 단축, 상대비교 목적이면 3-fold도 축 자체의 신호는 충분히 봄).
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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "reg_strength_hypothesis_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
GROUP_COL = "pitcher_id"
GROUPKFOLD_SPLITS = 3
ITERATIONS = 1000
SEED = 42

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

PARAM_SETS = {
    "baseline": dict(
        learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
        bagging_temperature=1.0, random_strength=1.0, border_count=32,
    ),
    "reg_strong": dict(
        learning_rate=0.01, depth=5, l2_leaf_reg=50.0,
        bagging_temperature=2.5, random_strength=5.0, border_count=32,
    ),
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


def run_train_eval(params, train_df, eval_df, tag):
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

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **params,
    )
    model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0

    calib_pred_raw = model.predict_proba(X_calib)[:, 1]
    a, b = fit_platt(calib_pred_raw, y_calib)

    eval_pred_raw = model.predict_proba(X_eval)[:, 1]
    eval_pred_calib = apply_platt(eval_pred_raw, a, b)

    result = {
        "tag": tag,
        "n_train": len(train_df), "n_eval": len(eval_df),
        "best_iteration": model.get_best_iteration(),
        "auc": roc_auc_score(y_eval, eval_pred_raw),
        "bss_raw": bss_score(eval_pred_raw, y_eval),
        "bss_calibrated": bss_score(eval_pred_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    print(
        f"  [{tag}] n_train={result['n_train']} n_eval={result['n_eval']} "
        f"auc={result['auc']:.4f} bss_raw={result['bss_raw']:.2f} "
        f"bss_calib={result['bss_calibrated']:.2f} ({elapsed:.1f}s)",
        flush=True,
    )
    return result


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}

    for param_name, params in PARAM_SETS.items():
        print(f"\n=== param set: {param_name} = {params} ===", flush=True)
        param_results = {}

        # --- season walk-forward: fold0 (->2022), fold2 (->2024) ---
        fold0_train = df[df["season"] <= 2021]
        fold0_eval = df[df["season"] == 2022]
        param_results["fold0_2022"] = run_train_eval(params, fold0_train, fold0_eval, f"{param_name}/fold0_2022")

        fold2_train = df[df["season"] <= 2023]
        fold2_eval = df[df["season"] == 2024]
        param_results["fold2_2024"] = run_train_eval(params, fold2_train, fold2_eval, f"{param_name}/fold2_2024")

        # --- GroupKFold(pitcher_id), 3-fold ---
        skf = StratifiedGroupKFold(n_splits=GROUPKFOLD_SPLITS, shuffle=True, random_state=SEED)
        groups = df[GROUP_COL].astype(str)
        y_all = df[TARGET_COL]
        gkf_results = []
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(df, y_all, groups)):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)
            gkf_results.append(
                run_train_eval(params, train_df, valid_df, f"{param_name}/groupkfold_{fold_idx}")
            )
        param_results["groupkfold"] = gkf_results
        param_results["groupkfold_bss_calibrated_mean"] = float(
            np.mean([r["bss_calibrated"] for r in gkf_results])
        )

        all_results[param_name] = param_results

        # 중간 저장 (중단되더라도 여기까지는 남도록)
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # --- 세 축 비교 요약 ---
    b = all_results["baseline"]
    r = all_results["reg_strong"]
    summary = {
        "fold0_2022": {
            "baseline": b["fold0_2022"]["bss_calibrated"],
            "reg_strong": r["fold0_2022"]["bss_calibrated"],
            "delta": r["fold0_2022"]["bss_calibrated"] - b["fold0_2022"]["bss_calibrated"],
        },
        "fold2_2024": {
            "baseline": b["fold2_2024"]["bss_calibrated"],
            "reg_strong": r["fold2_2024"]["bss_calibrated"],
            "delta": r["fold2_2024"]["bss_calibrated"] - b["fold2_2024"]["bss_calibrated"],
        },
        "groupkfold_mean": {
            "baseline": b["groupkfold_bss_calibrated_mean"],
            "reg_strong": r["groupkfold_bss_calibrated_mean"],
            "delta": r["groupkfold_bss_calibrated_mean"] - b["groupkfold_bss_calibrated_mean"],
        },
    }
    all_three_positive = all(v["delta"] > 0 for v in summary.values())
    summary["all_three_axes_positive"] = all_three_positive
    all_results["summary"] = summary

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (baseline -> reg_strong, calibrated BSS) ===", flush=True)
    for k, v in summary.items():
        if k == "all_three_axes_positive":
            continue
        print(f"  {k}: {v['baseline']:.2f} -> {v['reg_strong']:.2f}  (delta {v['delta']:+.2f})", flush=True)
    print(f"\n  ALL THREE AXES POSITIVE: {all_three_positive}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
