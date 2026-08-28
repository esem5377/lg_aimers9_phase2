"""987점 베이스(v22_drop_ingredients_1seed: control_risk_score/_weighted 추가+
원재료(reverse/middle/ball_rate) 3개 제거, 70피처) 위에, 8/20 tune_lowpriority.py에서
검증했던 "corr>0.95 다중공선성 제거 + LGBM importance==0 제거" pruning을 적용.

주의(중요, 사용자에게 명시적으로 고지 완료 후 진행):
  - 8/20 원본 검증은 974점 레시피(71피처, control_risk_score 없음) 위에서 fold0 -13.41/
    fold2 -1.23으로 "둘 다 음수"로 기각됐던 실험. 이 프로젝트에서 "두 폴드 다 음수"였던
    다른 모든 실험(정규화강화/Monotonic/Lossguide/team target encoding 등)은 지금까지
    단 한 번도 실제 제출까지 간 적이 없음 -- 이번이 그 패턴의 첫 예외.
  - 게다가 987 베이스(control_risk_score 추가 후)에서는 pruning 자체가 재검증된 적이
    없음(8/20 실험은 974 베이스). 사용자가 재검증 생략하고 바로 987 베이스에 적용해
    제출 준비하는 것을 명시적으로 선택함(2026-08-25).

pruning 방식(8/20 tune_lowpriority.py의 variant_pruned()와 동일 로직, fold 대신
train_final(95% carve-out 이후) 파티션에 fit):
  - CAT_COLS를 제외한 수치형 컬럼(raw id 포함) 간 상관계수 절대값 > 0.95인 쌍에서
    하나를 제거(다중공선성)
  - 남은 컬럼으로 LGBMClassifier(n_estimators=100)를 빠르게 학습해 feature_importances_
    가 0인 컬럼 추가 제거
  - selector는 calibration carve-out(5%) 정답을 전혀 안 봄(train_final 95%에만 fit)

나머지(CAT_COLS 네이티브/RAW_ID_COLS label-encoded/trackman context/BEST_PARAMS/Platt
calibration)는 v20/v22와 동일.
"""
import json
import os
import time

import joblib
import lightgbm as lgb
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
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v30_pruning_987base_1seed\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v30_pruning_987base_1seed\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

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


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    df = df.drop(columns=INGREDIENT_COLS)
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


def prune_features(X_train_final, y_train_final, cat_cols):
    """8/20 variant_pruned()와 동일 로직: corr>0.95 제거 -> LGBM importance==0 제거.
    train_final(95%, calibration carve-out 제외) 파티션에만 fit."""
    num_cols = [c for c in X_train_final.columns if c not in cat_cols]
    corr = X_train_final[num_cols].astype(float).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
    print(f" corr>0.95 제거 대상({len(to_drop_corr)}개): {to_drop_corr}", flush=True)

    kept_cols = [c for c in X_train_final.columns if c not in to_drop_corr]
    X_tmp = X_train_final[kept_cols].copy()
    for c in cat_cols:
        if c in X_tmp.columns:
            X_tmp[c] = X_tmp[c].astype("category")
    sel = lgb.LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1)
    sel.fit(X_tmp, y_train_final, categorical_feature=[c for c in cat_cols if c in kept_cols])
    importances = pd.Series(sel.feature_importances_, index=kept_cols)
    to_drop_imp0 = importances[importances == 0].index.tolist()
    print(f" LGBM importance==0 제거 대상({len(to_drop_imp0)}개): {to_drop_imp0}", flush=True)

    selected = importances[importances > 0].index.tolist()
    selected_cat = [c for c in cat_cols if c in selected]
    return selected, selected_cat, to_drop_corr, to_drop_imp0


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data (전체) + control_risk_score(*2) 추가, 원재료 3개 제거...", flush=True)
    df = load_data()
    df = add_risk_score_drop_ingredients(df)
    print(f" shape={df.shape}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    print(f" n_features(pruning 전)={X_all.shape[1]} (v22의 70)", flush=True)

    print("\nCalibration carve-out(5%) 분리...", flush=True)
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}", flush=True)

    print("\nPruning (corr>0.95 + LGBM importance==0), train_final(95%)에만 fit...", flush=True)
    t0 = time.time()
    selected, selected_cat, dropped_corr, dropped_imp0 = prune_features(X_train_final, y_train_final, CAT_COLS)
    print(f" pruning 완료 ({time.time()-t0:.1f}s): "
          f"{X_train_final.shape[1]} -> {len(selected)}피처 "
          f"(corr제거 {len(dropped_corr)}개 + importance0제거 {len(dropped_imp0)}개)", flush=True)
    print(f" 최종 선택 피처: {selected}", flush=True)

    X_train_final = X_train_final[selected]
    X_calib = X_calib[selected]
    cat_idx = [X_train_final.columns.get_loc(c) for c in selected_cat]

    print(f"\n=== seed={SEED} 학습 (iterations={ITERATIONS}, n_features={len(selected)}) ===", flush=True)
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
        "n_features_before_pruning": int(X_all.shape[1]),
        "n_features_after_pruning": len(selected),
        "dropped_corr": dropped_corr,
        "dropped_importance0": dropped_imp0,
        "carveout_bss_raw": bss_score(calib_raw, y_calib),
        "carveout_bss_calibrated": bss_score(calib_pred, y_calib),
    }
    print(f" carve-out BSS: raw={metrics['carveout_bss_raw']:.2f}  "
          f"calibrated={metrics['carveout_bss_calibrated']:.2f}", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v30_pruning.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model_path = os.path.join(MODEL_DIR, f"catboost_seed{SEED}.cbm")
    model.save_model(model_path)
    print(f" saved: {model_path}", flush=True)

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": selected,
            "cat_cols": selected_cat,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "seeds": [SEED],
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved 1 model + feature_meta.json to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
