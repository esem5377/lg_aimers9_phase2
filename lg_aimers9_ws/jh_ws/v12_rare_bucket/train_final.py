"""974점(es_ws v7) 레시피 + 범주형 희귀 레벨 묶기(rare-level bucketing) 전처리 추가.

변경점은 이것 하나뿐 -- 나머지(CAT_COLS 네이티브 cat_features, RAW_ID_COLS
label-encoded 수치형, trackman context, BEST_PARAMS, Platt calibration)는
전부 es_ws/work/pipeline/train_catboost.py(974점)와 동일하게 유지한다.

전처리 내용: train.csv 전수 스캔 결과 다른 범주형 컬럼(top_bottom/game_type/
base_state/pitcher_hand/batter_hand)은 각 값이 최소 3만 건 이상이라 희귀
레벨이 없었지만, pitcher_team_id/batter_team_id는 팀 22(676/773건),
23(4437/4885건), 25(292/292건)가 나머지 10개 팀(13만~21만건)과 비교해
압도적으로 적어 실제로 희귀했다. 이 3개 팀 값을 "OTHER_TEAM" 하나로 묶는다.
묶는 기준(RARE_TEAM_MIN_COUNT=5000)은 전체 train 데이터로 계산하고,
동일한 keep-set을 feature_meta.json에 저장해 script.py 추론 시에도 그대로
적용한다(test.csv에만 있는 새로운 team_id도 자동으로 OTHER_TEAM 처리됨).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
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


# ---- 로컬 경로 (Windows) ----
DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v12_rare_bucket\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v12_rare_bucket\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

RARE_TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
RARE_TEAM_MIN_COUNT = 5000
RARE_TEAM_BUCKET_LABEL = "OTHER_TEAM"

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


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


def build_team_keep_sets(df):
    """min_count 이상 등장하는 team_id만 '유지'하고 나머지는 OTHER_TEAM으로 묶는다."""
    keep_sets = {}
    for c in RARE_TEAM_COLS:
        vc = df[c].astype(str).value_counts()
        keep = sorted(vc[vc >= RARE_TEAM_MIN_COUNT].index.tolist())
        keep_sets[c] = keep
        print(f" {c}: keep={keep} (n_kept_levels={len(keep)}), "
              f"bucketed_to_OTHER={sorted(vc[vc < RARE_TEAM_MIN_COUNT].index.tolist())}")
    return keep_sets


def apply_team_bucketing(series, keep_list):
    keep_set = set(keep_list)
    s = series.astype(str)
    return s.where(s.isin(keep_set), RARE_TEAM_BUCKET_LABEL)


def build_features(df, id_mappings, team_keep_sets):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    for c in RARE_TEAM_COLS:
        X[c] = apply_team_bucketing(X[c], team_keep_sets[c])
    return X


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data...")
    df = load_data()
    print(f" shape={df.shape}")

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  batter_id n={len(id_mappings['batter_id'])}")

    print("Build team rare-level keep-sets (전체 train 기준)...")
    team_keep_sets = build_team_keep_sets(df)

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    X_all = build_features(df, id_mappings, team_keep_sets)
    y_all = df[TARGET_COL]
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[valid_mask], y_all[valid_mask]
    print(f" train={X_tr.shape} valid(2024)={X_va.shape}")

    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

    print("Fit (time-split validation, 참고용 리포트)...")
    model_cv = CatBoostClassifier(
        iterations=500, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=200,
        **BEST_PARAMS,
    )
    model_cv.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    va_pred = model_cv.predict_proba(X_va)[:, 1]
    metrics = {
        "valid_season": 2024,
        "auc": roc_auc_score(y_va, va_pred),
        "logloss": log_loss(y_va, va_pred),
        "accuracy@0.5": accuracy_score(y_va, (va_pred >= 0.5).astype(int)),
        "bss_raw": bss_score(va_pred, y_va),
        "best_iteration": model_cv.get_best_iteration(),
    }
    print("Validation metrics (raw, 보정 전):", json.dumps(metrics, indent=2))

    a, b = fit_platt_scaling(va_pred, y_va)
    va_pred_calib = apply_platt_scaling(va_pred, a, b)
    sk_calib = CalibratedClassifierCV(estimator=FrozenEstimator(model_cv), method="sigmoid")
    sk_calib.fit(X_va, y_va)
    sk_pred_calib = sk_calib.predict_proba(X_va)[:, 1]
    max_abs_diff = float(np.max(np.abs(va_pred_calib - sk_pred_calib)))
    metrics["bss_calibrated"] = bss_score(va_pred_calib, y_va)
    metrics["platt_vs_sklearn_max_abs_diff"] = max_abs_diff
    print(f" 보정 후 BSS(2024, 참고용) = {metrics['bss_calibrated']:.2f} "
          f"(raw {metrics['bss_raw']:.2f} 대비 {metrics['bss_calibrated'] - metrics['bss_raw']:+.2f})")
    with open(os.path.join(OUT_DIR, "metrics_v12_rare_bucket.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nRefit on full data with best_iteration (보정용 5% carve-out 분리)...")
    best_iter = max(model_cv.get_best_iteration(), 1)
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=42,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}")
    model_final = CatBoostClassifier(
        iterations=best_iter, loss_function="Logloss", random_seed=42,
        cat_features=cat_idx, verbose=False,
        **BEST_PARAMS,
    )
    model_final.fit(X_train_final, y_train_final)

    final_calib_pred_raw = model_final.predict_proba(X_calib)[:, 1]
    a_final, b_final = fit_platt_scaling(final_calib_pred_raw, y_calib)
    final_calib_pred = apply_platt_scaling(final_calib_pred_raw, a_final, b_final)
    print(f" carve-out BSS: raw={bss_score(final_calib_pred_raw, y_calib):.2f}  "
          f"calibrated={bss_score(final_calib_pred, y_calib):.2f}")

    model_final.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "rare_team_cols": RARE_TEAM_COLS,
            "rare_team_bucket_label": RARE_TEAM_BUCKET_LABEL,
            "team_keep_sets": team_keep_sets,
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved model to {MODEL_DIR}\\catboost.cbm")


if __name__ == "__main__":
    main()
