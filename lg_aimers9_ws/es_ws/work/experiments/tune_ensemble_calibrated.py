"""실험: jh_ws v9(924점) 방식의 "3모델(LGBM+CatBoost+XGBoost) + 확률보정(sigmoid) +
Optuna 블렌드"를 es_ws 피처셋 위에서 재현.

기존 tune_blend.py(8/16, +0.00015로 기각)는 LightGBM+CatBoost 2모델을 확률보정
없이 그냥 가중 평균만 스윕한 것이었다. 이번 실험은 jh_ws v9와 동일하게
(1) XGBoost를 세 번째 모델로 추가하고
(2) 각 모델에 CalibratedClassifierCV(sigmoid, prefit)로 확률보정을 적용하고
(3) Optuna로 3-way 블렌드 가중치를 탐색
해서 "앙상블+보정" 자체의 효과를 raw id와 분리해서 검증한다 (raw pitcher_id/
batter_id는 이번에도 제외 — train_catboost.py 현재 채택 버전과 동일 피처셋).

CatBoost는 train_catboost.py의 BEST_PARAMS 그대로, LightGBM/XGBoost는
jh_ws/v9_timesplit/train.py에서 쓴 하이퍼파라미터를 그대로 가져왔다.
시간 분할(season<=2023 학습/2024 검증)도 동일.
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

CB_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
LGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", early_stopping_rounds=100, tree_method="hist",
)

BASELINE_AUC = 0.55056  # train_catboost.py 현재 채택 버전 (CatBoost 단일, 보정 없음)
TUNE_BLEND_AUC = 0.55070  # tune_blend.py (LGBM+CatBoost 2모델, 보정 없음, 8/16 기각)


def bss_score(p, y):
    """jh_ws/v9_timesplit/train.py와 동일한 Brier Skill Score.
    AUC는 순위만 보는 지표라 sigmoid 보정(monotonic 변환)의 효과를 못 잡는다.
    실제 대회 리더보드 점수 스케일(738.0, 924 등)이 이 지표와 일치하므로,
    보정의 진짜 효과는 AUC가 아니라 이걸로 봐야 한다.
    """
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def calibrated_pred(base_model, X_va, y_va):
    calib = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="sigmoid")
    calib.fit(X_va, y_va)
    return calib.predict_proba(X_va)[:, 1]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def fit_catboost(X_tr, y_tr, X_va, y_va):
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=200,
        **CB_BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    raw_pred = model.predict_proba(X_va)[:, 1]
    calib_pred = calibrated_pred(model, X_va, y_va)
    return raw_pred, calib_pred


def fit_lgb(X_tr, y_tr, X_va, y_va):
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category").cat.set_categories(X_tr[c].cat.categories)
    model = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1, **LGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    raw_pred = model.predict_proba(X_va)[:, 1]
    calib_pred = calibrated_pred(model, X_va, y_va)
    return raw_pred, calib_pred


def fit_xgb(X_tr, y_tr, X_va, y_va):
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category").cat.set_categories(X_tr[c].cat.categories)
    model = XGBClassifier(enable_categorical=True, **XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    raw_pred = model.predict_proba(X_va)[:, 1]
    calib_pred = calibrated_pred(model, X_va, y_va)
    return raw_pred, calib_pred


def optimize_blend(preds_dict, y_va, n_trials=300, metric="bss"):
    names = list(preds_dict.keys())

    def score(pred):
        return bss_score(pred, y_va) if metric == "bss" else roc_auc_score(y_va, pred)

    def objective(trial):
        ws = {n: trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in names}
        total = sum(ws.values()) + 1e-6
        pred = sum(preds_dict[n] * (ws[n] / total) for n in names)
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        return -score(pred)  # maximize score == minimize -score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_p = study.best_params
    total = sum(best_p.values()) + 1e-6
    weights = {n: best_p[f"w_{n}"] / total for n in names}
    blend = np.clip(sum(preds_dict[n] * weights[n] for n in names), 1e-6, 1 - 1e-6)
    return weights, score(blend)


def main():
    print("Load data...")
    X_tr, y_tr, X_va, y_va = load_data()
    print(f" train={X_tr.shape} valid={X_va.shape}")

    print("\n=== CatBoost ===")
    cb_raw, cb_calib = fit_catboost(X_tr, y_tr, X_va, y_va)
    print(f"catboost raw   auc={roc_auc_score(y_va, cb_raw):.5f}  bss={bss_score(cb_raw, y_va):.2f}")
    print(f"catboost calib auc={roc_auc_score(y_va, cb_calib):.5f}  bss={bss_score(cb_calib, y_va):.2f}")

    print("\n=== LightGBM ===")
    lgb_raw, lgb_calib = fit_lgb(X_tr, y_tr, X_va, y_va)
    print(f"lightgbm raw   auc={roc_auc_score(y_va, lgb_raw):.5f}  bss={bss_score(lgb_raw, y_va):.2f}")
    print(f"lightgbm calib auc={roc_auc_score(y_va, lgb_calib):.5f}  bss={bss_score(lgb_calib, y_va):.2f}")

    print("\n=== XGBoost ===")
    xgb_raw, xgb_calib = fit_xgb(X_tr, y_tr, X_va, y_va)
    print(f"xgboost raw   auc={roc_auc_score(y_va, xgb_raw):.5f}  bss={bss_score(xgb_raw, y_va):.2f}")
    print(f"xgboost calib auc={roc_auc_score(y_va, xgb_calib):.5f}  bss={bss_score(xgb_calib, y_va):.2f}")

    print("\n=== Optuna 3-way blend, BSS 기준 최적화 (보정된 예측 사용) ===")
    weights_calib, blend_bss_calib = optimize_blend(
        {"cb": cb_calib, "lgb": lgb_calib, "xgb": xgb_calib}, y_va, n_trials=300, metric="bss",
    )
    print(f"best weights: {weights_calib}")
    print(f"blend bss = {blend_bss_calib:.2f}")

    print("\n=== Optuna 3-way blend, BSS 기준 최적화 (보정 없는 raw 예측) — 보정 효과 분리용 ===")
    weights_raw, blend_bss_raw = optimize_blend(
        {"cb": cb_raw, "lgb": lgb_raw, "xgb": xgb_raw}, y_va, n_trials=300, metric="bss",
    )
    print(f"best weights: {weights_raw}")
    print(f"blend bss = {blend_bss_raw:.2f}")

    cb_single_auc = roc_auc_score(y_va, cb_raw)
    cb_single_bss = bss_score(cb_raw, y_va)
    blend_auc_calib = roc_auc_score(
        y_va, np.clip(sum(p * w for p, w in zip([cb_calib, lgb_calib, xgb_calib], weights_calib.values())), 1e-6, 1 - 1e-6)
    )
    print("\n=== summary (BSS가 실제 리더보드 지표와 동일 스케일 — 이게 진짜 판단 기준) ===")
    print(f"baseline(현재 채택, CatBoost 단일·보정없음)         auc={BASELINE_AUC:.5f}  bss(이번 실행)={cb_single_bss:.2f}")
    print(f"tune_blend.py(LGBM+CatBoost 2모델, 보정없음, 기각)  auc={TUNE_BLEND_AUC:.5f}")
    print(f"3모델 블렌드, 보정 없음(raw)                        bss={blend_bss_raw:.2f}")
    print(f"3모델 블렌드, sigmoid 보정 적용                     bss={blend_bss_calib:.2f}  auc={blend_auc_calib:.5f}")
    print(f"보정의 순수 효과 (calib블렌드 - raw블렌드)          delta_bss={blend_bss_calib - blend_bss_raw:+.2f}")
    print(f"앙상블+보정 전체 효과 (calib블렌드 - CatBoost단일)  delta_bss={blend_bss_calib - cb_single_bss:+.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_ensemble_calibrated_result.json"), "w") as f:
        json.dump({
            "auc_catboost_raw": float(roc_auc_score(y_va, cb_raw)),
            "auc_catboost_calibrated": float(roc_auc_score(y_va, cb_calib)),
            "auc_lgb_raw": float(roc_auc_score(y_va, lgb_raw)),
            "auc_lgb_calibrated": float(roc_auc_score(y_va, lgb_calib)),
            "auc_xgb_raw": float(roc_auc_score(y_va, xgb_raw)),
            "auc_xgb_calibrated": float(roc_auc_score(y_va, xgb_calib)),
            "bss_catboost_single_raw": float(cb_single_bss),
            "bss_blend_raw_uncalibrated": float(blend_bss_raw),
            "bss_blend_calibrated": float(blend_bss_calib),
            "blend_weights_calibrated": weights_calib,
            "blend_weights_raw": weights_raw,
            "delta_bss_calibration_effect": float(blend_bss_calib - blend_bss_raw),
            "delta_bss_ensemble_plus_calibration_vs_single": float(blend_bss_calib - cb_single_bss),
            "baseline_auc": BASELINE_AUC,
            "tune_blend_auc_no_calibration": TUNE_BLEND_AUC,
        }, f, indent=2)


if __name__ == "__main__":
    main()
