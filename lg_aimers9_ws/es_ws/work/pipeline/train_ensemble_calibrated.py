"""CatBoost+LightGBM+XGBoost 3모델 앙상블 + 확률보정 — 프로덕션 후보.

`train_catboost.py`(CatBoost 단일 + 보정, 리더보드 959점 실측)에서 한 단계
더 나아가, `tune_ensemble_calibrated.py` 실험(로컬 BSS: CatBoost단일+보정
825.57 -> 3모델+보정 블렌드 827.73, +2.16)을 프로덕션 파이프라인으로 반영한다.
기존과 동일하게 CalibratedClassifierCV 객체를 직접 pickle하지 않고, 모델별
Platt scaling(a, b) 스칼라만 저장한다. 블렌드 가중치도 시간분할 검증에서
확정한 뒤(재탐색 없이) 최종 재학습에 그대로 고정 사용 — jh_ws train_final.py
와 동일한 패턴.

동일 피처셋(trackman context 포함, raw pitcher_id/batter_id 제외)·동일 시간
분할(2019~2023 학습/2024 검증)을 사용해 `train_catboost.py`와 직접 비교 가능.
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
TRACKMAN_MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model_ensemble"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

CB_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)  # train_catboost.py의 BEST_PARAMS와 동일 (tune_catboost.py 랜덤서치 결과)
LGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)  # jh_ws/v9_timesplit/train.py와 동일
XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", tree_method="hist",
)  # jh_ws/v9_timesplit/train.py와 동일


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(TRACKMAN_MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    return X


def to_cat_str(X):
    X = X.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def to_cat_category(X, categories=None):
    X = X.copy()
    for c in CAT_COLS:
        if categories is None:
            X[c] = X[c].astype("category")
        else:
            X[c] = X[c].astype("category").cat.set_categories(categories[c])
    return X


def fit_catboost(X_tr, y_tr, X_va, y_va, iterations, early_stopping_rounds):
    Xtr, Xva = to_cat_str(X_tr), to_cat_str(X_va)
    cat_idx = [Xtr.columns.get_loc(c) for c in CAT_COLS]
    kwargs = dict(
        iterations=iterations, loss_function="Logloss", random_seed=42,
        cat_features=cat_idx, verbose=False, **CB_PARAMS,
    )
    if early_stopping_rounds:
        kwargs.update(eval_metric="AUC", early_stopping_rounds=early_stopping_rounds)
        model = CatBoostClassifier(**kwargs)
        model.fit(Xtr, y_tr, eval_set=(Xva, y_va))
    else:
        model = CatBoostClassifier(**kwargs)
        model.fit(Xtr, y_tr)
    return model, model.predict_proba(Xva)[:, 1]


def fit_lgbm(X_tr, y_tr, X_va, y_va, n_estimators, early_stopping_rounds, categories=None):
    Xtr = to_cat_category(X_tr, categories)
    Xva = to_cat_category(X_va, categories or {c: Xtr[c].cat.categories for c in CAT_COLS})
    params = dict(LGB_PARAMS)
    params["n_estimators"] = n_estimators
    model = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1, **params)
    if early_stopping_rounds:
        model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    else:
        model.fit(Xtr, y_tr)
    return model, model.predict_proba(Xva)[:, 1]


def fit_xgb(X_tr, y_tr, X_va, y_va, n_estimators, early_stopping_rounds, categories=None):
    Xtr = to_cat_category(X_tr, categories)
    Xva = to_cat_category(X_va, categories or {c: Xtr[c].cat.categories for c in CAT_COLS})
    params = dict(XGB_PARAMS)
    params["n_estimators"] = n_estimators
    if early_stopping_rounds:
        params["early_stopping_rounds"] = early_stopping_rounds
    model = XGBClassifier(enable_categorical=True, **params)
    if early_stopping_rounds:
        model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
    else:
        model.fit(Xtr, y_tr)
    return model, model.predict_proba(Xva)[:, 1]


def optimize_weights_grid(preds, y, step=0.02):
    """cb/lgb/xgb 3-way 가중치를 그리드로 스윕해 BSS 최댓값을 찾는다."""
    best = None
    ws = np.arange(0.0, 1.0001, step)
    for w_cb in ws:
        for w_lgb in np.arange(0.0, 1.0001 - w_cb, step):
            w_xgb = 1.0 - w_cb - w_lgb
            if w_xgb < -1e-9:
                continue
            w_xgb = max(w_xgb, 0.0)
            blend = np.clip(preds["cb"] * w_cb + preds["lgb"] * w_lgb + preds["xgb"] * w_xgb, 1e-6, 1 - 1e-6)
            s = bss_score(blend, y)
            if best is None or s > best[0]:
                best = (s, {"cb": float(w_cb), "lgb": float(w_lgb), "xgb": float(w_xgb)})
    return best[1], best[0]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Load data...")
    df = load_data()
    print(f" shape={df.shape}")

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    X_all = build_features(df)
    y_all = df[TARGET_COL]
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[valid_mask], y_all[valid_mask]
    print(f" train={X_tr.shape} valid(2024)={X_va.shape}")

    print("\n[2/4] 시간분할 검증: 3모델 학습 + Platt 보정 + 블렌드 가중치 탐색...")
    cb_model, cb_raw = fit_catboost(X_tr, y_tr, X_va, y_va, iterations=2000, early_stopping_rounds=100)
    cb_best_iter = cb_model.get_best_iteration()
    print(f" catboost  raw auc={roc_auc_score(y_va, cb_raw):.5f} bss={bss_score(cb_raw, y_va):.2f} best_iter={cb_best_iter}")

    lgb_model, lgb_raw = fit_lgbm(X_tr, y_tr, X_va, y_va, n_estimators=2000, early_stopping_rounds=100)
    lgb_best_iter = lgb_model.best_iteration_
    print(f" lightgbm  raw auc={roc_auc_score(y_va, lgb_raw):.5f} bss={bss_score(lgb_raw, y_va):.2f} best_iter={lgb_best_iter}")

    xgb_model, xgb_raw = fit_xgb(X_tr, y_tr, X_va, y_va, n_estimators=1000, early_stopping_rounds=100)
    xgb_best_iter = xgb_model.best_iteration
    print(f" xgboost   raw auc={roc_auc_score(y_va, xgb_raw):.5f} bss={bss_score(xgb_raw, y_va):.2f} best_iter={xgb_best_iter}")

    ab_cb = fit_platt(cb_raw, y_va)
    ab_lgb = fit_platt(lgb_raw, y_va)
    ab_xgb = fit_platt(xgb_raw, y_va)
    cb_calib = apply_platt(cb_raw, ab_cb)
    lgb_calib = apply_platt(lgb_raw, ab_lgb)
    xgb_calib = apply_platt(xgb_raw, ab_xgb)
    print(f" catboost  calibrated bss={bss_score(cb_calib, y_va):.2f}")
    print(f" lightgbm  calibrated bss={bss_score(lgb_calib, y_va):.2f}")
    print(f" xgboost   calibrated bss={bss_score(xgb_calib, y_va):.2f}")

    weights, blend_bss = optimize_weights_grid({"cb": cb_calib, "lgb": lgb_calib, "xgb": xgb_calib}, y_va)
    cb_solo_bss = bss_score(cb_calib, y_va)
    print(f"\n 최적 블렌드 가중치: {weights}")
    print(f" 블렌드 bss={blend_bss:.2f}  (CatBoost단일+보정 bss={cb_solo_bss:.2f} 대비 {blend_bss - cb_solo_bss:+.2f})")
    print(f" train_catboost.py 실측 대비 -- 이전 로컬 bss(raw)=800.75, calibrated=825.57 (리더보드 889->959로 실측 확인됨)")

    val_summary = {
        "auc": {"cb": roc_auc_score(y_va, cb_raw), "lgb": roc_auc_score(y_va, lgb_raw), "xgb": roc_auc_score(y_va, xgb_raw)},
        "bss_raw": {"cb": bss_score(cb_raw, y_va), "lgb": bss_score(lgb_raw, y_va), "xgb": bss_score(xgb_raw, y_va)},
        "bss_calibrated": {"cb": bss_score(cb_calib, y_va), "lgb": bss_score(lgb_calib, y_va), "xgb": bss_score(xgb_calib, y_va)},
        "blend_weights": weights,
        "blend_bss": blend_bss,
        "delta_vs_catboost_solo_calibrated": blend_bss - cb_solo_bss,
        "best_iterations": {"cb": int(cb_best_iter), "lgb": int(lgb_best_iter), "xgb": int(xgb_best_iter)},
    }
    with open(os.path.join(OUT_DIR, "metrics_ensemble_calibrated.json"), "w") as f:
        json.dump(val_summary, f, indent=2)

    print("\n[3/4] 전체 데이터 재학습 (5% stratified carve-out으로 보정+블렌드 검증)...")
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=42,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}")

    cb_final, cb_calib_raw = fit_catboost(X_train_final, y_train_final, X_calib, y_calib,
                                           iterations=max(cb_best_iter, 1), early_stopping_rounds=0)
    cat_categories = {c: sorted(X_train_final[c].astype(str).unique()) for c in CAT_COLS}
    lgb_final, lgb_calib_raw = fit_lgbm(X_train_final, y_train_final, X_calib, y_calib,
                                         n_estimators=max(lgb_best_iter, 1), early_stopping_rounds=0,
                                         categories=cat_categories)
    xgb_final, xgb_calib_raw = fit_xgb(X_train_final, y_train_final, X_calib, y_calib,
                                        n_estimators=max(xgb_best_iter, 1), early_stopping_rounds=0,
                                        categories=cat_categories)

    ab_cb_final = fit_platt(cb_calib_raw, y_calib)
    ab_lgb_final = fit_platt(lgb_calib_raw, y_calib)
    ab_xgb_final = fit_platt(xgb_calib_raw, y_calib)
    cb_calib_final = apply_platt(cb_calib_raw, ab_cb_final)
    lgb_calib_final = apply_platt(lgb_calib_raw, ab_lgb_final)
    xgb_calib_final = apply_platt(xgb_calib_raw, ab_xgb_final)
    blend_final = np.clip(
        cb_calib_final * weights["cb"] + lgb_calib_final * weights["lgb"] + xgb_calib_final * weights["xgb"],
        1e-6, 1 - 1e-6,
    )
    print(f" carve-out bss: cb_raw={bss_score(cb_calib_raw, y_calib):.2f} cb_calib={bss_score(cb_calib_final, y_calib):.2f} "
          f"blend(frozen weights)={bss_score(blend_final, y_calib):.2f}")

    print("\n[4/4] 아티팩트 저장...")
    cb_final.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))
    joblib.dump(lgb_final, os.path.join(MODEL_DIR, "lgbm.pkl"))
    xgb_final.save_model(os.path.join(MODEL_DIR, "xgboost.json"))
    meta = {
        "columns": list(X_all.columns),
        "cat_cols": CAT_COLS,
        "cat_categories": cat_categories,
        "calibration": {
            "cb": {"a": ab_cb_final[0], "b": ab_cb_final[1]},
            "lgb": {"a": ab_lgb_final[0], "b": ab_lgb_final[1]},
            "xgb": {"a": ab_xgb_final[0], "b": ab_xgb_final[1]},
        },
        "blend_weights": weights,
    }
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {MODEL_DIR}/ (catboost.cbm, lgbm.pkl, xgboost.json, feature_meta.json)")


if __name__ == "__main__":
    main()
