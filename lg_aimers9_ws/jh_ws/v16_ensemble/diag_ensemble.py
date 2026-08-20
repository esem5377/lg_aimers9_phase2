"""974 레시피(트랙맨 컨텍스트 4개 METRIC_COLS + raw id + CAT_COLS) 기준으로
LGBM/XGBoost를 추가해 3모델 앙상블이 실제로 이득인지 재검증.

8/18에는 트랙맨 컨텍스트를 붙이기 *전* 버전(raw id도 없던 더 약한 베이스)
에서 앙상블 기여가 +2.16으로 미미하다고 판단해 제외했었음. raw id+트랙맨
컨텍스트가 들어간 지금(974) 베이스가 이미 훨씬 강해졌으니, 그 위에서
LGBM/XGBoost를 다시 붙여봐도 여전히 미미한지 확인하는 게 목적.

season<=2023 학습 / season==2024 검증(기존 진단과 동일 분할)만 보고 1차
판단 -- GroupKFold는 시간 문제로 스킵(사용자 요청, 8/20 패턴 유지).
calibration은 기존 recipe의 진단 블록과 동일하게 X_va에 직접 fit(참고용
리포트 목적, 기존 805.35 숫자와 비교 가능하게 하기 위함 -- 프로덕션
모델 자체는 항상 정직한 carve-out으로 보정해왔고 이 스크립트는 그
production 단계 이전의 순수 진단용).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "diag_ensemble_results.json")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


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


def build_features(df, id_mappings, cat_dtype):
    """cat_dtype: 'str'(CatBoost용) 또는 'category'(LGBM/XGBoost 네이티브 categorical용)."""
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        if cat_dtype == "str":
            X[c] = X[c].astype(str)
        else:
            X[c] = X[c].astype(str).astype("category")
    return X


def main():
    t0 = time.time()
    print("Load train data...")
    df = load_data()
    print(f" shape={df.shape}")

    id_mappings = build_id_mappings(df)
    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    y_all = df[TARGET_COL]

    # ---- CatBoost (str categorical) ----
    X_all_cb = build_features(df, id_mappings, "str")
    X_tr_cb, y_tr = X_all_cb[train_mask], y_all[train_mask]
    X_va_cb, y_va = X_all_cb[valid_mask], y_all[valid_mask]
    cat_idx = [X_all_cb.columns.get_loc(c) for c in CAT_COLS]

    print(f"\n[CatBoost] fit... ({time.time()-t0:.0f}s elapsed)")
    cat_model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **CAT_BEST_PARAMS,
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va))
    cat_raw = cat_model.predict_proba(X_va_cb)[:, 1]
    print(f" CatBoost best_iter={cat_model.get_best_iteration()} auc={roc_auc_score(y_va, cat_raw):.5f}")

    # ---- LightGBM (category dtype categorical) ----
    X_all_lgb = build_features(df, id_mappings, "category")
    X_tr_lgb, X_va_lgb = X_all_lgb[train_mask], X_all_lgb[valid_mask]

    print(f"\n[LightGBM] fit... ({time.time()-t0:.0f}s elapsed)")
    lgb_model = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(
        X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)],
        categorical_feature=CAT_COLS,
        callbacks=[__import__("lightgbm").early_stopping(100, verbose=False)],
    )
    lgb_raw = lgb_model.predict_proba(X_va_lgb)[:, 1]
    print(f" LightGBM best_iter={lgb_model.best_iteration_} auc={roc_auc_score(y_va, lgb_raw):.5f}")

    # ---- XGBoost (category dtype, enable_categorical) ----
    X_all_xgb = build_features(df, id_mappings, "category")
    X_tr_xgb, X_va_xgb = X_all_xgb[train_mask], X_all_xgb[valid_mask]

    print(f"\n[XGBoost] fit... ({time.time()-t0:.0f}s elapsed)")
    xgb_model = XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=42, early_stopping_rounds=100,
        eval_metric="auc", n_jobs=-1,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    xgb_raw = xgb_model.predict_proba(X_va_xgb)[:, 1]
    print(f" XGBoost best_iter={xgb_model.best_iteration} auc={roc_auc_score(y_va, xgb_raw):.5f}")

    # ---- 개별 Platt 보정 (진단용, X_va에 직접 fit -- 기존 recipe 진단 블록과 동일 관례) ----
    print(f"\nCalibrate + blend search... ({time.time()-t0:.0f}s elapsed)")
    preds_raw = {"cat": cat_raw, "lgb": lgb_raw, "xgb": xgb_raw}
    preds_calib = {}
    for name, raw_p in preds_raw.items():
        a, b = fit_platt_scaling(raw_p, y_va)
        preds_calib[name] = apply_platt_scaling(raw_p, a, b)
        print(f" {name}: bss_raw={bss_score(raw_p, y_va):.2f}  bss_calib={bss_score(preds_calib[name], y_va):.2f}")

    # ---- 블렌드 가중치 그리드서치 (0.05 스텝, 3모델 합=1) ----
    best = {"bss": -1, "w": None}
    grid_results = []
    step = 0.05
    n_steps = int(round(1 / step))
    for i in range(n_steps + 1):
        w_cat = i * step
        for j in range(n_steps + 1 - i):
            w_lgb = j * step
            w_xgb = 1.0 - w_cat - w_lgb
            if w_xgb < -1e-9:
                continue
            blend = w_cat * preds_calib["cat"] + w_lgb * preds_calib["lgb"] + w_xgb * preds_calib["xgb"]
            bss = bss_score(blend, y_va)
            if bss > best["bss"]:
                best = {"bss": bss, "w": {"cat": round(w_cat, 2), "lgb": round(w_lgb, 2), "xgb": round(w_xgb, 2)}}

    single_best_name = max(preds_calib, key=lambda k: bss_score(preds_calib[k], y_va))
    single_best_bss = bss_score(preds_calib[single_best_name], y_va)

    print(f"\n=== 결과 ===")
    print(f" 단일 최고({single_best_name}) calibrated BSS = {single_best_bss:.2f}")
    print(f" 블렌드 최고 BSS = {best['bss']:.2f}  weights={best['w']}")
    print(f" 블렌드 - 단일최고 delta = {best['bss'] - single_best_bss:+.2f}")
    print(f" 참고: 기존 974 레시피(CatBoost 단일) 동일 진단 BSS = 805.35")
    print(f" 블렌드 - 974단일 delta = {best['bss'] - 805.35:+.2f}")

    summary = {
        "single_bss": {k: bss_score(v, y_va) for k, v in preds_calib.items()},
        "single_best_name": single_best_name,
        "single_best_bss": single_best_bss,
        "blend_best_bss": best["bss"],
        "blend_best_weights": best["w"],
        "delta_blend_vs_single_best": best["bss"] - single_best_bss,
        "delta_blend_vs_974_baseline": best["bss"] - 805.35,
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
