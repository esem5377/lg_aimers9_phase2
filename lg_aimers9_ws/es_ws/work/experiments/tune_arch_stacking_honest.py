"""jh_ws가 "모델간 상관 높아 기대 낮다"고 판단만 하고 실제 검증은 안 했던
'진짜 OOF 스태킹' 가설을 확인. tune_arch_blend_honest.py와 동일한
fold0(->2022)/fold2(->2024) + calib_fit/calib_eval 정직 분리 하네스를
재사용하되, 블렌드 가중치 선택 방식을 두 가지로 비교한다:
  (a) 기존 채택안: 0.05 스텝 constrained grid search (3개 가중치 합=1, 0~1)
  (b) 신규: LogisticRegression(3개 raw 예측을 입력 피처로) 스태킹 --
      가중치 합=1 제약도, 0~1 제약도 없어 그리드서치보다 자유도가 높음
(a)/(b) 둘 다 calib_fit에서 fit, calib_eval에서 평가해 정직하게 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
MODEL_DIR = os.path.join(_BASE, "model")
OUT_DIR = os.path.join(_BASE, "output")

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

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]
CALIB_FIT_SIZE = 0.5


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
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
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
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str) if cat_dtype == "str" else X[c].astype(str).astype("category")
    return X


def grid_search_blend(preds_fit, y_fit):
    best = {"bss": -1, "w": None}
    step = 0.05
    n_steps = int(round(1 / step))
    for i in range(n_steps + 1):
        w_cat = i * step
        for j in range(n_steps + 1 - i):
            w_lgb = j * step
            w_xgb = 1.0 - w_cat - w_lgb
            if w_xgb < -1e-9:
                continue
            blend = w_cat * preds_fit["cat"] + w_lgb * preds_fit["lgb"] + w_xgb * preds_fit["xgb"]
            bss = bss_score(blend, y_fit)
            if bss > best["bss"]:
                best = {"bss": bss, "w": {"cat": round(w_cat, 2), "lgb": round(w_lgb, 2), "xgb": round(w_xgb, 2)}}
    return best["w"]


def run_fold(fold, df, id_mappings):
    t0 = time.time()
    train_mask = df["season"].isin(fold["train_seasons"])
    valid_mask = df["season"] == fold["valid_season"]
    y_all = df[TARGET_COL]
    y_tr = y_all[train_mask]
    print(f"\n===== {fold['name']} n_train={train_mask.sum()} n_valid={valid_mask.sum()} =====")

    va_idx = df.index[valid_mask]
    y_va_all = y_all.loc[va_idx]
    idx_fit, idx_eval = train_test_split(va_idx, train_size=CALIB_FIT_SIZE, stratify=y_va_all, random_state=42)

    X_all_cb = build_features(df, id_mappings, "str")
    X_tr_cb, X_va_cb = X_all_cb.loc[train_mask], X_all_cb.loc[valid_mask]
    y_va = y_all.loc[valid_mask]
    cat_idx = [X_all_cb.columns.get_loc(c) for c in CAT_COLS]
    print(f" [CatBoost] fit... ({time.time()-t0:.0f}s)")
    cat_model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, **CAT_BEST_PARAMS,
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va))
    cat_raw_all = pd.Series(cat_model.predict_proba(X_va_cb)[:, 1], index=X_va_cb.index)
    print(f"  auc={roc_auc_score(y_va, cat_raw_all):.5f} ({time.time()-t0:.0f}s)")

    X_all_lgb = build_features(df, id_mappings, "category")
    X_tr_lgb, X_va_lgb = X_all_lgb.loc[train_mask], X_all_lgb.loc[valid_mask]
    print(f" [LightGBM] fit... ({time.time()-t0:.0f}s)")
    lgb_model = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)], categorical_feature=CAT_COLS,
                  callbacks=[lgb_early_stopping(100, verbose=False)])
    lgb_raw_all = pd.Series(lgb_model.predict_proba(X_va_lgb)[:, 1], index=X_va_lgb.index)
    print(f"  auc={roc_auc_score(y_va, lgb_raw_all):.5f} ({time.time()-t0:.0f}s)")

    X_all_xgb = build_features(df, id_mappings, "category")
    X_tr_xgb, X_va_xgb = X_all_xgb.loc[train_mask], X_all_xgb.loc[valid_mask]
    print(f" [XGBoost] fit... ({time.time()-t0:.0f}s)")
    xgb_model = XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=42, early_stopping_rounds=100,
        eval_metric="auc", n_jobs=-1,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    xgb_raw_all = pd.Series(xgb_model.predict_proba(X_va_xgb)[:, 1], index=X_va_xgb.index)
    print(f"  auc={roc_auc_score(y_va, xgb_raw_all):.5f} ({time.time()-t0:.0f}s)")

    raw_all = {"cat": cat_raw_all, "lgb": lgb_raw_all, "xgb": xgb_raw_all}
    y_fit, y_eval = y_all.loc[idx_fit], y_all.loc[idx_eval]

    # 개별 Platt (calib_fit에서)
    ab = {}
    fit_platt_preds = {}
    for name, raw in raw_all.items():
        a, b = fit_platt_scaling(raw.loc[idx_fit], y_fit)
        ab[name] = (a, b)
        fit_platt_preds[name] = apply_platt_scaling(raw.loc[idx_fit], a, b)
    eval_platt_preds = {name: apply_platt_scaling(raw.loc[idx_eval], *ab[name]) for name, raw in raw_all.items()}
    single_bss_eval = {k: bss_score(v, y_eval) for k, v in eval_platt_preds.items()}
    single_best = max(single_bss_eval, key=single_bss_eval.get)

    # (a) 기존: constrained grid search (calibrated 예측 위에서)
    grid_w = grid_search_blend(fit_platt_preds, y_fit)
    grid_eval = sum(grid_w[k] * eval_platt_preds[k] for k in ("cat", "lgb", "xgb"))
    grid_bss_eval = bss_score(grid_eval, y_eval)

    # (b) 신규: LogisticRegression 스태킹 (raw 예측 3개를 피처로, calib_fit에서 fit)
    X_stack_fit = np.column_stack([raw_all[k].loc[idx_fit] for k in ("cat", "lgb", "xgb")])
    X_stack_eval = np.column_stack([raw_all[k].loc[idx_eval] for k in ("cat", "lgb", "xgb")])
    stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    stacker.fit(X_stack_fit, y_fit)
    stack_eval_pred = stacker.predict_proba(X_stack_eval)[:, 1]
    stack_bss_eval = bss_score(stack_eval_pred, y_eval)
    stack_coef = {"cat": float(stacker.coef_[0][0]), "lgb": float(stacker.coef_[0][1]),
                  "xgb": float(stacker.coef_[0][2]), "intercept": float(stacker.intercept_[0])}

    dt = time.time() - t0
    print(f" 단일최고({single_best})={single_bss_eval[single_best]:.2f}")
    print(f" (a) grid blend={grid_bss_eval:.2f} (w={grid_w})  delta={grid_bss_eval - single_bss_eval[single_best]:+.2f}")
    print(f" (b) LR stacking={stack_bss_eval:.2f} (coef={stack_coef})  delta={stack_bss_eval - single_bss_eval[single_best]:+.2f}")
    print(f" (b)-(a) delta = {stack_bss_eval - grid_bss_eval:+.2f}  (fold {dt:.0f}s)")

    return {
        "fold": fold["name"],
        "single_bss_eval": single_bss_eval,
        "single_best": single_best,
        "grid_weights": grid_w,
        "grid_bss_eval": grid_bss_eval,
        "grid_delta": grid_bss_eval - single_bss_eval[single_best],
        "stack_coef": stack_coef,
        "stack_bss_eval": stack_bss_eval,
        "stack_delta": stack_bss_eval - single_bss_eval[single_best],
        "stack_vs_grid_delta": stack_bss_eval - grid_bss_eval,
        "elapsed_sec": dt,
    }


def main():
    t0 = time.time()
    print("Load data...")
    df = load_data()
    print(f" shape={df.shape}")
    id_mappings = build_id_mappings(df)

    results = [run_fold(f, df, id_mappings) for f in FOLDS]

    print("\n=== 요약: (b) LR stacking - (a) grid blend ===")
    for r in results:
        print(f"  {r['fold']}: stack_vs_grid={r['stack_vs_grid_delta']:+.2f}  "
              f"(grid_delta={r['grid_delta']:+.2f}, stack_delta={r['stack_delta']:+.2f})")
    both_stack_better = all(r["stack_vs_grid_delta"] > 0 for r in results)
    print(f"  fold0 & fold2 모두 stacking이 grid보다 나음 = {both_stack_better}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_arch_stacking_honest_results.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": results, "both_stack_better": both_stack_better}, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_arch_stacking_honest_results.json")
    print(f"총 소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
