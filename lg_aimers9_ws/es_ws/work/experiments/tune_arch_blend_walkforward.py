"""jh_ws v16_ensemble/diag_ensemble.py(CatBoost+LGBM+XGBoost 블렌드, +2.90,
season<=2023/==2024 단일 분할)를 fold0(->2022)까지 확장해 walk-forward로
재검증.

배경: 팀 현재 최고는 982점(jh_ws v18_seed_bagging, 974 레시피를 6시드로
평균낸 순수 분산감소 기법 -- 새 정보 추가 없음). jh_ws 결론: "새 피처
추가류는 사실상 다 막혔다", 남은 후보로 "진짜 OOF 스태킹"은 상관성이 높아
기대 낮다고 봄. 다만 diag_ensemble.py의 아키텍처 블렌드(CatBoost 0.85 :
LGBM 0.05 : XGBoost 0.10, +2.90)는 season<=2023/==2024 단일 분할 1회만
확인됐고, 이 프로젝트 전체 히스토리(raw id, calibration, seed bagging 등)에서
"실제로 리더보드에 살아남은 개선"은 전부 fold0(->2022)/fold2(->2024) 두
독립 폴드에서 같은 방향으로 확인된 것들뿐이었다(단일 폴드 신호는 여러 번
반증됨: regime2023, freq974 등). 이 스크립트는 그 기준을 아키텍처 블렌드에도
적용해 fold0에서도 유지되는지 확인한다.

주의(진단 한계, jh_ws 원본과 동일한 관례 유지): 블렌드 가중치 그리드서치를
평가셋(X_va)에 직접 fit하는 점은 낙관 편향이 있는 진단용 방법론이다(production
처럼 별도 carve-out에서 가중치를 고르지 않음). 두 폴드 모두에서 낙관 편향을
동일하게 받으므로, 그럼에도 fold0/fold2가 불일치하면 신호가 약하다는 뜻이고
일치해도 "진짜 이득의 상한 추정"으로만 해석해야 함 -- 실제 채택 전엔 반드시
production 스타일의 정직한 carve-out 가중치 탐색으로 재확인 필요.
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

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../es_ws/work
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
        if cat_dtype == "str":
            X[c] = X[c].astype(str)
        else:
            X[c] = X[c].astype(str).astype("category")
    return X


def run_fold(fold, df, id_mappings):
    t0 = time.time()
    train_mask = df["season"].isin(fold["train_seasons"])
    valid_mask = df["season"] == fold["valid_season"]
    y_all = df[TARGET_COL]
    y_tr, y_va = y_all[train_mask], y_all[valid_mask]

    print(f"\n===== {fold['name']} n_train={train_mask.sum()} n_valid={valid_mask.sum()} =====")

    # ---- CatBoost ----
    X_all_cb = build_features(df, id_mappings, "str")
    X_tr_cb, X_va_cb = X_all_cb[train_mask], X_all_cb[valid_mask]
    cat_idx = [X_all_cb.columns.get_loc(c) for c in CAT_COLS]
    print(f" [CatBoost] fit... ({time.time()-t0:.0f}s elapsed)")
    cat_model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **CAT_BEST_PARAMS,
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va))
    cat_raw = cat_model.predict_proba(X_va_cb)[:, 1]
    print(f"  best_iter={cat_model.get_best_iteration()} auc={roc_auc_score(y_va, cat_raw):.5f} ({time.time()-t0:.0f}s)")

    # ---- LightGBM ----
    X_all_lgb = build_features(df, id_mappings, "category")
    X_tr_lgb, X_va_lgb = X_all_lgb[train_mask], X_all_lgb[valid_mask]
    print(f" [LightGBM] fit... ({time.time()-t0:.0f}s elapsed)")
    lgb_model = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(
        X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)],
        categorical_feature=CAT_COLS,
        callbacks=[lgb_early_stopping(100, verbose=False)],
    )
    lgb_raw = lgb_model.predict_proba(X_va_lgb)[:, 1]
    print(f"  best_iter={lgb_model.best_iteration_} auc={roc_auc_score(y_va, lgb_raw):.5f} ({time.time()-t0:.0f}s)")

    # ---- XGBoost ----
    X_all_xgb = build_features(df, id_mappings, "category")
    X_tr_xgb, X_va_xgb = X_all_xgb[train_mask], X_all_xgb[valid_mask]
    print(f" [XGBoost] fit... ({time.time()-t0:.0f}s elapsed)")
    xgb_model = XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=42, early_stopping_rounds=100,
        eval_metric="auc", n_jobs=-1,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    xgb_raw = xgb_model.predict_proba(X_va_xgb)[:, 1]
    print(f"  best_iter={xgb_model.best_iteration} auc={roc_auc_score(y_va, xgb_raw):.5f} ({time.time()-t0:.0f}s)")

    # ---- 개별 Platt 보정 (jh_ws 원본과 동일: X_va에 직접 fit, 진단용) ----
    preds_raw = {"cat": cat_raw, "lgb": lgb_raw, "xgb": xgb_raw}
    preds_calib = {}
    for name, raw_p in preds_raw.items():
        a, b = fit_platt_scaling(raw_p, y_va)
        preds_calib[name] = apply_platt_scaling(raw_p, a, b)
        print(f"  {name}: bss_raw={bss_score(raw_p, y_va):.2f}  bss_calib={bss_score(preds_calib[name], y_va):.2f}")

    # ---- 블렌드 가중치 그리드서치 (0.05 스텝) ----
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
            blend = w_cat * preds_calib["cat"] + w_lgb * preds_calib["lgb"] + w_xgb * preds_calib["xgb"]
            bss = bss_score(blend, y_va)
            if bss > best["bss"]:
                best = {"bss": bss, "w": {"cat": round(w_cat, 2), "lgb": round(w_lgb, 2), "xgb": round(w_xgb, 2)}}

    single_best_name = max(preds_calib, key=lambda k: bss_score(preds_calib[k], y_va))
    single_best_bss = bss_score(preds_calib[single_best_name], y_va)
    delta = best["bss"] - single_best_bss
    dt = time.time() - t0
    print(f" 단일 최고({single_best_name}) calib BSS = {single_best_bss:.2f}")
    print(f" 블렌드 최고 BSS = {best['bss']:.2f}  weights={best['w']}")
    print(f" 블렌드 - 단일최고 delta = {delta:+.2f}  (fold 소요 {dt:.0f}s)")

    return {
        "fold": fold["name"],
        "single_bss": {k: bss_score(v, y_va) for k, v in preds_calib.items()},
        "single_best_name": single_best_name,
        "single_best_bss": single_best_bss,
        "blend_best_bss": best["bss"],
        "blend_best_weights": best["w"],
        "delta_blend_vs_single_best": delta,
        "elapsed_sec": dt,
    }


def main():
    t0 = time.time()
    print("Load data...")
    df = load_data()
    print(f" shape={df.shape}")
    id_mappings = build_id_mappings(df)

    results = [run_fold(f, df, id_mappings) for f in FOLDS]

    print("\n=== 요약 (blend - 단일최고 delta) ===")
    for r in results:
        print(f"  {r['fold']}: delta={r['delta_blend_vs_single_best']:+.2f}  weights={r['blend_best_weights']}")
    both_positive = all(r["delta_blend_vs_single_best"] > 0 for r in results)
    print(f"  fold0 & fold2 모두 양수 = {both_positive}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_arch_blend_walkforward_results.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": results, "both_positive": both_positive, "total_elapsed_sec": time.time() - t0}, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_arch_blend_walkforward_results.json")
    print(
        "\n판단 기준: fold0(->2022)과 fold2(->2024) 둘 다 독립적으로 양수여야 "
        "채택 후보(단, 위 docstring의 진단 한계 -- 가중치를 eval셋에 직접 fit한 낙관편향 -- "
        "감안해 production 반영 전 정직한 carve-out 재검증 필수)."
    )


if __name__ == "__main__":
    main()
