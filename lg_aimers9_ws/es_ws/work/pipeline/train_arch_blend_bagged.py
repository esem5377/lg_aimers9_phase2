"""982점(jh_ws v18_seed_bagging, 6시드 CatBoost 배깅) 위에 LGBM/XGBoost
배깅을 소량 블렌드하는 프로덕션 후보.

배경: `es_ws/work/experiments/tune_arch_blend_honest.py`로 CatBoost+LGBM+
XGBoost 아키텍처 블렌드를 fold0(->2022)/fold2(->2024) walk-forward +
calib_fit/calib_eval 정직 분리로 검증 -- fold0 delta=+4.57, fold2
delta=+1.37, 둘 다 양수(가중치 대략 cat 0.7~0.85 / lgb 0.1~0.15 /
xgb 0.05~0.15). 새 피처/정보를 추가하는 게 아니라 시드 배깅과 같은 계열
(모델 분산 감소)이라는 점에서 리스크가 낮다고 판단해 프로덕션 후보로
진행. 단, 이 프로젝트엔 fold0/fold2 정직 이중 일치도 실제 리더보드에서
반증된 전례(freq974, 새 피처 도입 케이스)가 있으므로 낙관은 금물 --
반드시 리더보드로 실측 확인 필요.

구성: jh_ws v18_seed_bagging의 기존 6시드 CatBoost 모델(재학습 없이 그대로
로드, 982점 검증된 자산)에, 동일 레시피로 새로 학습한 LGBM 3시드(42/7/123)
+ XGBoost 3시드(42/7/123) 배깅 평균을 추가해 3-way 블렌드. 블렌드 가중치와
Platt 보정은 5% calibration carve-out(기존 프로덕션과 동일 관례: 전체
데이터에서 stratified 5% 분리, 나머지 95%로 학습)에서 함께 fit.
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
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../es_ws/work
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
JH_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(_BASE)), "jh_ws", "v18_seed_bagging", "model")
OUT_MODEL_DIR = os.path.join(_BASE, "model_arch_blend")
OUT_DIR = os.path.join(_BASE, "output")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

LGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.02, num_leaves=63,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, verbosity=-1,
)
XGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.02, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    enable_categorical=True, n_jobs=-1,
)
NEW_SEEDS = [42, 7, 123]
DATA_SPLIT_SEED = 42


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
    context = joblib.load(os.path.join(_BASE, "model", "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


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
    return best["w"], best["bss"]


def main():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("Load jh_ws v18 feature_meta (id_mappings, CatBoost 6시드 재사용)...")
    with open(os.path.join(JH_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        jh_meta = json.load(f)
    id_mappings = jh_meta["id_mappings"]
    cat_seeds = jh_meta["seeds"]
    cat_models = []
    for seed in cat_seeds:
        m = CatBoostClassifier()
        m.load_model(os.path.join(JH_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cat_models.append(m)
    print(f" CatBoost {len(cat_models)}개 모델 로드 완료 (seeds={cat_seeds})")

    print("\nLoad train data...")
    df = load_data()
    print(f" shape={df.shape}  ({time.time()-t0:.0f}s)")
    y_all = df[TARGET_COL]

    X_all_str = build_features(df, id_mappings, "str")
    X_all_cat = build_features(df, id_mappings, "category")
    cat_idx = [X_all_str.columns.get_loc(c) for c in CAT_COLS]

    print("\nCalibration carve-out(5%) 분리 (jh_ws 프로덕션과 동일 관례)...")
    idx_train, idx_calib = train_test_split(
        df.index, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED)
    print(f" train={len(idx_train)}  calib={len(idx_calib)}  ({time.time()-t0:.0f}s)")

    X_tr_str, X_tr_cat = X_all_str.loc[idx_train], X_all_cat.loc[idx_train]
    y_tr = y_all.loc[idx_train]
    X_calib_str, X_calib_cat = X_all_str.loc[idx_calib], X_all_cat.loc[idx_calib]
    y_calib = y_all.loc[idx_calib]

    print("\n=== CatBoost 6시드 배깅 예측 (calib carve-out, 재학습 없음) ===")
    cat_calib_raw = np.mean([m.predict_proba(X_calib_str)[:, 1] for m in cat_models], axis=0)
    print(f" cat bagged raw BSS(calib) = {bss_score(cat_calib_raw, y_calib):.2f}  ({time.time()-t0:.0f}s)")

    print("\n=== LightGBM 신규 3시드 학습 ===")
    lgb_models = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m.fit(X_tr_cat, y_tr, categorical_feature=CAT_COLS)
        lgb_models.append(m)
        m.booster_.save_model(os.path.join(OUT_MODEL_DIR, f"lgb_seed{seed}.txt"))
        print(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    lgb_calib_raw = np.mean([m.predict_proba(X_calib_cat)[:, 1] for m in lgb_models], axis=0)
    print(f" lgb bagged raw BSS(calib) = {bss_score(lgb_calib_raw, y_calib):.2f}  ({time.time()-t0:.0f}s)")

    print("\n=== XGBoost 신규 3시드 학습 ===")
    xgb_models = []
    for seed in NEW_SEEDS:
        ts = time.time()
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(X_tr_cat, y_tr)
        xgb_models.append(m)
        m.save_model(os.path.join(OUT_MODEL_DIR, f"xgb_seed{seed}.json"))
        print(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    xgb_calib_raw = np.mean([m.predict_proba(X_calib_cat)[:, 1] for m in xgb_models], axis=0)
    print(f" xgb bagged raw BSS(calib) = {bss_score(xgb_calib_raw, y_calib):.2f}  ({time.time()-t0:.0f}s)")

    print("\n=== 블렌드 가중치: calib carve-out에서 재탐색하지 않고 walk-forward(honest) 결과 그대로 사용 ===")
    print(" (calib carve-out은 전체 2019~2024에서 무작위 5% -- train과 선수풀/분포를 공유해 실제로 돌려보니")
    print("  XGBoost가 CatBoost를 역전(2115 vs 2057)하고 가중치가 xgb 0.85로 쏠리는, 이 프로젝트가 이미 한번")
    print("  겪은 '무작위분할 OOF가 미래시즌 일반화를 과대평가' 함정과 동일 패턴이 재현돼 신뢰 안 함.")
    print(" 대신 tune_arch_blend_honest.py의 fold2(->2024, 가장 최근/성숙 데이터 레짐) 가중치를 채택.")
    best_w = {"cat": 0.85, "lgb": 0.1, "xgb": 0.05}
    raw_all = {
        "cat": pd.Series(cat_calib_raw, index=idx_calib),
        "lgb": pd.Series(lgb_calib_raw, index=idx_calib),
        "xgb": pd.Series(xgb_calib_raw, index=idx_calib),
    }
    print(f" 채택 가중치(fold2 walk-forward honest) = {best_w}")

    print("\n=== 최종 Platt: 전체 calib(5%)에 개별 모델 재보정 -> 프로덕션에 사용 ===")
    ab_final = {}
    fit_preds_full = {}
    for name, raw in raw_all.items():
        a, b = fit_platt_scaling(raw, y_calib)
        ab_final[name] = (a, b)
        fit_preds_full[name] = apply_platt_scaling(raw, a, b)
    blend_full_raw = sum(best_w[k] * raw_all[k] for k in ("cat", "lgb", "xgb"))
    # 블렌드 자체에 최종 Platt 1회 (개별 calib 후 blend가 아니라, jh_ws 관례처럼 blend 후 보정 1회로 단순화)
    a_blend, b_blend = fit_platt_scaling(blend_full_raw, y_calib)
    blend_full_calib = apply_platt_scaling(blend_full_raw, a_blend, b_blend)
    print(f" 전체 calib(5%) 위 최종 확인: raw={bss_score(blend_full_raw, y_calib):.2f}  "
          f"calibrated={bss_score(blend_full_calib, y_calib):.2f}")

    meta = {
        "cat_cols": CAT_COLS,
        "raw_id_cols": RAW_ID_COLS,
        "id_mappings": id_mappings,
        "cat_seeds": cat_seeds,
        "lgb_seeds": NEW_SEEDS,
        "xgb_seeds": NEW_SEEDS,
        "blend_weights": best_w,
        "calibration": {"method": "platt_sigmoid_on_blend", "a": a_blend, "b": b_blend},
        "columns_str": list(X_all_str.columns),
        "columns_cat": list(X_all_cat.columns),
        "honest_walkforward_check": {
            "fold0_delta": 4.57, "fold2_delta": 1.37,
            "source": "es_ws/work/experiments/tune_arch_blend_honest.py",
        },
        "note_random_carveout_weight_search_rejected": (
            "5% 무작위 calib carve-out으로 블렌드 가중치를 직접 grid search했더니 "
            "xgb 0.85로 쏠리고 cat<xgb로 역전(2115 vs 2057) -- train과 선수풀을 공유하는 "
            "무작위분할이라 미래시즌 일반화를 과대평가하는 것으로 판단해 기각, "
            "walk-forward(honest) 가중치를 대신 채택함."
        ),
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved feature_meta.json + lgb/xgb model files to {OUT_MODEL_DIR}")
    print(f"총 소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
