"""진짜 OOF(out-of-fold) 스태킹 -- 지금까지 "상관성 높아 기대 낮음"이라 우선순위만
낮게 두고 실제 실행은 안 했던 마지막 후보. 팀원의 아키텍처 블렌드(고정 가중치
그리드서치, 986점)와 다르게, 메타러너(LogisticRegression)가 3개 베이스모델의
OOF 예측을 보고 스스로 조합 방식을 학습함 -- 단순 가중평균보다 표현력이 큼.

절차(각 fold별로):
  1) train 파티션을 3-fold(StratifiedKFold)로 나눠 각 fold마다 CatBoost/LGBM/
     XGBoost를 학습 -> 나머지 파티션에 대한 OOF 예측 생성(leakage 없음).
  2) 메타러너(LogisticRegression)를 OOF 예측 3컬럼 -> 정답으로 학습.
  3) 3개 베이스모델을 train 파티션 전체로 재학습 -> eval(계절 홀드아웃)에 예측.
  4) 메타러너로 eval 예측 결합 -> Platt calibration(carve-out) 적용 -> BSS.

베이스 피처셋: 992 레시피(control_risk_score/_weighted 포함, 원재료 유지,
73피처) -- 현재 실제 프로덕션(v20)과 동일.
LGBM은 오늘 세션 Optuna 탐색(optuna_lgbm_search_result.json)에서 찾은 튜닝
파라미터 사용(기존 미조정 baseline보다 로컬 +109 확인됨). XGBoost는 이
프로젝트에서 전용 튜닝이 한 번도 없어 팀원 레시피의 합리적 기본값 사용.
비용 절감을 위해 OOF 생성 단계는 iterations/n_estimators=500(3-fold), 최종
재학습은 1000(이 세션 다른 검증들과 동일 스케일).
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
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "true_oof_stacking_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
N_OOF_FOLDS = 3
OOF_ITERATIONS = 500
FINAL_ITERATIONS = 1000

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
# 오늘 세션 optuna_lgbm_search_result.json의 best_params (baseline 대비 로컬 +109 확인됨)
LGB_TUNED_PARAMS = dict(
    num_leaves=32, learning_rate=0.014795555938724558, min_child_samples=99,
    subsample=0.5557809026152152, colsample_bytree=0.9125866996530361,
    reg_lambda=0.00472692386317043, reg_alpha=0.011169231844620464,
)
# XGBoost 전용 튜닝 이력 없음 -- 팀원 arch_blend 레시피의 합리적 기본값 재사용
XGB_PARAMS = dict(
    learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist",
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


def add_risk_score(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df


def build_id_mappings(train_df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(train_df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_features_cat(df, id_mappings):
    """CatBoost용: cat_cols는 문자열 그대로."""
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def build_features_gbm(df, id_mappings):
    """LGBM/XGBoost용: cat_cols는 category dtype."""
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str).astype("category")
    return X


def fit_cat(X_tr, y_tr):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    m = CatBoostClassifier(
        iterations=OOF_ITERATIONS, loss_function="Logloss", random_seed=SEED,
        cat_features=cat_idx, verbose=False, thread_count=-1, **CAT_BEST_PARAMS,
    )
    m.fit(X_tr, y_tr)
    return m


def fit_lgb(X_tr, y_tr):
    m = LGBMClassifier(
        n_estimators=OOF_ITERATIONS, random_state=SEED, n_jobs=-1, verbosity=-1,
        **LGB_TUNED_PARAMS,
    )
    m.fit(X_tr, y_tr, categorical_feature=CAT_COLS)
    return m


def fit_xgb(X_tr, y_tr):
    m = XGBClassifier(
        n_estimators=OOF_ITERATIONS, random_state=SEED, n_jobs=-1,
        enable_categorical=True, eval_metric="logloss", **XGB_PARAMS,
    )
    m.fit(X_tr, y_tr)
    return m


def run_fold(fold_name, train_df, eval_df):
    t0 = time.time()
    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL].values

    Xc_train = build_features_cat(train_df, id_mappings)
    Xg_train = build_features_gbm(train_df, id_mappings)
    Xc_eval = build_features_cat(eval_df, id_mappings)
    Xg_eval = build_features_gbm(eval_df, id_mappings)
    y_eval = eval_df[TARGET_COL].values

    n = len(train_df)
    oof_cat = np.zeros(n)
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)

    print(f" [{fold_name}] OOF 생성 시작 ({N_OOF_FOLDS}-fold, n_train={n}) ({time.time()-t0:.0f}s)", flush=True)
    skf = StratifiedKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=SEED)
    for i, (tr_idx, va_idx) in enumerate(skf.split(Xc_train, y_train_full)):
        print(f"  -- OOF fold {i+1}/{N_OOF_FOLDS} ({time.time()-t0:.0f}s elapsed)", flush=True)
        m_cat = fit_cat(Xc_train.iloc[tr_idx], y_train_full[tr_idx])
        oof_cat[va_idx] = m_cat.predict_proba(Xc_train.iloc[va_idx])[:, 1]
        m_lgb = fit_lgb(Xg_train.iloc[tr_idx], y_train_full[tr_idx])
        oof_lgb[va_idx] = m_lgb.predict_proba(Xg_train.iloc[va_idx])[:, 1]
        m_xgb = fit_xgb(Xg_train.iloc[tr_idx], y_train_full[tr_idx])
        oof_xgb[va_idx] = m_xgb.predict_proba(Xg_train.iloc[va_idx])[:, 1]

    print(f" [{fold_name}] 메타러너 학습 ({time.time()-t0:.0f}s)", flush=True)
    meta_X = np.column_stack([oof_cat, oof_lgb, oof_xgb])
    meta_model = LogisticRegression(C=1.0, solver="lbfgs")
    meta_model.fit(meta_X, y_train_full)
    print(f"  meta coef: cat={meta_model.coef_[0][0]:.3f} lgb={meta_model.coef_[0][1]:.3f} "
          f"xgb={meta_model.coef_[0][2]:.3f} intercept={meta_model.intercept_[0]:.3f}", flush=True)

    print(f" [{fold_name}] 베이스모델 전체 재학습(iterations={FINAL_ITERATIONS}) ({time.time()-t0:.0f}s)", flush=True)
    global OOF_ITERATIONS
    _orig = OOF_ITERATIONS
    OOF_ITERATIONS = FINAL_ITERATIONS
    m_cat_full = fit_cat(Xc_train, y_train_full)
    m_lgb_full = fit_lgb(Xg_train, y_train_full)
    m_xgb_full = fit_xgb(Xg_train, y_train_full)
    OOF_ITERATIONS = _orig

    # calibration carve-out(5%, train 파티션 내부)
    calib_idx, _ = train_test_split(
        np.arange(len(train_df)), test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    cat_calib_p = m_cat_full.predict_proba(Xc_train.iloc[calib_idx])[:, 1]
    lgb_calib_p = m_lgb_full.predict_proba(Xg_train.iloc[calib_idx])[:, 1]
    xgb_calib_p = m_xgb_full.predict_proba(Xg_train.iloc[calib_idx])[:, 1]
    meta_calib_X = np.column_stack([cat_calib_p, lgb_calib_p, xgb_calib_p])
    stacked_calib_raw = meta_model.predict_proba(meta_calib_X)[:, 1]
    y_calib = y_train_full[calib_idx]
    a, b = fit_platt(stacked_calib_raw, y_calib)

    print(f" [{fold_name}] eval 예측 + calibration 적용 ({time.time()-t0:.0f}s)", flush=True)
    cat_eval_p = m_cat_full.predict_proba(Xc_eval)[:, 1]
    lgb_eval_p = m_lgb_full.predict_proba(Xg_eval)[:, 1]
    xgb_eval_p = m_xgb_full.predict_proba(Xg_eval)[:, 1]
    meta_eval_X = np.column_stack([cat_eval_p, lgb_eval_p, xgb_eval_p])
    stacked_raw = meta_model.predict_proba(meta_eval_X)[:, 1]
    stacked_calib = apply_platt(stacked_raw, a, b)

    # 참고용: CatBoost 단독(같은 iterations) 성능도 같은 calib 방식으로
    a_cat, b_cat = fit_platt(cat_calib_p, y_calib)
    cat_only_calib = apply_platt(cat_eval_p, a_cat, b_cat)

    result = {
        "fold": fold_name,
        "elapsed_sec": time.time() - t0,
        "meta_coef": {"cat": float(meta_model.coef_[0][0]), "lgb": float(meta_model.coef_[0][1]),
                      "xgb": float(meta_model.coef_[0][2]), "intercept": float(meta_model.intercept_[0])},
        "cat_only_auc": roc_auc_score(y_eval, cat_eval_p),
        "cat_only_bss_calibrated": bss_score(cat_only_calib, y_eval),
        "stacked_auc": roc_auc_score(y_eval, stacked_raw),
        "stacked_bss_raw": bss_score(stacked_raw, y_eval),
        "stacked_bss_calibrated": bss_score(stacked_calib, y_eval),
    }
    result["delta_stacked_vs_cat_only"] = result["stacked_bss_calibrated"] - result["cat_only_bss_calibrated"]
    print(
        f" [{fold_name}] cat_only calib_bss={result['cat_only_bss_calibrated']:.2f}  "
        f"stacked calib_bss={result['stacked_bss_calibrated']:.2f}  "
        f"delta={result['delta_stacked_vs_cat_only']:+.2f}  (총 {result['elapsed_sec']:.0f}s)",
        flush=True,
    )
    return result


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    df = add_risk_score(df)
    print(f" shape={df.shape}", flush=True)

    all_results = {}
    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    for fold_name, (train_df, eval_df) in fold_specs.items():
        print(f"\n=== {fold_name} ===", flush=True)
        r = run_fold(fold_name, train_df, eval_df)
        all_results[fold_name] = r
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = {fn: all_results[fn]["delta_stacked_vs_cat_only"] for fn in fold_specs}
    summary["all_axes_positive"] = all(v > 0 for v in summary.values())
    all_results["summary"] = summary
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (stacked vs CatBoost-only, calibrated BSS delta) ===", flush=True)
    print(f"  {summary}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
