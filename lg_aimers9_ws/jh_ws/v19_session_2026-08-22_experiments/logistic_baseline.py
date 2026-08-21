"""로지스틱 회귀 단독 성능 확인 -- CatBoost/LGBM/XGBoost 비교(8/21)에 이어
선형 모델까지 같은 축에 놓고 "CatBoost가 왜 유리한가"를 완결짓기 위한
빠른 sanity check. 974 레시피와 동일한 정보(CAT_COLS+raw id+trackman
context+asof_*)를 쓰되, 로지스틱 회귀에 맞게 전처리만 다르게 함:
  - CAT_COLS + raw id(pitcher_id/batter_id) -> One-Hot 인코딩(고카디널리티
    포함, CatBoost의 ordered target encoding과 달리 이게 표준적인 선형모델
    처리 방식)
  - 나머지 수치형 -> 결측은 중앙값으로 채우고(로지스틱 회귀는 NaN 처리
    불가) StandardScaler로 표준화(로지스틱 회귀는 스케일에 민감)

fold0/fold2 season split, calibration은 다른 실험들과 동일하게 정직한
carve-out(TRAIN 파티션 내부 5%)에만 Platt fit.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "logistic_baseline_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42

CAT_ONEHOT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "pitcher_id", "batter_id",
]

# 참고용 -- 같은 축(fold0/fold2)에서 8/21에 확인한 다른 모델들의 calibrated BSS
KNOWN_RESULTS = {
    "catboost_974_fold0": None,  # fold0 축은 diag_ensemble.py에서 별도 확인 안 함(fold2만 진단)
    "catboost_974_fold2": 833.08,
    "lgbm_untuned_fold2": 754.98,   # diag_ensemble.py 진단 (season<=2023/2024, 단일 폴드)
    "xgboost_untuned_fold2": 719.50,
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


def run_fold(tag, train_df, eval_df):
    y_train_full = train_df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    y_train_sub = train_sub_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]

    num_cols = [
        c for c in train_df.columns
        if c not in CAT_ONEHOT_COLS + [ID_COL, TARGET_COL]
    ]

    for c in CAT_ONEHOT_COLS:
        for d in (train_sub_df, calib_df, eval_df):
            d[c] = d[c].astype(str)

    t0 = time.time()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat_tr = ohe.fit_transform(train_sub_df[CAT_ONEHOT_COLS])
    X_cat_calib = ohe.transform(calib_df[CAT_ONEHOT_COLS])
    X_cat_eval = ohe.transform(eval_df[CAT_ONEHOT_COLS])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num_tr = scaler.fit_transform(imputer.fit_transform(train_sub_df[num_cols]))
    X_num_calib = scaler.transform(imputer.transform(calib_df[num_cols]))
    X_num_eval = scaler.transform(imputer.transform(eval_df[num_cols]))

    X_tr = sparse.hstack([X_cat_tr, sparse.csr_matrix(X_num_tr)]).tocsr()
    X_calib = sparse.hstack([X_cat_calib, sparse.csr_matrix(X_num_calib)]).tocsr()
    X_eval = sparse.hstack([X_cat_eval, sparse.csr_matrix(X_num_eval)]).tocsr()
    print(f"  [{tag}] n_features(one-hot 후)={X_tr.shape[1]}  전처리 완료 ({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    model = LogisticRegression(solver="saga", max_iter=300, n_jobs=-1, random_state=SEED)
    model.fit(X_tr, y_train_sub)
    fit_elapsed = time.time() - t1
    print(f"  [{tag}] 학습 완료 ({fit_elapsed:.1f}s)", flush=True)

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = model.predict_proba(X_eval)[:, 1]
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag,
        "n_features_onehot": X_tr.shape[1],
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "fit_elapsed_sec": fit_elapsed,
    }
    print(
        f"  [{tag}] auc={result['auc']:.4f} bss_raw={result['bss_raw']:.2f} "
        f"bss_calib={result['bss_calibrated']:.2f}",
        flush=True,
    )
    return result


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}
    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021].copy(), df[df["season"] == 2022].copy()),
        "fold2_2024": (df[df["season"] <= 2023].copy(), df[df["season"] == 2024].copy()),
    }

    for fold_name, (train_df, eval_df) in fold_specs.items():
        print(f"\n=== {fold_name} ===", flush=True)
        all_results[fold_name] = run_fold(fold_name, train_df, eval_df)
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    all_results["known_other_models"] = KNOWN_RESULTS
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===", flush=True)
    for fold_name in fold_specs:
        print(f"  {fold_name}: LogisticRegression calibrated BSS = {all_results[fold_name]['bss_calibrated']:.2f}", flush=True)
    print(f"  참고(fold2만): CatBoost=833.08  LGBM(미튜닝)=754.98  XGBoost(미튜닝)=719.50", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
