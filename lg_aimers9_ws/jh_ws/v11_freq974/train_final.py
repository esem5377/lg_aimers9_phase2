"""es_ws 974점 레시피(submit_v7_catboost_calibrated_rawid) + raw id 학습셋
빈도 피처(pitcher_train_freq/batter_train_freq) 프로덕션 학습.

배경: 8/20 로컬 3-fold walk-forward(iterations=1000)에서 이 피처가
fold0(->2022) +3.23, fold2(->2024) +5.59로 -- 8/19~8/20에 시도된 9개 이상의
후보(EWM/D앙상블/Isotonic/regime화 x2/iterations x2/segment-Platt/
recency가중/corr+importance필터링) 중 처음으로 fold0/fold2 독립 양수 일치
기준을 통과했음. 사용자 판단으로 정식 스케일 재확인 없이 바로 프로덕션
학습 -> 제출까지 진행.

es_ws train_catboost.py와 다른 점(속도/단순화를 위한 의도적 설계):
 - time-split CV(season<=2023/season==2024)로 best_iteration을 먼저 구하는
   별도 단계를 생략. 대신 5% stratified carve-out을 early stopping(최대
   iterations=2000)과 calibration 양쪽에 동시에 사용 -- 두 목적 다 "학습에
   쓰지 않은 홀드아웃"이라는 요건은 동일하게 충족하므로 방법론상 문제 없음.
 - pitcher_id/batter_id는 기존과 동일(label-encoded 수치형, OOV=-1).
 - pitcher_train_freq/batter_train_freq: 전체 학습 데이터(season<=2024)
   내 등장 횟수. test.csv(2025)의 신인/컴백 선수는 매핑에 없으므로 0으로
   채움(OOV=0, id 매핑의 -1과는 다른 sentinel -- 빈도는 자연스럽게 0이
   "본 적 없음"을 의미하므로 인위적 음수가 필요 없음).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
ES_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(OUT_DIR, "model")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
FREQ_SPECS = [("pitcher_id", "pitcher_train_freq"), ("batter_id", "batter_train_freq")]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
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
    context = joblib.load(os.path.join(ES_MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Load train data...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    id_mappings = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in RAW_ID_COLS}
    freq_maps = {raw_col: df[raw_col].value_counts().to_dict() for raw_col, _ in FREQ_SPECS}
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X_all[c] = X_all[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X_all[c] = X_all[c].astype(str)
    for raw_col, out_col in FREQ_SPECS:
        X_all[out_col] = df[raw_col].map(freq_maps[raw_col]).fillna(0).astype(int)
    y_all = df[TARGET_COL]
    print(f" n_features={X_all.shape[1]}", flush=True)

    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS]

    print("\nSplit 5% stratified carve-out (early stopping + calibration, 학습에 절대 사용 안 함)...", flush=True)
    X_tr, X_co, y_tr, y_co = train_test_split(X_all, y_all, test_size=0.05, stratify=y_all, random_state=42)
    print(f" train={X_tr.shape}  carve-out={X_co.shape}", flush=True)

    print("\nFit final model (early stopping on carve-out, max iterations=2000)...", flush=True)
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=200,
        **BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_co, y_co))
    best_iter = model.get_best_iteration()
    print(f" best_iteration={best_iter}", flush=True)

    co_pred_raw = model.predict_proba(X_co)[:, 1]
    auc = roc_auc_score(y_co, co_pred_raw)
    bss_raw = bss_score(co_pred_raw, y_co)
    a, b = fit_platt(co_pred_raw, y_co)
    co_pred_calib = apply_platt(co_pred_raw, a, b)
    bss_calib = bss_score(co_pred_calib, y_co)
    print(f" carve-out: auc={auc:.5f}  bss_raw={bss_raw:.2f}  bss_calib={bss_calib:.2f}", flush=True)

    model.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))
    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_all.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "freq_specs": FREQ_SPECS,
            "freq_maps": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in freq_maps.items()},
            "calibration": {"method": "platt_sigmoid", "a": a, "b": b},
            "carveout_metrics": {"auc": auc, "bss_raw": bss_raw, "bss_calib": bss_calib, "best_iteration": int(best_iter)},
        }, f, indent=2, ensure_ascii=False)
    joblib.dump(joblib.load(os.path.join(ES_MODEL_DIR, "trackman_context.pkl")), os.path.join(MODEL_DIR, "trackman_context.pkl"))
    print(f"\nSaved model to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
