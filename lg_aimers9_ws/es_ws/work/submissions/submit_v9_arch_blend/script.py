# script.py
# 982점(jh_ws v18_seed_bagging, CatBoost 6시드 배깅) + LGBM 3시드 + XGBoost
# 3시드를 소량 블렌드(cat 0.85 / lgb 0.10 / xgb 0.05)한 버전.
# 블렌드 가중치는 fold0(->2022)/fold2(->2024) walk-forward + calib_fit/calib_eval
# 정직 분리 검증에서 둘 다 양수(+4.57/+1.37)로 확인된 fold2 가중치를 그대로 사용
# (calib carve-out에서 직접 grid search하면 무작위분할 특유의 낙관편향으로
# xgb 쪽에 가중치가 쏠리는 걸 확인해 기각 -- es_ws/work/pipeline/train_arch_blend_bagged.py
# 참고). 같은 calib carve-out에서 CatBoost 단독 대비 apples-to-apples 비교:
# calibrated BSS 2069.21 -> 2080.01 (+10.80).
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import Booster as LGBBooster
from xgboost import XGBClassifier

ID_COL = "row_id"
TARGET_COL = "control_success"


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


def attach_trackman_context(df, context):
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, meta, cat_dtype):
    """cat_dtype: 'str'(CatBoost) 또는 'category'(LGBM/XGBoost 네이티브 categorical)."""
    X = df.drop(columns=[ID_COL])
    for c in meta["raw_id_cols"]:
        mapping = meta["id_mappings"][c]
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(str) if cat_dtype == "str" else X[c].astype(str).astype("category")
    cols = meta["columns_str"] if cat_dtype == "str" else meta["columns_cat"]
    return X[cols]


def apply_calibration(raw_p, calib):
    if calib is None:
        return raw_p
    a, b = calib["a"], calib["b"]
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
    CONTEXT_PATH = os.path.join(MODEL_DIR, "trackman_context.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load models + meta...")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    calib = meta["calibration"]
    w = meta["blend_weights"]

    cat_models = []
    for seed in meta["cat_seeds"]:
        m = CatBoostClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cat_models.append(m)

    lgb_models = []
    for seed in meta["lgb_seeds"]:
        lgb_models.append(LGBBooster(model_file=os.path.join(MODEL_DIR, f"lgb_seed{seed}.txt")))

    xgb_models = []
    for seed in meta["xgb_seeds"]:
        m = XGBClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"xgb_seed{seed}.json"))
        xgb_models.append(m)

    context = joblib.load(CONTEXT_PATH)
    print(f" OK. cat={len(cat_models)} lgb={len(lgb_models)} xgb={len(xgb_models)}  "
          f"weights={w}  calibration={calib}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    test = attach_trackman_context(test, context)

    print("Build features...")
    ids = test[ID_COL].tolist()
    X_str = build_features(test, meta, "str")
    X_cat = build_features(test, meta, "category")
    n_oov_pitcher = int((X_str["pitcher_id"] == -1).sum())
    n_oov_batter = int((X_str["batter_id"] == -1).sum())
    print(f" features={X_str.shape[1]}  OOV pitcher_id={n_oov_pitcher}  OOV batter_id={n_oov_batter}")

    print("Inference (아키텍처별 배깅 -> 고정 가중치 블렌드 -> Platt 보정)...")
    if len(X_str):
        cat_raw = np.mean([m.predict_proba(X_str)[:, 1] for m in cat_models], axis=0)
        lgb_raw = np.mean([m.predict(X_cat) for m in lgb_models], axis=0)
        xgb_raw = np.mean([m.predict_proba(X_cat)[:, 1] for m in xgb_models], axis=0)
        blend_raw = w["cat"] * cat_raw + w["lgb"] * lgb_raw + w["xgb"] * xgb_raw
        preds = apply_calibration(blend_raw, calib)
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
