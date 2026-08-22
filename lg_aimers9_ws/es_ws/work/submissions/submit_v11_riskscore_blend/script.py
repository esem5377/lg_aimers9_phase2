# script.py
# 992점(jh_ws v20_control_risk_score, CatBoost 6시드 배깅 + control_risk_score
# /_weighted 피처) + LGBM 3시드를 블렌드(cat 0.78 / lgb 0.22)한 버전.
# XGBoost는 fold0/fold2 walk-forward 검증에서 가중치가 둘 다 0에 가까워
# (0.05, -0.0) 제외 -- 불필요한 의존성도 줄임.
# 블렌드 가중치는 fold0(->2022)/fold2(->2024) walk-forward + calib_fit/
# calib_eval 정직 분리 검증에서 둘 다 거의 동일 크기로 양수(+3.02/+3.00)로
# 확인. 같은 calib carve-out에서 CatBoost 단독 대비 apples-to-apples 비교:
# calibrated BSS 2068.46 -> 2070.44 (+1.98).
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import Booster as LGBBooster

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


def add_risk_score(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df


def build_features(df, meta, cat_dtype):
    """cat_dtype: 'str'(CatBoost) 또는 'category'(LGBM 네이티브 categorical)."""
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

    lgb_models = [LGBBooster(model_file=os.path.join(MODEL_DIR, f"lgb_seed{s}.txt")) for s in meta["lgb_seeds"]]

    context = joblib.load(CONTEXT_PATH)
    print(f" OK. cat={len(cat_models)} lgb={len(lgb_models)}  weights={w}  calibration={calib}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    test = attach_trackman_context(test, context)
    test = add_risk_score(test)

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
        blend_raw = w["cat"] * cat_raw + w["lgb"] * lgb_raw
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
