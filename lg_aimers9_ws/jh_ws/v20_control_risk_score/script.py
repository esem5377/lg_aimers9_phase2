# script.py
# 982점(v18 6시드 배깅) 레시피 + control_risk_score/control_risk_score_weighted
# 피처 추가. 나머지(trackman context 병합, raw id -1 OOV 인코딩, cat_cols
# 문자열화, 시드 배깅 평균 -> Platt calibration)는 v18과 동일.
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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


def build_features(df, meta):
    X = df.drop(columns=[ID_COL])
    for c in meta["raw_id_cols"]:
        mapping = meta["id_mappings"][c]
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(str)
    return X[meta["columns"]]


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
    calib = meta.get("calibration")
    seeds = meta["seeds"]
    models = []
    for seed in seeds:
        m = CatBoostClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))
        models.append(m)
    context = joblib.load(CONTEXT_PATH)
    print(f" OK. n_models={len(models)}  seeds={seeds}  calibration={calib}  "
          f"raw_id_cols={meta.get('raw_id_cols')}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    test = attach_trackman_context(test, context)
    test = add_risk_score(test)

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, meta)
    n_oov_pitcher = int((X["pitcher_id"] == -1).sum())
    n_oov_batter = int((X["batter_id"] == -1).sum())
    print(f" features={X.shape[1]}  OOV pitcher_id={n_oov_pitcher}  OOV batter_id={n_oov_batter}")

    print("Inference (시드 배깅: 모델별 raw 확률 -> 평균 -> 보정)...")
    if len(X):
        raw_preds_per_seed = [m.predict_proba(X)[:, 1] for m in models]
        raw_preds_bagged = np.mean(raw_preds_per_seed, axis=0)
        preds = apply_calibration(raw_preds_bagged, calib)
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
