# script.py
# 974점(es_ws v7) 레시피 + 범주형 희귀 레벨 묶기(pitcher_team_id/batter_team_id
# 중 표본 5000건 미만 팀을 OTHER_TEAM 하나로 묶음) 추가 버전.
# test.csv에만 있는 새 team_id도 keep-set에 없으면 자동으로 OTHER_TEAM 처리됨
# (학습 때와 동일한 규칙, feature_meta.json의 team_keep_sets 기준).
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 데이터 로드 유틸
# =======================

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


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def attach_trackman_context(df, context):
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def apply_team_bucketing(series, keep_list, bucket_label):
    keep_set = set(keep_list)
    s = series.astype(str)
    return s.where(s.isin(keep_set), bucket_label)


def build_features(df, meta):
    """모델 입력 추출 -- row_id만 빼고 학습 때와 동일한 컬럼 순서로 정렬.

    - pitcher_id/batter_id: 학습 때 만든 매핑으로 label-encoding, 매핑에
      없는 값(신인 선수 등)은 -1(OOV).
    - CatBoost 범주형 컬럼은 문자열로 받는다.
    - pitcher_team_id/batter_team_id: 학습 때 keep-set에 없는 값(희귀팀 또는
      test에만 있는 새 team_id)은 OTHER_TEAM으로 묶는다.
    """
    X = df.drop(columns=[ID_COL])
    for c in meta["raw_id_cols"]:
        mapping = meta["id_mappings"][c]
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(str)
    for c in meta["rare_team_cols"]:
        X[c] = apply_team_bucketing(
            X[c], meta["team_keep_sets"][c], meta["rare_team_bucket_label"])
    return X[meta["columns"]]


def apply_calibration(raw_p, calib):
    if calib is None:
        return raw_p
    a, b = calib["a"], calib["b"]
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


# =======================
# 제출 파일 생성 유틸
# =======================

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


# =======================
# main
# =======================

def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "catboost.cbm")
    META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
    CONTEXT_PATH = os.path.join(MODEL_DIR, "trackman_context.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    calib = meta.get("calibration")
    context = joblib.load(CONTEXT_PATH)
    print(f" OK. calibration={calib}  raw_id_cols={meta.get('raw_id_cols')}  "
          f"rare_team_cols={meta.get('rare_team_cols')}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    test = attach_trackman_context(test, context)

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, meta)
    n_oov_pitcher = int((X["pitcher_id"] == -1).sum())
    n_oov_batter = int((X["batter_id"] == -1).sum())
    n_other_pteam = int((X["pitcher_team_id"] == meta["rare_team_bucket_label"]).sum())
    n_other_bteam = int((X["batter_team_id"] == meta["rare_team_bucket_label"]).sum())
    print(f" features={X.shape[1]}  OOV pitcher_id={n_oov_pitcher}  OOV batter_id={n_oov_batter}  "
          f"OTHER pitcher_team_id={n_other_pteam}  OTHER batter_team_id={n_other_bteam}")

    print("Inference model...")
    raw_preds = model.predict_proba(X)[:, 1] if len(X) else []
    preds = apply_calibration(raw_preds, calib) if len(X) else []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
