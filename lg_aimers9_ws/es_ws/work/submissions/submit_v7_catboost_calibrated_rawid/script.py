# script.py
# CatBoost + 확률보정(Platt scaling) + raw pitcher_id/batter_id(수치형) 버전.
# submit_v6_catboost_calibrated(959점) 대비 변경점: pitcher_id/batter_id를
# jh_ws 방식(label-encoded 수치형, CatBoost cat_features 아님)으로 다시 포함.
# 로컬 검증(2024 홀드아웃 + 3-fold walk-forward): 보정 후 BSS 825.57 -> 835.05
# (+9.48), fold(->2022)에서도 +9.47로 일관됨(fold(->2023)만 -69.52로
# 무너졌지만 raw id 유무와 무관하게 그 폴드 자체가 붕괴 -- 2023 데이터의
# 알려진 이상치, es_ws/worklog/2026-08-18.md 참고).
# test.csv에만 있는 신인 선수 id는 학습 시 매핑에 없으므로 -1(OOV)로 인코딩.
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
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
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
    """trackman_history 기반 상황 단위 통계 피처를 좌측 조인으로 붙인다 (학습 때와 동일)."""
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, meta):
    """모델 입력 추출 — row_id만 빼고 학습 때와 동일한 컬럼 순서로 정렬.

    - pitcher_id/batter_id: 학습 때 만든 매핑으로 label-encoding. 매핑에
      없는 값(신인 선수 등)은 -1(OOV)로 처리.
    - CatBoost 범주형 컬럼은 문자열로 받는다.
    """
    X = df.drop(columns=[ID_COL])
    for c in meta["raw_id_cols"]:
        mapping = meta["id_mappings"][c]
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(str)
    return X[meta["columns"]]


def apply_calibration(raw_p, calib):
    """학습 때 fit한 Platt scaling(a, b)을 그대로 적용."""
    if calib is None:
        return raw_p
    a, b = calib["a"], calib["b"]
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합."""
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
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # catboost.cbm, feature_meta.json 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "catboost.cbm")
    META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
    CONTEXT_PATH = os.path.join(MODEL_DIR, "trackman_context.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    calib = meta.get("calibration")
    context = joblib.load(CONTEXT_PATH)
    print(f" OK. calibration={calib}  raw_id_cols={meta.get('raw_id_cols')}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- trackman context 피처 부착 (학습과 동일) ----
    test = attach_trackman_context(test, context)

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, meta)
    n_oov_pitcher = int((X["pitcher_id"] == -1).sum())
    n_oov_batter = int((X["batter_id"] == -1).sum())
    print(f" features={X.shape[1]}  OOV pitcher_id={n_oov_pitcher}  OOV batter_id={n_oov_batter}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    raw_preds = model.predict_proba(X)[:, 1] if len(X) else []
    preds = apply_calibration(raw_preds, calib) if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
