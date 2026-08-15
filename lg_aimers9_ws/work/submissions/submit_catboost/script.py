# script.py
# CatBoost 버전 — 리더보드 점수 확인용 실험 패키지.
# 로컬 검증 AUC 0.55005 (LightGBM 버전 0.54887 대비 +0.0012).
# 평가 서버가 requirements.txt로 catboost까지 설치해줄지 확인이 안 된
# 상태라 리스크를 감수하고 테스트 제출하는 용도.
import json
import os

import joblib
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

    CatBoost는 범주형 컬럼을 문자열로 받는다 (LightGBM의 category dtype과 다름).
    """
    X = df.drop(columns=[ID_COL])
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(str)
    return X[meta["columns"]]


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
    context = joblib.load(CONTEXT_PATH)
    print(" OK.")

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
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
