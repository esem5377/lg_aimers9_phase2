# script.py
import os

import joblib
import pandas as pd

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
    """trackman_history 기반 상황 단위 통계 피처를 좌측 조인으로 붙인다 (학습 때와 동일).

    카운트/아웃, 투타 좌우, 이닝/초말 같은 공통 상황 키로 집계한 값이라
    선수 단위 정보나 개별 행의 결과 정보를 담지 않는다. model/trackman_context.pkl
    에 미리 계산되어 있어 평가 환경에서 trackman_history.csv 자체는 필요 없다.
    """
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, meta):
    """모델 입력 추출 — row_id만 빼고 학습 때와 동일한 컬럼 순서로 정렬.

    범주형 컬럼(top_bottom, game_type, base_state, pitcher_hand, batter_hand,
    pitcher_id, batter_id, pitcher_team_id, batter_team_id)은 학습 시 저장해둔
    카테고리 목록(feature_meta.pkl)을 그대로 적용한다. 학습 데이터에 없던 값은
    NaN으로 처리되며, LightGBM은 결측치를 자체적으로 다룬다.
    """
    X = df.drop(columns=[ID_COL])
    for c in meta["cat_cols"]:
        X[c] = X[c].astype("category").cat.set_categories(meta["cat_categories"][c])
    return X[meta["columns"]]


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
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
    MODEL_DIR = "./model"          # lgbm.pkl, feature_meta.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "lgbm.pkl")
    META_PATH = os.path.join(MODEL_DIR, "feature_meta.pkl")
    CONTEXT_PATH = os.path.join(MODEL_DIR, "trackman_context.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    context = joblib.load(CONTEXT_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

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
