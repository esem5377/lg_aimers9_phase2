"""학습된 LightGBM 모델로 test.csv에 대한 제출 파일 생성."""
import os

import joblib
import pandas as pd

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/output"

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
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def attach_trackman_context(df, context):
    """trackman_history 기반 상황 단위 통계 피처를 좌측 조인으로 붙인다 (학습 때와 동일)."""
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, meta):
    """학습 때와 동일한 컬럼 순서 / 범주형 코드로 변환.

    test에만 존재하고 train에는 없던 범주값은 NaN으로 처리되어
    LightGBM이 결측치로 자연스럽게 다룬다.
    """
    X = df.drop(columns=[ID_COL])
    for c in meta["cat_cols"]:
        X[c] = X[c].astype("category").cat.set_categories(meta["cat_categories"][c])
    return X[meta["columns"]]


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model...")
    model = joblib.load(os.path.join(MODEL_DIR, "lgbm.pkl"))
    meta = joblib.load(os.path.join(MODEL_DIR, "feature_meta.pkl"))

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Attach trackman context features...")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    test = attach_trackman_context(test, context)

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, meta)

    print("Inference...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(" preds:", list(zip(ids, preds)))

    sub = merge_predictions(sub, ids, preds)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
