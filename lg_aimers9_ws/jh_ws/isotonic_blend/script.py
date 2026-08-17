import os

# NOTE: lightgbm/catboost는 반드시 pandas보다 먼저 import해야 한다 (train.py 참고).
# joblib.load가 내부적으로 이 모듈들을 import하므로 여기서 먼저 import해 순서를 고정한다.
import lightgbm as lgb  # noqa: F401
from catboost import CatBoostClassifier  # noqa: F401

import joblib
import numpy as np
import pandas as pd


def load_artifacts(model_dir='model'):
    lgb_models = joblib.load(os.path.join(model_dir, 'lgb_models.pkl'))
    cat_models = joblib.load(os.path.join(model_dir, 'cat_models.pkl'))
    calibrator = joblib.load(os.path.join(model_dir, 'calibrator.pkl'))
    meta = joblib.load(os.path.join(model_dir, 'meta.pkl'))
    return lgb_models, cat_models, calibrator, meta


def load_test_data():
    # 평가 서버는 data/test.csv 를 읽기 전용으로 자동 제공한다.
    path = os.path.join('data', 'test.csv') if os.path.exists(os.path.join('data', 'test.csv')) else 'test.csv'
    return pd.read_csv(path)


def encode_features(test_df, meta):
    feature_cols = meta['feature_cols']
    cat_cols = meta['cat_cols']
    cat_mappings = meta['cat_mappings']

    X = test_df.copy()

    # 학습 시 없던 컬럼은 NaN으로 채워 동일한 피처 셋을 보장 (row-wise 독립 처리)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    for c in cat_cols:
        mapping = cat_mappings[c]
        unk_code = len(mapping)
        vals = X[c].astype(str)
        codes = vals.map(mapping).fillna(unk_code).astype(int)
        X[c] = pd.Categorical(codes, categories=list(range(unk_code + 1)))

    return X[feature_cols]


def predict(lgb_models, cat_models, calibrator, meta, X):
    w_lgb = meta['best_w_lgb']
    w_cat = meta['best_w_cat']
    clip_eps = meta['clip_eps']

    preds_lgb = np.mean([m.predict_proba(X)[:, 1] for m in lgb_models], axis=0)
    preds_cat = np.mean([m.predict_proba(X)[:, 1] for m in cat_models], axis=0)

    blend = w_lgb * preds_lgb + w_cat * preds_cat
    calibrated = calibrator.predict(blend)
    return np.clip(calibrated, clip_eps, 1.0 - clip_eps)


def save_submission(test_df, preds, out_path=os.path.join('output', 'submission.csv')):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    id_col = 'row_id' if 'row_id' in test_df.columns else test_df.columns[0]
    submission = pd.DataFrame({id_col: test_df[id_col], 'control_success': preds})
    submission.to_csv(out_path, index=False)
    return submission


def main():
    lgb_models, cat_models, calibrator, meta = load_artifacts()
    test_df = load_test_data()

    X = encode_features(test_df, meta)
    preds = predict(lgb_models, cat_models, calibrator, meta, X)
    submission = save_submission(test_df, preds)

    print("--- 제출 파일 검증 ---")
    print(f"행 개수: {len(submission)} (test 원본: {len(test_df)})")
    print(f"결측치 개수: {submission['control_success'].isnull().sum()}")
    print(f"최소/최대 예측 확률: {submission['control_success'].min():.5f} / {submission['control_success'].max():.5f}")
    print(">> output/submission.csv 생성 완료!")


if __name__ == '__main__':
    main()
