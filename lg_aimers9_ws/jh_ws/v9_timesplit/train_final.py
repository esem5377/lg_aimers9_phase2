"""최종 제출용 재학습 스크립트.

train.py(시간 기반 검증: 2019~2023 학습 / 2024 검증)로 확인한 구조와
블렌드 가중치(LGBM 0.086 / CatBoost 0.914 / XGBoost 0.000, 2024 홀드아웃 BSS=738.0)를
그대로 고정한 채, 실제 제출용 모델은 2019~2024 전체 데이터로 재학습한다.
early stopping/보정(calibration)용으로만 소량(5%)을 따로 떼어내고, 블렌드 가중치는
재탐색하지 않고 train.py에서 이미 검증된 값을 그대로 사용한다.
"""
import os
import glob
import joblib
import random
import warnings
import numpy as np
import pandas as pd

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
    def get_calibrated_model(base_model):
        return CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method='sigmoid')
except ImportError:
    def get_calibrated_model(base_model):
        return CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))

# train.py에서 시간 기반 검증으로 확정한 블렌드 가중치 (재탐색하지 않고 고정)
FROZEN_WEIGHTS = {'w_lgb': 0.086, 'w_cat': 0.914, 'w_xgb': 0.000}

DATA_DIR = './data'
MODEL_DIR = './model'
os.makedirs(MODEL_DIR, exist_ok=True)

print("[1/5] 데이터 로딩 및 사전 집계...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tm_path = os.path.join(DATA_DIR, 'trackman_history.csv')

trackman_records = []
hist_p_col = 'pitcher_id'
if os.path.exists(tm_path):
    tm_df = pd.read_csv(tm_path)
    hist_p_col = [c for c in tm_df.columns if 'pitcher' in c.lower()][0]

    rel_x = [c for c in tm_df.columns if 'rel' in c.lower() and ('x' in c.lower() or 'side' in c.lower())][0]
    rel_z = [c for c in tm_df.columns if 'rel' in c.lower() and ('z' in c.lower() or 'height' in c.lower())][0]

    agg_df = tm_df.groupby(hist_p_col).agg(
        avg_rel_x=(rel_x, 'mean'),
        avg_rel_z=(rel_z, 'mean'),
        std_rel_x=(rel_x, 'std'),
        std_rel_z=(rel_z, 'std')
    ).reset_index()

    trackman_records = agg_df.to_dict(orient='records')
    train_p_col = [c for c in train_df.columns if 'pitcher' in c.lower()][0]
    train_df = train_df.merge(agg_df, left_on=train_p_col, right_on=hist_p_col, how='left')

print("[2/5] 파생변수 생성...")
if 'avg_rel_x' in train_df.columns and 'rel_x' in train_df.columns:
    train_df['release_dev'] = np.sqrt(
        (train_df['rel_x'] - train_df['avg_rel_x'])**2 +
        (train_df['rel_z'] - train_df['avg_rel_z'])**2
    )

if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
    train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
    train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
    train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)

if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
    train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

ignore_cols = ['row_id', 'control_success']
train_p_col = [c for c in train_df.columns if 'pitcher' in c.lower()][0]
features = [c for c in train_df.columns if c not in ignore_cols]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]

cat_mappings = {}
for c in cat_cols:
    train_df[c] = train_df[c].astype(str).fillna('missing')
    unique_vals = sorted(train_df[c].unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    cat_mappings[c] = mapping
    train_df[c] = train_df[c].map(mapping)

X = train_df[features].copy()
y = train_df['control_success'].copy()

print("[3/5] 다중공선성 제거 및 중요 변수 선별...")
num_cols = [c for c in features if c not in cat_cols and c != train_p_col]
corr_matrix = X[num_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

if to_drop:
    X = X.drop(columns=to_drop)
    features = [f for f in features if f not in to_drop]

selector_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
selector_model.fit(X, y)

importance_df = pd.DataFrame({
    'feature': features,
    'importance': selector_model.feature_importances_
}).sort_values('importance', ascending=False)

selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
X = X[selected_features].copy()

print("[4/5] 전체 데이터(2019~2024) 최종 모델 학습 (early stopping/보정용 5% carve-out)...")
X_train, X_es, y_train, y_es = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=42
)
print(f"  └ train n={len(X_train)}, early-stop/calibration carve-out n={len(X_es)}")

fold = 0
base_lgb = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.02, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1
)
base_lgb.fit(X_train, y_train, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(50, verbose=False)])
calib_lgb = get_calibrated_model(base_lgb)
calib_lgb.fit(X_es, y_es)
joblib.dump(calib_lgb, os.path.join(MODEL_DIR, f'lgb_fold{fold}.pkl'))
print(f"  └ LightGBM 완료 | carve-out BSS={bss_score(calib_lgb.predict_proba(X_es)[:, 1], y_es.values):.1f}")

base_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42+fold, verbose=0)
base_cat.fit(X_train, y_train, eval_set=(X_es, y_es), early_stopping_rounds=50)
calib_cat = get_calibrated_model(base_cat)
calib_cat.fit(X_es, y_es)
joblib.dump(calib_cat, os.path.join(MODEL_DIR, f'cat_fold{fold}.pkl'))
print(f"  └ CatBoost 완료 | carve-out BSS={bss_score(calib_cat.predict_proba(X_es)[:, 1], y_es.values):.1f}")

base_xgb = XGBClassifier(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42+fold, eval_metric='logloss', early_stopping_rounds=50
)
base_xgb.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=False)
calib_xgb = get_calibrated_model(base_xgb)
calib_xgb.fit(X_es, y_es)
joblib.dump(calib_xgb, os.path.join(MODEL_DIR, f'xgb_fold{fold}.pkl'))
print(f"  └ XGBoost 완료 | carve-out BSS={bss_score(calib_xgb.predict_proba(X_es)[:, 1], y_es.values):.1f}")

blend = (calib_lgb.predict_proba(X_es)[:, 1] * FROZEN_WEIGHTS['w_lgb']
         + calib_cat.predict_proba(X_es)[:, 1] * FROZEN_WEIGHTS['w_cat']
         + calib_xgb.predict_proba(X_es)[:, 1] * FROZEN_WEIGHTS['w_xgb'])
blend = np.clip(blend, 0.005, 0.995)
print(f"  └ [고정 가중치 블렌드] carve-out BSS={bss_score(blend, y_es.values):.1f} (참고용, train.py의 2024 홀드아웃 BSS=738.0과는 다른 표본)")

print("[5/5] 아티팩트 저장...")
meta_info = {
    'features': selected_features,
    'cat_cols': [c for c in cat_cols if c in selected_features],
    'cat_mappings': cat_mappings,
    'trackman_records': trackman_records,
    'hist_pitcher_col': hist_p_col,
    'train_pitcher_col': train_p_col,
    'opt_weights': FROZEN_WEIGHTS
}
joblib.dump(meta_info, os.path.join(MODEL_DIR, 'meta_info.pkl'))
print("완료! 전체 데이터 기반 최종 모델 저장됨 (model/lgb_fold0.pkl, cat_fold0.pkl, xgb_fold0.pkl, meta_info.pkl)")
