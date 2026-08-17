"""v14 최종 제출용 재학습: 시간 기반 검증(2019~2023/2024)으로 확정한 구조
(situational 구종 확률 피처 + 3모델 앙상블, 2024 홀드아웃 BSS=751.8, 베이스라인 738.0
대비 +13.8) + 블렌드 가중치(LGBM 0.234 / CatBoost 0.462 / XGBoost 0.304)를 그대로
고정한 채, 2019~2024 전체 데이터로 재학습한다.
"""
import os
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

FROZEN_WEIGHTS = {'w_lgb': 0.234, 'w_cat': 0.462, 'w_xgb': 0.304}

DATA_DIR = './data'
MODEL_DIR = './model_v14'
os.makedirs(MODEL_DIR, exist_ok=True)

HAND_MAP = {'Left': '1', 'Right': '2'}
SIT_COLS = ['balls_before', 'strikes_before', 'outs_before', 'pitcher_hand', 'batter_hand']

print("[1/6] 데이터 로딩...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tm_df = pd.read_csv(os.path.join(DATA_DIR, 'trackman_history.csv'), usecols=SIT_COLS + ['pitch_type_group'])

print("[2/6] 상황 기반 구종 확률 피처...")
tm_df['pitcher_hand'] = tm_df['pitcher_hand'].map(HAND_MAP)
tm_df['batter_hand'] = tm_df['batter_hand'].map(HAND_MAP)
tm_df['is_fastball'] = (tm_df['pitch_type_group'] == 'fastball').astype(int)
tm_df['is_breaking'] = (tm_df['pitch_type_group'] == 'breaking').astype(int)
tm_df['is_offspeed'] = (tm_df['pitch_type_group'] == 'offspeed').astype(int)

sit_stats = tm_df.groupby(SIT_COLS).agg(
    sit_fastball_rate=('is_fastball', 'mean'),
    sit_breaking_rate=('is_breaking', 'mean'),
    sit_offspeed_rate=('is_offspeed', 'mean'),
    sit_n=('is_fastball', 'size'),
).reset_index()
global_fb = float(tm_df['is_fastball'].mean())
global_bk = float(tm_df['is_breaking'].mean())
global_os = float(tm_df['is_offspeed'].mean())

train_df['pitcher_hand'] = train_df['pitcher_hand'].astype(str)
train_df['batter_hand'] = train_df['batter_hand'].astype(str)
train_df = train_df.merge(sit_stats, on=SIT_COLS, how='left')
train_df['sit_fastball_rate'] = train_df['sit_fastball_rate'].fillna(global_fb)
train_df['sit_breaking_rate'] = train_df['sit_breaking_rate'].fillna(global_bk)
train_df['sit_offspeed_rate'] = train_df['sit_offspeed_rate'].fillna(global_os)
train_df['sit_n'] = train_df['sit_n'].fillna(0)

if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
    train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
    train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
    train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)
if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
    train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

ignore_cols = ['row_id', 'control_success']
train_p_col = [c for c in train_df.columns if 'pitcher' in c.lower() and c != 'pitcher_hand'][0]
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

print("[3/6] 다중공선성 제거 및 중요 변수 선별...")
num_cols = [c for c in features if c not in cat_cols and c != train_p_col]
corr_matrix = X[num_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
if to_drop:
    X = X.drop(columns=to_drop)
    features = [f for f in features if f not in to_drop]

selector_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
selector_model.fit(X, y)
importance_df = pd.DataFrame({'feature': features, 'importance': selector_model.feature_importances_}).sort_values('importance', ascending=False)
selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
X = X[selected_features].copy()
print(f"  └ 선택된 피처 수={len(selected_features)}")

print("[4/6] 전체 데이터(2019~2024) 최종 모델 학습 (early stopping/보정용 5% carve-out)...")
X_train, X_es, y_train, y_es = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42)
print(f"  └ train n={len(X_train)}, carve-out n={len(X_es)}")

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
print(f"  └ [고정 가중치 블렌드] carve-out BSS={bss_score(blend, y_es.values):.1f} (참고용, 2024 홀드아웃 BSS=751.8과는 다른 표본)")

print("[5/6] 아티팩트 저장...")
meta_info = {
    'features': selected_features,
    'cat_cols': [c for c in cat_cols if c in selected_features],
    'cat_mappings': cat_mappings,
    'train_pitcher_col': train_p_col,
    'opt_weights': FROZEN_WEIGHTS,
    'sit_stats': sit_stats.to_dict(orient='records'),
    'sit_cols': SIT_COLS,
    'hand_map': HAND_MAP,
    'sit_global_fallback': {'sit_fastball_rate': global_fb, 'sit_breaking_rate': global_bk, 'sit_offspeed_rate': global_os},
}
joblib.dump(meta_info, os.path.join(MODEL_DIR, 'meta_info.pkl'))
print("[6/6] 완료! model_v14/에 최종 모델 저장됨")
