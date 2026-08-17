"""상황(카운트+투타매치업) 기반 구종 확률 피처 실험 (CatBoost 단독, 빠른 비교용).

trackman_history.csv를 pitcher_id로 조인하는 게 아니라, train/trackman이 공유하는
상황 컬럼(balls_before/strikes_before/outs_before/pitcher_hand/batter_hand)
기준으로 population-level 구종 비율을 집계해 lookup 피처로 붙인다.
개별 투구를 특정 선수로 재식별하지 않는 안전한 방식.

주의: train의 pitcher_hand/batter_hand는 '1'/'2'로 익명화돼 있고 trackman은
'Left'/'Right' 문자열이라 공식 매핑이 없다. 두 데이터의 비율(train '2'=74.1%
투수우세, trackman 'Right'=74.9%; batter도 유사)을 비교해 1=Left, 2=Right로
역추정해서 매핑한다 (개별 선수 식별이 아니라 카테고리 라벨 복원이라 안전).
"""
import os
import joblib
import random
import warnings
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
import lightgbm as lgb
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

DATA_DIR = './data'
HAND_MAP = {'Left': '1', 'Right': '2'}  # trackman -> train 코드 (비율 기반 역추정)

print("[1/5] 데이터 로딩...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tm_df = pd.read_csv(os.path.join(DATA_DIR, 'trackman_history.csv'),
                     usecols=['balls_before', 'strikes_before', 'outs_before',
                              'pitcher_hand', 'batter_hand', 'pitch_type_group'])

print("[2/5] 상황 기반 구종 비율 집계...")
tm_df['pitcher_hand'] = tm_df['pitcher_hand'].map(HAND_MAP)
tm_df['batter_hand'] = tm_df['batter_hand'].map(HAND_MAP)
tm_df['is_fastball'] = (tm_df['pitch_type_group'] == 'fastball').astype(int)
tm_df['is_breaking'] = (tm_df['pitch_type_group'] == 'breaking').astype(int)
tm_df['is_offspeed'] = (tm_df['pitch_type_group'] == 'offspeed').astype(int)

SIT_COLS = ['balls_before', 'strikes_before', 'outs_before', 'pitcher_hand', 'batter_hand']
sit_stats = tm_df.groupby(SIT_COLS).agg(
    sit_fastball_rate=('is_fastball', 'mean'),
    sit_breaking_rate=('is_breaking', 'mean'),
    sit_offspeed_rate=('is_offspeed', 'mean'),
    sit_n=('is_fastball', 'size'),
).reset_index()
print(f"  └ 상황 조합 수={len(sit_stats)}, 평균 표본={sit_stats['sit_n'].mean():.0f}")

global_fb = tm_df['is_fastball'].mean()
global_bk = tm_df['is_breaking'].mean()
global_os = tm_df['is_offspeed'].mean()

# train의 pitcher_hand/batter_hand는 이미 '1'/'2' 코드 (astype str로 통일)
train_df['pitcher_hand'] = train_df['pitcher_hand'].astype(str)
train_df['batter_hand'] = train_df['batter_hand'].astype(str)

train_df = train_df.merge(sit_stats, on=SIT_COLS, how='left')
n_matched = train_df['sit_n'].notna().sum()
print(f"  └ train 매칭률: {n_matched}/{len(train_df)} ({n_matched/len(train_df)*100:.1f}%)")
train_df['sit_fastball_rate'] = train_df['sit_fastball_rate'].fillna(global_fb)
train_df['sit_breaking_rate'] = train_df['sit_breaking_rate'].fillna(global_bk)
train_df['sit_offspeed_rate'] = train_df['sit_offspeed_rate'].fillna(global_os)
train_df['sit_n'] = train_df['sit_n'].fillna(0)

print("[3/5] 나머지 파생변수 생성...")
if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
    train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
    train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
    train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)
if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
    train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

ignore_cols = ['row_id', 'control_success']
features = [c for c in train_df.columns if c not in ignore_cols]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]

for c in cat_cols:
    train_df[c] = train_df[c].astype(str).fillna('missing')
    unique_vals = sorted(train_df[c].unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    train_df[c] = train_df[c].map(mapping)

X = train_df[features].copy()
y = train_df['control_success'].copy()

print("[4/5] 다중공선성 제거 및 중요 변수 선별...")
num_cols = [c for c in features if c not in cat_cols]
corr_matrix = X[num_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
if to_drop:
    X = X.drop(columns=to_drop)
    features = [f for f in features if f not in to_drop]
    print(f"  └ 다중공선성으로 제거된 컬럼: {to_drop}")

selector_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
selector_model.fit(X, y)
importance_df = pd.DataFrame({'feature': features, 'importance': selector_model.feature_importances_}).sort_values('importance', ascending=False)
selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
print(f"  └ sit_* 피처 importance 순위: {importance_df[importance_df['feature'].str.startswith('sit_')]}")
X = X[selected_features].copy()

train_idx = np.where((train_df['season'] <= 2023).values)[0]
val_idx = np.where((train_df['season'] == 2024).values)[0]
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

print(f"[5/5] CatBoost 학습 (train n={len(X_train)}, val n={len(X_val)})...")
base_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
base_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
calib_cat = get_calibrated_model(base_cat)
calib_cat.fit(X_val, y_val)
preds = calib_cat.predict_proba(X_val)[:, 1]
bss = bss_score(preds, y_val.values)
print(f"  └ 2024 BSS={bss:.1f} (베이스라인 737.5와 비교)")
