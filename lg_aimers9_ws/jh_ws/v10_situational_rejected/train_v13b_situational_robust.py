"""sit_* 피처(상황 기반 구종 확률)의 효과를 여러 seed에 걸친 paired 비교로 검증.

지난번 recency weighting에서 단일 seed 결과(+2.9)가 스윕해보니 노이즈였던 걸
확인했기 때문에, 이번엔 각 seed마다 "sit_* 피처 있음 vs 없음"을 나란히 학습해서
paired delta를 여러 번 확인한다. 델타가 seed에 관계없이 계속 양수면 진짜 신호,
부호가 왔다갔다하면 노이즈로 판단.
"""
import os
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

def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))

DATA_DIR = './data'
HAND_MAP = {'Left': '1', 'Right': '2'}
SEEDS = [42, 123, 7]

print("[1/4] 데이터 로딩...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tm_df = pd.read_csv(os.path.join(DATA_DIR, 'trackman_history.csv'),
                     usecols=['balls_before', 'strikes_before', 'outs_before',
                              'pitcher_hand', 'batter_hand', 'pitch_type_group'])

print("[2/4] 상황 기반 구종 비율 집계...")
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
global_fb = tm_df['is_fastball'].mean()
global_bk = tm_df['is_breaking'].mean()
global_os = tm_df['is_offspeed'].mean()

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

SIT_FEATURE_COLS = ['sit_fastball_rate', 'sit_breaking_rate', 'sit_offspeed_rate', 'sit_n']

def prepare_xy(df, drop_sit):
    ignore_cols = ['row_id', 'control_success']
    feats = [c for c in df.columns if c not in ignore_cols]
    if drop_sit:
        feats = [c for c in feats if c not in SIT_FEATURE_COLS]
    d = df[feats].copy()
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(d[c])]
    for c in cat_cols:
        d[c] = d[c].astype(str).fillna('missing')
        mapping = {v: i for i, v in enumerate(sorted(d[c].unique()))}
        d[c] = d[c].map(mapping)
    return d, df['control_success'].copy()

train_idx = np.where((train_df['season'] <= 2023).values)[0]
val_idx = np.where((train_df['season'] == 2024).values)[0]

print("[3/4] paired 비교 시작 (fastselect 없이 원본 피처셋으로 baseline/처치 둘 다 학습)...")
results = []
for drop_sit, label in [(True, 'baseline(sit_*없음)'), (False, 'sit_* 포함')]:
    X, y = prepare_xy(train_df, drop_sit)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    for seed in SEEDS:
        random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
        base_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=seed, verbose=0)
        base_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        calib_cat = get_calibrated_model(base_cat)
        calib_cat.fit(X_val, y_val)
        preds = calib_cat.predict_proba(X_val)[:, 1]
        bss = bss_score(preds, y_val.values)
        results.append((label, seed, bss))
        print(f"  └ {label} | seed={seed} | 2024 BSS={bss:.1f}")

print("[4/4] 요약 (paired delta)")
by_seed = {}
for label, seed, bss in results:
    by_seed.setdefault(seed, {})[label] = bss
for seed, d in by_seed.items():
    delta = d.get('sit_* 포함', float('nan')) - d.get('baseline(sit_*없음)', float('nan'))
    print(f"  seed={seed}: baseline={d.get('baseline(sit_*없음)'):.1f} -> sit_*포함={d.get('sit_* 포함'):.1f} (delta={delta:+.1f})")
