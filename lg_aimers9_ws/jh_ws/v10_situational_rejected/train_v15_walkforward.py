"""Expanding-window walk-forward 검증으로 sit_* 피처의 진짜 효과 확인.

단일 홀드아웃(2024 하나)만으로는 노이즈와 진짜 신호를 구분할 수 없다는 걸
924->906 결과로 확인했다. 이번엔 검증년도를 2021/2022/2023/2024로 늘려가며
(각 fold마다 그 시점까지의 trackman 데이터만 사용해 leak 없이) baseline과
sit_* 포함 버전을 나란히 학습해 delta가 여러 해에 걸쳐 일관되는지 확인한다.

fold 구성:
  fold 2021: train(<=2020) -> val(2021)
  fold 2022: train(<=2021) -> val(2022)
  fold 2023: train(<=2022) -> val(2023)
  fold 2024: train(<=2023) -> val(2024)
"""
import os
import random
import warnings
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
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
SIT_COLS = ['balls_before', 'strikes_before', 'outs_before', 'pitcher_hand', 'batter_hand']
VAL_YEARS = [2021, 2022, 2023, 2024]
SEED = 42

print("[1/3] 데이터 로딩...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tm_df = pd.read_csv(os.path.join(DATA_DIR, 'trackman_history.csv'),
                     usecols=['season', 'balls_before', 'strikes_before', 'outs_before',
                              'pitcher_hand', 'batter_hand', 'pitch_type_group'])
tm_df['pitcher_hand'] = tm_df['pitcher_hand'].map(HAND_MAP)
tm_df['batter_hand'] = tm_df['batter_hand'].map(HAND_MAP)
tm_df['is_fastball'] = (tm_df['pitch_type_group'] == 'fastball').astype(int)
tm_df['is_breaking'] = (tm_df['pitch_type_group'] == 'breaking').astype(int)
tm_df['is_offspeed'] = (tm_df['pitch_type_group'] == 'offspeed').astype(int)

train_df['pitcher_hand'] = train_df['pitcher_hand'].astype(str)
train_df['batter_hand'] = train_df['batter_hand'].astype(str)
if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
    train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
    train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
    train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)
if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
    train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

BASE_FEATURE_COLS = [c for c in train_df.columns if c not in ['row_id', 'control_success']]


def build_sit_stats(cutoff_year):
    """cutoff_year보다 이전 시즌의 trackman만 사용 (그 시점 미래 정보 배제)."""
    sub = tm_df[tm_df['season'] < cutoff_year]
    stats = sub.groupby(SIT_COLS).agg(
        sit_fastball_rate=('is_fastball', 'mean'),
        sit_breaking_rate=('is_breaking', 'mean'),
        sit_offspeed_rate=('is_offspeed', 'mean'),
        sit_n=('is_fastball', 'size'),
    ).reset_index()
    fb, bk, os_ = sub['is_fastball'].mean(), sub['is_breaking'].mean(), sub['is_offspeed'].mean()
    return stats, fb, bk, os_


def prepare_xy(df, sit_stats, fallback, drop_sit):
    d = df.copy()
    if not drop_sit:
        stats = sit_stats
        fb, bk, os_ = fallback
        d = d.merge(stats, on=SIT_COLS, how='left')
        d['sit_fastball_rate'] = d['sit_fastball_rate'].fillna(fb)
        d['sit_breaking_rate'] = d['sit_breaking_rate'].fillna(bk)
        d['sit_offspeed_rate'] = d['sit_offspeed_rate'].fillna(os_)
        d['sit_n'] = d['sit_n'].fillna(0)
        feats = BASE_FEATURE_COLS + ['sit_fastball_rate', 'sit_breaking_rate', 'sit_offspeed_rate', 'sit_n']
    else:
        feats = BASE_FEATURE_COLS
    dd = d[feats].copy()
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(dd[c])]
    for c in cat_cols:
        dd[c] = dd[c].astype(str).fillna('missing')
        mapping = {v: i for i, v in enumerate(sorted(dd[c].unique()))}
        dd[c] = dd[c].map(mapping)
    return dd, d['control_success'].copy()


print("[2/3] fold별 학습 (baseline vs sit_*, walk-forward)...")
results = []
for val_year in VAL_YEARS:
    train_mask = (train_df['season'] < val_year).values
    val_mask = (train_df['season'] == val_year).values
    sit_stats, fb, bk, os_ = build_sit_stats(val_year)
    print(f"  === fold val={val_year}: train n={train_mask.sum()}, val n={val_mask.sum()}, "
          f"trackman(<{val_year}) n={len(tm_df[tm_df['season']<val_year])} ===")

    for drop_sit, label in [(True, 'baseline'), (False, 'sit_*')]:
        X, y = prepare_xy(train_df, sit_stats, (fb, bk, os_), drop_sit)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X.loc[val_mask], y.loc[val_mask]

        # leakage 재확인: train에 val_year가 섞이면 안 되고, val은 val_year만 있어야 함
        train_seasons = set(X_train['season'].unique().tolist())
        val_seasons = set(X_val['season'].unique().tolist())
        assert val_year not in train_seasons, f"LEAK! train에 {val_year} 포함됨: {train_seasons}"
        assert val_seasons == {val_year}, f"LEAK! val에 {val_year} 외 다른 시즌 포함됨: {val_seasons}"
        assert len(X_train) == train_mask.sum() and len(X_val) == val_mask.sum(), "행 수 불일치 (merge로 행이 늘거나 줄었을 가능성)"

        random.seed(SEED); os.environ['PYTHONHASHSEED'] = str(SEED); np.random.seed(SEED)
        base_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=SEED, verbose=0)
        base_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)
        calib_cat = get_calibrated_model(base_cat)
        calib_cat.fit(X_val, y_val)
        preds = calib_cat.predict_proba(X_val)[:, 1]
        bss = bss_score(preds, y_val.values)
        results.append((val_year, label, bss))
        print(f"    └ {label} | best_iter={base_cat.get_best_iteration()} | BSS={bss:.1f}")

print("[3/3] 요약 (fold별 delta + 평균)")
by_year = {}
for val_year, label, bss in results:
    by_year.setdefault(val_year, {})[label] = bss
deltas = []
for val_year in VAL_YEARS:
    d = by_year[val_year]
    delta = d['sit_*'] - d['baseline']
    deltas.append(delta)
    print(f"  val={val_year}: baseline={d['baseline']:.1f} -> sit_*={d['sit_*']:.1f} (delta={delta:+.1f})")
print(f"  평균 delta = {np.mean(deltas):+.1f}, 표준편차 = {np.std(deltas):.1f}, "
      f"양수 fold 수 = {sum(1 for x in deltas if x > 0)}/{len(deltas)}")
