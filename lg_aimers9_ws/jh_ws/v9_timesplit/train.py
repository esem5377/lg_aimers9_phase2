import os
import glob
import joblib
import random
import warnings
import numpy as np
import pandas as pd
import optuna

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# scikit-learn 버전별 prefit/FrozenEstimator 호환성 처리
try:
    from sklearn.frozen import FrozenEstimator
    def get_calibrated_model(base_model):
        return CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method='sigmoid')
except ImportError:
    # 구버전 scikit-learn 대비
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
MODEL_DIR = './model'
os.makedirs(MODEL_DIR, exist_ok=True)

print("[1/6] 데이터 로딩 및 사전 집계...")
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

print("[2/6] 파생변수 생성...")
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

importance_df = pd.DataFrame({
    'feature': features,
    'importance': selector_model.feature_importances_
}).sort_values('importance', ascending=False)

selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
X = X[selected_features].copy()

print("[4/6] 확률 보정(Probability Calibration) 적용 3종 GBDT 학습 (시간 기반 검증: 2019~2023 학습 / 2024 검증)...")
# test.csv는 2025시즌(완전 미래)이라, 학습 때도 실제와 같은 시간적 분포 이동을
# 검증에 반영하기 위해 랜덤/그룹 K-Fold 대신 시즌 기준 단일 홀드아웃을 쓴다.
train_idx = np.where((train_df['season'] <= 2023).values)[0]
val_idx = np.where((train_df['season'] == 2024).values)[0]

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
print(f"  └ train(~2023) n={len(X_train)}, val(2024) n={len(X_val)}")

fold = 0
# 1. LightGBM + Calibration
base_lgb = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.02, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1
)
base_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
calib_lgb = get_calibrated_model(base_lgb)
calib_lgb.fit(X_val, y_val)
joblib.dump(calib_lgb, os.path.join(MODEL_DIR, f'lgb_fold{fold}.pkl'))
oof_lgb[val_idx] = calib_lgb.predict_proba(X_val)[:, 1]
print(f"  └ LightGBM 완료 | 2024 BSS={bss_score(oof_lgb[val_idx], y_val.values):.1f}")

# 2. CatBoost + Calibration
base_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42+fold, verbose=0)
base_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
calib_cat = get_calibrated_model(base_cat)
calib_cat.fit(X_val, y_val)
joblib.dump(calib_cat, os.path.join(MODEL_DIR, f'cat_fold{fold}.pkl'))
oof_cat[val_idx] = calib_cat.predict_proba(X_val)[:, 1]
print(f"  └ CatBoost 완료 | 2024 BSS={bss_score(oof_cat[val_idx], y_val.values):.1f}")

# 3. XGBoost + Calibration
base_xgb = XGBClassifier(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42+fold, eval_metric='logloss', early_stopping_rounds=50
)
base_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
calib_xgb = get_calibrated_model(base_xgb)
calib_xgb.fit(X_val, y_val)
joblib.dump(calib_xgb, os.path.join(MODEL_DIR, f'xgb_fold{fold}.pkl'))
oof_xgb[val_idx] = calib_xgb.predict_proba(X_val)[:, 1]
print(f"  └ XGBoost 완료 | 2024 BSS={bss_score(oof_xgb[val_idx], y_val.values):.1f}")

print("[5/6] Optuna 기반 앙상블 최적 가중치 탐색 중...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    w_lgb = trial.suggest_float('w_lgb', 0.0, 1.0)
    w_cat = trial.suggest_float('w_cat', 0.0, 1.0)
    w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
    
    total_w = w_lgb + w_cat + w_xgb + 1e-6
    w_lgb, w_cat, w_xgb = w_lgb / total_w, w_cat / total_w, w_xgb / total_w
    
    pred = (oof_lgb[val_idx] * w_lgb) + (oof_cat[val_idx] * w_cat) + (oof_xgb[val_idx] * w_xgb)
    pred = np.clip(pred, 0.005, 0.995)
    return log_loss(y_val, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

best_p = study.best_params
total_w = best_p['w_lgb'] + best_p['w_cat'] + best_p['w_xgb'] + 1e-6
opt_weights = {
    'w_lgb': best_p['w_lgb'] / total_w,
    'w_cat': best_p['w_cat'] / total_w,
    'w_xgb': best_p['w_xgb'] / total_w
}

blend_val = (oof_lgb[val_idx] * opt_weights['w_lgb']) + (oof_cat[val_idx] * opt_weights['w_cat']) + (oof_xgb[val_idx] * opt_weights['w_xgb'])
blend_val = np.clip(blend_val, 0.005, 0.995)
print(f"  └ [최적 가중치] LGBM: {opt_weights['w_lgb']:.3f} | Cat: {opt_weights['w_cat']:.3f} | XGB: {opt_weights['w_xgb']:.3f}")
print(f"  └ [최적 OOF Log Loss]: {study.best_value:.5f}")
print(f"  └ [2024 홀드아웃 블렌드 BSS]: {bss_score(blend_val, y_val.values):.1f}")

meta_info = {
    'features': selected_features,
    'cat_cols': [c for c in cat_cols if c in selected_features],
    'cat_mappings': cat_mappings,
    'trackman_records': trackman_records,
    'hist_pitcher_col': hist_p_col,
    'train_pitcher_col': train_p_col,
    'opt_weights': opt_weights
}
joblib.dump(meta_info, os.path.join(MODEL_DIR, 'meta_info.pkl'))

print("[6/6] 모든 파이프라인 완성!")