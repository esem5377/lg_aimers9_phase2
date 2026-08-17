"""jh_ws/train.py 기반 실험 버전 (v2) — 두 가지 수정 적용:
1. raw pitcher_id/batter_id를 피처에서 제외 (GroupKFold 그룹핑에는 계속 사용).
2. 깨져있던 trackman_history 조인을 es_ws에서 검증된 상황 키(count_state/
   hand_matchup/inning_state) 기반 컨텍스트 피처로 교체.

나머지 구조(3모델 GroupKFold+보정+Optuna 블렌드)는 원본 train.py와 동일.
"""
import os
import joblib
import random
import warnings
import numpy as np
import pandas as pd
import optuna

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score

try:
    from sklearn.frozen import FrozenEstimator
    def get_calibrated_model(base_model):
        return CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method='sigmoid')
except ImportError:
    def get_calibrated_model(base_model):
        return CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

DATA_DIR = './data'
MODEL_DIR = './model_v2'
TRACKMAN_CONTEXT_PATH = '/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl'
os.makedirs(MODEL_DIR, exist_ok=True)

print("[1/6] 데이터 로딩 및 트랙맨 컨텍스트(상황 키 기반) 부착...", flush=True)
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

trackman_context = joblib.load(TRACKMAN_CONTEXT_PATH)
for spec in trackman_context.values():
    train_df = train_df.merge(spec['table'], on=spec['keys'], how='left')

print("[2/6] 핵심 검증 파생변수 생성...", flush=True)
if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
    train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
    train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
    train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)

if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
    train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

ignore_cols = ['row_id', 'control_success', 'pitcher_id', 'batter_id']
train_p_col = 'pitcher_id'
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

print("[3/6] 다중공선성 제거 및 중요 변수 선별...", flush=True)
num_cols = [c for c in features if c not in cat_cols]
corr_matrix = X[num_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

if to_drop:
    print(f"  └ 상관관계>0.95로 제거: {to_drop}")
    X = X.drop(columns=to_drop)
    features = [f for f in features if f not in to_drop]

selector_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
selector_model.fit(X, y)

importance_df = pd.DataFrame({
    'feature': features,
    'importance': selector_model.feature_importances_
}).sort_values('importance', ascending=False)

selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
X = X[selected_features].copy()
print(f"  └ 선택된 피처 개수: {len(selected_features)}개", flush=True)
print(f"  └ 선택된 피처: {selected_features}")

print("[4/6] 5-Fold 확률 보정(Probability Calibration) GBDT 학습...", flush=True)
gkf = StratifiedGroupKFold(n_splits=5)
groups = train_df[train_p_col]

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    base_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1, n_jobs=-1
    )
    base_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    calib_lgb = get_calibrated_model(base_lgb)
    calib_lgb.fit(X_val, y_val)
    joblib.dump(calib_lgb, os.path.join(MODEL_DIR, f'lgb_fold{fold}.pkl'))
    oof_lgb[val_idx] = calib_lgb.predict_proba(X_val)[:, 1]

    base_cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6, random_seed=42+fold, verbose=0, thread_count=-1
    )
    base_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    calib_cat = get_calibrated_model(base_cat)
    calib_cat.fit(X_val, y_val)
    joblib.dump(calib_cat, os.path.join(MODEL_DIR, f'cat_fold{fold}.pkl'))
    oof_cat[val_idx] = calib_cat.predict_proba(X_val)[:, 1]

    base_xgb = XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42+fold, eval_metric='logloss', early_stopping_rounds=50, n_jobs=-1
    )
    base_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    calib_xgb = get_calibrated_model(base_xgb)
    calib_xgb.fit(X_val, y_val)
    joblib.dump(calib_xgb, os.path.join(MODEL_DIR, f'xgb_fold{fold}.pkl'))
    oof_xgb[val_idx] = calib_xgb.predict_proba(X_val)[:, 1]

    print(f"  └ Fold {fold} 학습 및 보정 완료!", flush=True)

print("[5/6] Optuna 앙상블 최적 가중치 & 확률 클리핑 범위 탐색...", flush=True)
def ensemble_objective(trial):
    w_lgb = trial.suggest_float('w_lgb', 0.0, 1.0)
    w_cat = trial.suggest_float('w_cat', 0.0, 1.0)
    w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
    clip_eps = trial.suggest_float('clip_eps', 0.001, 0.02)

    total_w = w_lgb + w_cat + w_xgb + 1e-6
    w_lgb, w_cat, w_xgb = w_lgb / total_w, w_cat / total_w, w_xgb / total_w

    pred = (oof_lgb * w_lgb) + (oof_cat * w_cat) + (oof_xgb * w_xgb)
    pred = np.clip(pred, clip_eps, 1.0 - clip_eps)

    return log_loss(y, pred)

study_ens = optuna.create_study(direction='minimize')
study_ens.optimize(ensemble_objective, n_trials=300)

best_ens = study_ens.best_params
total_w = best_ens['w_lgb'] + best_ens['w_cat'] + best_ens['w_xgb'] + 1e-6
opt_weights = {
    'w_lgb': best_ens['w_lgb'] / total_w,
    'w_cat': best_ens['w_cat'] / total_w,
    'w_xgb': best_ens['w_xgb'] / total_w,
    'clip_eps': best_ens['clip_eps']
}

print(f"  └ [최적 가중치] LGBM: {opt_weights['w_lgb']:.3f} | Cat: {opt_weights['w_cat']:.3f} | XGB: {opt_weights['w_xgb']:.3f}")
print(f"  └ [최적 확률 클리핑 범위]: [{opt_weights['clip_eps']:.4f}, {1-opt_weights['clip_eps']:.4f}]")
print(f"  └ [최적 OOF Log Loss]: {study_ens.best_value:.5f}")

final_pred = oof_lgb * opt_weights['w_lgb'] + oof_cat * opt_weights['w_cat'] + oof_xgb * opt_weights['w_xgb']
print(f"  └ [OOF AUC blend]: {roc_auc_score(y, final_pred):.5f}")
print(f"  └ [OOF AUC lgb/cat/xgb]: {roc_auc_score(y, oof_lgb):.5f} / {roc_auc_score(y, oof_cat):.5f} / {roc_auc_score(y, oof_xgb):.5f}")

meta_info = {
    'features': selected_features,
    'cat_cols': [c for c in cat_cols if c in selected_features],
    'cat_mappings': cat_mappings,
    'trackman_context_path': TRACKMAN_CONTEXT_PATH,
    'train_pitcher_col': train_p_col,
    'opt_weights': opt_weights
}
joblib.dump(meta_info, os.path.join(MODEL_DIR, 'meta_info.pkl'))

print("[6/6] v2(raw id 제거 + 트랙맨 조인 수정) 학습 완료!")
