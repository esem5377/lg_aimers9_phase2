import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss

DATA_DIR = './data'
MODEL_DIR = './model'

meta = joblib.load(os.path.join(MODEL_DIR, 'meta_info.pkl'))
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
    train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
    train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
    train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)
if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
    train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

selected_features = meta['features']
cat_mappings = meta['cat_mappings']
train_p_col = meta['train_pitcher_col']

X = train_df[selected_features].copy()
for col, cat_map in cat_mappings.items():
    if col in X.columns:
        vals = [str(v) if pd.notna(v) else 'missing' for v in X[col]]
        X[col] = [cat_map.get(v, -1) for v in vals]

y = train_df['control_success'].copy()
groups = train_df[train_p_col]

gkf = StratifiedGroupKFold(n_splits=5)
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_val = X.iloc[val_idx]
    calib_lgb = joblib.load(os.path.join(MODEL_DIR, f'lgb_fold{fold}.pkl'))
    calib_cat = joblib.load(os.path.join(MODEL_DIR, f'cat_fold{fold}.pkl'))
    calib_xgb = joblib.load(os.path.join(MODEL_DIR, f'xgb_fold{fold}.pkl'))
    oof_lgb[val_idx] = calib_lgb.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = calib_cat.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = calib_xgb.predict_proba(X_val)[:, 1]
    print(f'fold {fold} predicted', flush=True)

w = meta['opt_weights']
pred = oof_lgb * w['w_lgb'] + oof_cat * w['w_cat'] + oof_xgb * w['w_xgb']
pred_clipped = np.clip(pred, w['clip_eps'], 1 - w['clip_eps'])

print('OOF AUC (blend):', roc_auc_score(y, pred))
print('OOF LogLoss (blend, clipped):', log_loss(y, pred_clipped), '(원래 기록된 0.68269와 대조용)')
print('OOF AUC lgb only:', roc_auc_score(y, oof_lgb))
print('OOF AUC cat only:', roc_auc_score(y, oof_cat))
print('OOF AUC xgb only:', roc_auc_score(y, oof_xgb))
