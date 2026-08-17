"""DeepFM에서 뽑은 pitcher/batter 임베딩을 CatBoost에 추가 피처로 넣었을 때
효과가 있는지 빠르게 확인 (단일 CatBoost, 진짜 group k-fold 단일 split).
baseline(임베딩 없음) vs with_emb(임베딩 32차원 추가) 비교.
"""
import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss

DATA_DIR = './data'

print("[1/4] 데이터 및 임베딩 로딩...", flush=True)
df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
emb_data = joblib.load('./deepfm_cv_result.pkl')

pitcher_map = emb_data['pitcher_id_to_idx']
batter_map = emb_data['batter_id_to_idx']
pitcher_emb = emb_data['pitcher_emb']
batter_emb = emb_data['batter_emb']
print(f"  pitcher_emb shape={pitcher_emb.shape}, batter_emb shape={batter_emb.shape}", flush=True)

p_idx = df['pitcher_id'].astype(str).map(pitcher_map).values
b_idx = df['batter_id'].astype(str).map(batter_map).values

p_emb_cols = pitcher_emb[p_idx]  # (N, 16)
b_emb_cols = batter_emb[b_idx]   # (N, 16)
dot_interaction = (p_emb_cols * b_emb_cols).sum(axis=1, keepdims=True)  # (N, 1) 매치업 상호작용 스칼라

for i in range(pitcher_emb.shape[1]):
    df[f'pemb_{i}'] = p_emb_cols[:, i]
    df[f'bemb_{i}'] = b_emb_cols[:, i]
df['pb_dot'] = dot_interaction[:, 0]

CAT_COLS = ['top_bottom', 'game_type', 'base_state', 'pitcher_hand', 'batter_hand',
            'pitcher_team_id', 'batter_team_id']
DROP_COLS = ['row_id', 'control_success']

groups = df['pitcher_id'].values
y = df['control_success']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, y, groups=groups))
print(f"[2/4] group split: train={len(train_idx)}, val={len(val_idx)}, "
      f"겹침={len(set(groups[train_idx]) & set(groups[val_idx]))}", flush=True)

emb_cols = [f'pemb_{i}' for i in range(pitcher_emb.shape[1])] + \
           [f'bemb_{i}' for i in range(batter_emb.shape[1])] + ['pb_dot']


def run(feature_cols, label):
    X = df[feature_cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str)
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS if c in X.columns]

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=8, random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    ll = log_loss(y_va, pred)
    print(f"[{label}] AUC={auc:.5f} LogLoss={ll:.5f} best_iter={model.get_best_iteration()} n_features={len(feature_cols)}", flush=True)
    return auc, model


base_cols = [c for c in df.columns if c not in DROP_COLS + emb_cols]
print(f"[3/4] baseline (임베딩 없음, {len(base_cols)}개 피처)...", flush=True)
auc_base, _ = run(base_cols, 'baseline')

with_emb_cols = base_cols + emb_cols
print(f"[4/4] with_emb (임베딩 33개 추가, {len(with_emb_cols)}개 피처)...", flush=True)
auc_emb, model_emb = run(with_emb_cols, 'with_emb')

print(f"\n=== 결과 ===")
print(f"baseline AUC:  {auc_base:.5f}")
print(f"with_emb AUC:  {auc_emb:.5f}")
print(f"차이: {auc_emb - auc_base:+.5f}")

imp = pd.DataFrame({'feature': with_emb_cols, 'importance': model_emb.get_feature_importance()}).sort_values('importance', ascending=False)
print("\ntop-20 importance (with_emb 모델):")
print(imp.head(20).to_string(index=False))
print("\n임베딩 피처 importance 순위:")
emb_rank = imp.reset_index(drop=True)
emb_rank['rank'] = emb_rank.index + 1
print(emb_rank[emb_rank['feature'].isin(emb_cols)].to_string(index=False))
