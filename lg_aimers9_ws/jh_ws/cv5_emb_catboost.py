"""임베딩+CatBoost 하이브리드를 '진짜' 5-fold group k-fold(투수 단위)로 검증.

test_emb_catboost.py는 deepfm_cv_result.pkl의 임베딩(90/10 단일 split으로 학습됨)을
새로운 단일 group split에 재사용해서 baseline vs with_emb를 비교했는데, 이 임베딩의
~90% 투수는 이미 DeepFM 학습에 쓰였던 투수라서 그 fold의 검증 투수와 겹칠 수 있음
(임베딩 자체가 해당 투수의 라벨로 학습된 상태로 CatBoost 검증에 재사용되는 리키지 위험).

이 스크립트는 fold마다 DeepFM을 그 fold의 train 투수만으로 새로 학습해서 임베딩을
뽑고(val 투수는 완전 미학습 상태 임베딩), 그 fold의 CatBoost baseline/with_emb를
동일 train/val로 학습·평가해서 리키지 없는 5-fold OOF AUC를 비교한다.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}", flush=True)

DATA_DIR = './data'
CAT_COLS_DFM = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
                'pitcher_hand', 'batter_hand', 'top_bottom', 'game_type', 'base_state']
EMB_DIM = 16
CAT_COLS_CB = ['top_bottom', 'game_type', 'base_state', 'pitcher_hand', 'batter_hand',
               'pitcher_team_id', 'batter_team_id']
DROP_COLS = ['row_id', 'control_success']


class DeepFM(nn.Module):
    def __init__(self, cat_cardinalities, num_numeric, emb_dim=EMB_DIM, mlp_dims=(256, 128, 64)):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.emb = nn.ModuleDict({c: nn.Embedding(card, emb_dim) for c, card in cat_cardinalities.items()})
        self.lin_emb = nn.ModuleDict({c: nn.Embedding(card, 1) for c, card in cat_cardinalities.items()})
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.01)
        for e in self.lin_emb.values():
            nn.init.zeros_(e.weight)
        self.num_linear = nn.Linear(num_numeric, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        mlp_in = emb_dim * len(self.cat_cols) + num_numeric
        layers = []
        prev = mlp_in
        for h in mlp_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_inputs, num_input):
        embs = [self.emb[c](cat_inputs[c]) for c in self.cat_cols]
        stacked = torch.stack(embs, dim=1)
        sum_sq = stacked.sum(dim=1) ** 2
        sq_sum = (stacked ** 2).sum(dim=1)
        fm_term = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)
        lin_terms = sum(self.lin_emb[c](cat_inputs[c]) for c in self.cat_cols)
        num_lin = self.num_linear(num_input)
        deep_in = torch.cat([stacked.flatten(1), num_input], dim=1)
        deep_out = self.mlp(deep_in)
        logit = self.bias + lin_terms.squeeze(-1) + num_lin.squeeze(-1) + fm_term.squeeze(-1) + deep_out.squeeze(-1)
        return logit


def prepare_dfm(df):
    y = df['control_success'].values.astype(np.float32)
    groups = df['pitcher_id'].values
    cat_maps, cat_arrays = {}, {}
    for c in CAT_COLS_DFM:
        vals = df[c].astype(str).fillna('missing')
        uniq = sorted(vals.unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        cat_maps[c] = mapping
        cat_arrays[c] = vals.map(mapping).values.astype(np.int64)
    num_cols = [c for c in df.columns if c not in CAT_COLS_DFM + ['row_id', 'control_success']]
    num_df = df[num_cols].astype(np.float32).fillna(0.0)
    scaler = StandardScaler()
    num_arr = scaler.fit_transform(num_df.values).astype(np.float32)
    return cat_arrays, num_arr, y, groups, cat_maps, num_cols


def to_tensors(cat_arrays, num_arr, idx, device):
    cat_t = {c: torch.from_numpy(cat_arrays[c][idx]).to(device) for c in CAT_COLS_DFM}
    num_t = torch.from_numpy(num_arr[idx]).to(device)
    return cat_t, num_t


def train_dfm_fold(cat_arrays, num_arr, y, train_idx, val_idx, cat_card, num_numeric, n_epochs=30, patience=5):
    model = DeepFM(cat_card, num_numeric).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    cat_tr, num_tr = to_tensors(cat_arrays, num_arr, train_idx, DEVICE)
    y_tr = torch.from_numpy(y[train_idx]).to(DEVICE)
    cat_va, num_va = to_tensors(cat_arrays, num_arr, val_idx, DEVICE)
    y_va_np = y[val_idx]

    n_train = len(train_idx)
    batch_size = 8192
    best_auc, best_state, bad_epochs = -1, None, 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            cat_b = {c: cat_tr[c][idx] for c in CAT_COLS_DFM}
            num_b = num_tr[idx]
            yb = y_tr[idx]
            opt.zero_grad()
            loss = loss_fn(model(cat_b, num_b), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, len(val_idx), batch_size):
                cat_b = {c: cat_va[c][i:i + batch_size] for c in CAT_COLS_DFM}
                num_b = num_va[i:i + batch_size]
                val_logits.append(model(cat_b, num_b).cpu())
            val_logits = torch.cat(val_logits).numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))
        auc = roc_auc_score(y_va_np, val_probs)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


def run_catboost(df, feature_cols, train_idx, val_idx, y, label, fold, seed=42):
    X = df[feature_cols].copy()
    for c in CAT_COLS_CB:
        if c in X.columns:
            X[c] = X[c].astype(str)
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS_CB if c in X.columns]

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=8, random_seed=seed,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    print(f"  fold {fold} [{label}] AUC={auc:.5f} best_iter={model.get_best_iteration()} n_features={len(feature_cols)}", flush=True)
    return pred, auc


def main():
    t0 = time.time()
    print("[1/3] 데이터 로딩...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    cat_arrays, num_arr, y_np, groups, cat_maps, num_cols = prepare_dfm(df)
    cat_card = {c: len(cat_maps[c]) for c in CAT_COLS_DFM}
    y = df['control_success']

    base_cols = [c for c in df.columns if c not in DROP_COLS]
    emb_cols = [f'pemb_{i}' for i in range(EMB_DIM)] + [f'bemb_{i}' for i in range(EMB_DIM)] + ['pb_dot']
    with_emb_cols = base_cols + emb_cols

    print("[2/3] 5-fold 진짜 group k-fold (pitcher_id) — fold별 DeepFM 재학습 + CatBoost 2종...", flush=True)
    gkf = StratifiedGroupKFold(n_splits=5)
    splits = list(gkf.split(num_arr, y_np, groups=groups))

    oof_base = np.zeros(len(y_np))
    oof_emb = np.zeros(len(y_np))
    fold_base_aucs, fold_emb_aucs, fold_dfm_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        overlap = len(set(groups[train_idx]) & set(groups[val_idx]))
        print(f"fold {fold}: train={len(train_idx)} val={len(val_idx)} pitcher_overlap={overlap}", flush=True)

        # (a) 이 fold의 train 투수만으로 DeepFM 학습 -> val 투수는 미학습 임베딩(초기화 그대로)
        dfm_model, dfm_auc = train_dfm_fold(cat_arrays, num_arr, y_np, train_idx, val_idx, cat_card, num_arr.shape[1])
        fold_dfm_aucs.append(dfm_auc)
        pitcher_emb = dfm_model.emb['pitcher_id'].weight.detach().cpu().numpy()
        batter_emb = dfm_model.emb['batter_id'].weight.detach().cpu().numpy()
        print(f"  fold {fold} DeepFM val_auc={dfm_auc:.5f} (참고용, 임베딩 추출 목적)", flush=True)

        # (b) 이 fold 전용 임베딩으로 피처 컬럼 부착 (val 투수는 학습 안 된 near-zero 임베딩)
        p_idx = cat_arrays['pitcher_id']
        b_idx = cat_arrays['batter_id']
        p_emb_cols = pitcher_emb[p_idx]
        b_emb_cols = batter_emb[b_idx]
        dot_interaction = (p_emb_cols * b_emb_cols).sum(axis=1)

        df_fold = df.copy()
        for i in range(EMB_DIM):
            df_fold[f'pemb_{i}'] = p_emb_cols[:, i]
            df_fold[f'bemb_{i}'] = b_emb_cols[:, i]
        df_fold['pb_dot'] = dot_interaction

        # (c) CatBoost baseline vs with_emb, 동일 train/val
        pred_base, auc_base = run_catboost(df_fold, base_cols, train_idx, val_idx, y, 'baseline', fold)
        pred_emb, auc_emb = run_catboost(df_fold, with_emb_cols, train_idx, val_idx, y, 'with_emb', fold)

        oof_base[val_idx] = pred_base
        oof_emb[val_idx] = pred_emb
        fold_base_aucs.append(auc_base)
        fold_emb_aucs.append(auc_emb)
        print(f"  fold {fold} 차이: {auc_emb - auc_base:+.5f}  ({time.time()-t0:.0f}s elapsed total)", flush=True)

    oof_auc_base = roc_auc_score(y_np, oof_base)
    oof_auc_emb = roc_auc_score(y_np, oof_emb)
    oof_ll_base = log_loss(y_np, np.clip(oof_base, 1e-6, 1 - 1e-6))
    oof_ll_emb = log_loss(y_np, np.clip(oof_emb, 1e-6, 1 - 1e-6))

    print("\n[3/3] === 5-fold OOF 최종 결과 (리키지 없음, fold별 DeepFM 재학습) ===", flush=True)
    print(f"baseline : OOF AUC={oof_auc_base:.5f} LogLoss={oof_ll_base:.5f}  fold_aucs={[f'{a:.5f}' for a in fold_base_aucs]}", flush=True)
    print(f"with_emb : OOF AUC={oof_auc_emb:.5f} LogLoss={oof_ll_emb:.5f}  fold_aucs={[f'{a:.5f}' for a in fold_emb_aucs]}", flush=True)
    print(f"차이(OOF)  : {oof_auc_emb - oof_auc_base:+.5f}", flush=True)
    print(f"fold별 차이 : {[f'{e-b:+.5f}' for e, b in zip(fold_emb_aucs, fold_base_aucs)]}", flush=True)
    print(f"DeepFM fold AUCs (참고): {[f'{a:.5f}' for a in fold_dfm_aucs]}", flush=True)

    joblib.dump({
        'oof_base': oof_base, 'oof_emb': oof_emb,
        'oof_auc_base': oof_auc_base, 'oof_auc_emb': oof_auc_emb,
        'fold_base_aucs': fold_base_aucs, 'fold_emb_aucs': fold_emb_aucs,
        'fold_dfm_aucs': fold_dfm_aucs,
    }, './cv5_emb_catboost_result.pkl')
    print(f"\ntotal elapsed: {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
