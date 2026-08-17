"""DeepFM류 임베딩 모델 — pitcher x batter 매치업 상호작용을 잡기 위한 시도.
기존 GBDT 피처는 전부 단변량(투수 개인 성공률, 타자 개인 성공률)이라
"이 투수 vs 이 타자" 상호작용을 직접 표현하지 못함. FM 성분이 저차원
임베딩끼리의 쌍별 내적으로 이 상호작용을 저비용으로 학습.

빠른 반복을 위해 우선 단일 group split(투수 단위 80/20)으로 학습/검증.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}", flush=True)

DATA_DIR = './data'

CAT_COLS = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
            'pitcher_hand', 'batter_hand', 'top_bottom', 'game_type', 'base_state']
EMB_DIM = 16

NUM_COLS = None  # 아래서 자동 산출 (row_id/control_success/CAT_COLS 제외 전부)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    return df


def prepare(df):
    global NUM_COLS
    y = df['control_success'].values.astype(np.float32)
    groups = df['pitcher_id'].values

    cat_maps = {}
    cat_arrays = {}
    for c in CAT_COLS:
        vals = df[c].astype(str).fillna('missing')
        uniq = sorted(vals.unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        cat_maps[c] = mapping
        cat_arrays[c] = vals.map(mapping).values.astype(np.int64)

    NUM_COLS = [c for c in df.columns if c not in CAT_COLS + ['row_id', 'control_success']]
    num_df = df[NUM_COLS].astype(np.float32).fillna(0.0)
    scaler = StandardScaler()
    num_arr = scaler.fit_transform(num_df.values).astype(np.float32)

    return cat_arrays, num_arr, y, groups, cat_maps, scaler


class DeepFM(nn.Module):
    def __init__(self, cat_cardinalities, num_numeric, emb_dim=EMB_DIM, mlp_dims=(256, 128, 64)):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.emb = nn.ModuleDict({
            c: nn.Embedding(card, emb_dim) for c, card in cat_cardinalities.items()
        })
        self.lin_emb = nn.ModuleDict({
            c: nn.Embedding(card, 1) for c, card in cat_cardinalities.items()
        })
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
        embs = [self.emb[c](cat_inputs[c]) for c in self.cat_cols]  # each (B, emb_dim)
        stacked = torch.stack(embs, dim=1)  # (B, F, emb_dim)

        # FM pairwise interaction (efficient sum-square trick)
        sum_sq = stacked.sum(dim=1) ** 2
        sq_sum = (stacked ** 2).sum(dim=1)
        fm_term = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)  # (B, 1)

        lin_terms = sum(self.lin_emb[c](cat_inputs[c]) for c in self.cat_cols)  # (B, 1)
        num_lin = self.num_linear(num_input)  # (B, 1)

        deep_in = torch.cat([stacked.flatten(1), num_input], dim=1)
        deep_out = self.mlp(deep_in)  # (B, 1)

        logit = self.bias + lin_terms.squeeze(-1) + num_lin.squeeze(-1) + fm_term.squeeze(-1) + deep_out.squeeze(-1)
        return logit


def to_tensors(cat_arrays, num_arr, idx, device):
    cat_t = {c: torch.from_numpy(cat_arrays[c][idx]).to(device) for c in CAT_COLS}
    num_t = torch.from_numpy(num_arr[idx]).to(device)
    return cat_t, num_t


def main():
    print("[1/5] 데이터 로딩...", flush=True)
    df = load_data()
    cat_arrays, num_arr, y, groups, cat_maps, scaler = prepare(df)
    print(f"  shape: {num_arr.shape}, cat fields: {CAT_COLS}", flush=True)

    print("[2/5] group split (투수 단위 80/20)...", flush=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(num_arr, y, groups=groups))
    print(f"  train={len(train_idx)}, val={len(val_idx)}, "
          f"투수 겹침={len(set(groups[train_idx]) & set(groups[val_idx]))}개", flush=True)

    cat_card = {c: len(cat_maps[c]) for c in CAT_COLS}
    print(f"  cardinalities: {cat_card}", flush=True)

    model = DeepFM(cat_card, num_arr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    print("[3/5] 학습 데이터 텐서 준비...", flush=True)
    cat_tr, num_tr = to_tensors(cat_arrays, num_arr, train_idx, DEVICE)
    y_tr = torch.from_numpy(y[train_idx]).to(DEVICE)
    cat_va, num_va = to_tensors(cat_arrays, num_arr, val_idx, DEVICE)
    y_va_np = y[val_idx]

    n_train = len(train_idx)
    batch_size = 8192
    n_epochs = 40
    best_auc = -1
    best_state = None
    patience, bad_epochs = 5, 0

    print("[4/5] 학습 시작...", flush=True)
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            cat_b = {c: cat_tr[c][idx] for c in CAT_COLS}
            num_b = num_tr[idx]
            yb = y_tr[idx]

            opt.zero_grad()
            logit = model(cat_b, num_b)
            loss = loss_fn(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, len(val_idx), batch_size):
                cat_b = {c: cat_va[c][i:i+batch_size] for c in CAT_COLS}
                num_b = num_va[i:i+batch_size]
                val_logits.append(model(cat_b, num_b).cpu())
            val_logits = torch.cat(val_logits).numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
            auc = roc_auc_score(y_va_np, val_probs)
            ll = log_loss(y_va_np, np.clip(val_probs, 1e-6, 1-1e-6))

        dt = time.time() - t0
        print(f"  epoch {epoch}: train_loss={total_loss/n_train:.5f} val_auc={auc:.5f} val_logloss={ll:.5f} ({dt:.1f}s)", flush=True)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch}", flush=True)
                break

    print(f"[5/5] 최종 best val AUC: {best_auc:.5f}", flush=True)
    torch.save(best_state, './model_deepfm_best.pt')
    print("done", flush=True)


if __name__ == "__main__":
    main()
