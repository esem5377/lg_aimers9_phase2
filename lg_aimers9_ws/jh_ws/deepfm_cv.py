"""DeepFM 5-fold 진짜 group k-fold(투수 단위) 정식 검증 + 임베딩 추출.
안정화된 아키텍처(deepfm.py에서 검증됨: 단일 split AUC 0.566)를 그대로 사용.
5-fold OOF로 정식 AUC 확인 + 전체 데이터로 최종 재학습해 pitcher_id/batter_id
임베딩을 뽑아 GBDT에 추가 피처로 투입할 수 있게 저장.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}", flush=True)

DATA_DIR = './data'
CAT_COLS = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
            'pitcher_hand', 'batter_hand', 'top_bottom', 'game_type', 'base_state']
EMB_DIM = 16


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


def prepare(df):
    y = df['control_success'].values.astype(np.float32)
    groups = df['pitcher_id'].values
    cat_maps, cat_arrays = {}, {}
    for c in CAT_COLS:
        vals = df[c].astype(str).fillna('missing')
        uniq = sorted(vals.unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        cat_maps[c] = mapping
        cat_arrays[c] = vals.map(mapping).values.astype(np.int64)
    num_cols = [c for c in df.columns if c not in CAT_COLS + ['row_id', 'control_success']]
    num_df = df[num_cols].astype(np.float32).fillna(0.0)
    scaler = StandardScaler()
    num_arr = scaler.fit_transform(num_df.values).astype(np.float32)
    return cat_arrays, num_arr, y, groups, cat_maps, scaler, num_cols


def to_tensors(cat_arrays, num_arr, idx, device):
    cat_t = {c: torch.from_numpy(cat_arrays[c][idx]).to(device) for c in CAT_COLS}
    num_t = torch.from_numpy(num_arr[idx]).to(device)
    return cat_t, num_t


def train_one(cat_arrays, num_arr, y, train_idx, val_idx, cat_card, num_numeric, n_epochs=30, patience=5):
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
            idx = perm[i:i+batch_size]
            cat_b = {c: cat_tr[c][idx] for c in CAT_COLS}
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
                cat_b = {c: cat_va[c][i:i+batch_size] for c in CAT_COLS}
                num_b = num_va[i:i+batch_size]
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
    with torch.no_grad():
        val_logits = []
        for i in range(0, len(val_idx), batch_size):
            cat_b = {c: cat_va[c][i:i+batch_size] for c in CAT_COLS}
            num_b = num_va[i:i+batch_size]
            val_logits.append(model(cat_b, num_b).cpu())
        val_logits = torch.cat(val_logits).numpy()
    val_probs = 1 / (1 + np.exp(-val_logits))
    return model, val_probs, best_auc


def main():
    print("[1/4] 데이터 로딩...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    cat_arrays, num_arr, y, groups, cat_maps, scaler, num_cols = prepare(df)
    cat_card = {c: len(cat_maps[c]) for c in CAT_COLS}
    print(f"  shape={num_arr.shape}, cardinalities={cat_card}", flush=True)

    print("[2/4] 5-fold 진짜 group k-fold (pitcher_id)...", flush=True)
    gkf = StratifiedGroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y))
    fold_aucs = []
    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(num_arr, y, groups=groups)):
        model, val_probs, best_auc = train_one(cat_arrays, num_arr, y, train_idx, val_idx, cat_card, num_arr.shape[1])
        oof_pred[val_idx] = val_probs
        fold_aucs.append(best_auc)
        print(f"  fold {fold}: best_auc={best_auc:.5f} ({time.time()-t_start:.0f}s elapsed)", flush=True)

    oof_auc = roc_auc_score(y, oof_pred)
    oof_ll = log_loss(y, np.clip(oof_pred, 1e-6, 1-1e-6))
    print(f"[3/4] DeepFM 5-fold OOF: AUC={oof_auc:.5f} LogLoss={oof_ll:.5f} (fold별: {[f'{a:.5f}' for a in fold_aucs]})", flush=True)

    print("[4/4] 전체 데이터로 최종 재학습 (임베딩 추출용)...", flush=True)
    # 전체의 90%를 train, 10%를 early-stop 검증용으로 사용 (group 기준)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_idx, va_idx = next(gss.split(num_arr, y, groups=groups))
    final_model, _, final_auc = train_one(cat_arrays, num_arr, y, tr_idx, va_idx, cat_card, num_arr.shape[1])
    print(f"  최종 모델 검증 AUC(early-stop용 10%): {final_auc:.5f}", flush=True)

    pitcher_map = cat_maps['pitcher_id']
    batter_map = cat_maps['batter_id']
    pitcher_emb = final_model.emb['pitcher_id'].weight.detach().cpu().numpy()
    batter_emb = final_model.emb['batter_id'].weight.detach().cpu().numpy()

    out = {
        'oof_pred': oof_pred,
        'oof_auc': oof_auc,
        'oof_logloss': oof_ll,
        'fold_aucs': fold_aucs,
        'pitcher_id_to_idx': pitcher_map,
        'batter_id_to_idx': batter_map,
        'pitcher_emb': pitcher_emb,  # (792, 16)
        'batter_emb': batter_emb,    # (830, 16)
    }
    joblib.dump(out, './deepfm_cv_result.pkl')
    print("saved ./deepfm_cv_result.pkl", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
