"""개체 ID 없는 MLP(범주형 임베딩)로 CatBoost를 대체할 수 있는지 검증.

지금까지 실패한 방향(트랙맨 조인, DeepFM 매치업 임베딩, situational target
encoding)의 공통점은 둘 중 하나였다: (a) pitcher_id 기반이라 진짜
group k-fold(투수가 train/valid에 안 겹침)에서 cold-start로 죽거나,
(b) 이미 주어진 asof_* 피처와 정보가 겹쳐서 추가 신호가 없거나.

이번엔 피처는 그대로(raw pitcher_id/batter_id 제외, train_catboost.py와
동일한 피처셋 + trackman_context) 두고 **모델 클래스만** GBDT(CatBoost)에서
범주형 임베딩 + MLP로 바꾼다. 개체 식별자를 전혀 안 쓰므로 cold-start
문제가 원천적으로 없고, GBDT의 axis-aligned split과는 다른 형태의
비선형 상호작용(피처 곱셈 등)을 잡을 수 있는지가 관건이다.

tune_situational_te.py와 완전히 동일한 StratifiedGroupKFold(5, seed=42)를
재사용해 baseline(CatBoost) 결과와 폴드 단위로 직접 비교 가능하게 한다.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
DROP_COLS = ["pitcher_id", "batter_id"]

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "season", "game_month", "game_dayofweek",
]

N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8192
MAX_EPOCHS = 60
PATIENCE = 8
LR = 1e-3
EMB_DIM_CAP = 16


def merge_trackman_context(df):
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    out = df.copy()
    for spec in context.values():
        out = out.merge(spec["table"], on=spec["keys"], how="left")
    return out


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df = merge_trackman_context(df)
    return df


def split_columns(df):
    drop = [c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return cat_cols, num_cols


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class TabMLP(nn.Module):
    def __init__(self, cat_cardinalities, n_num):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cat_cardinalities:
            dim = min(EMB_DIM_CAP, max(2, (card + 1) // 2))
            self.embs.append(nn.Embedding(card + 1, dim))  # +1: unknown/NaN 슬롯
            emb_total += dim
        in_dim = emb_total + n_num
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x).squeeze(-1)


def build_cat_vocab(df, cat_cols):
    """전체 데이터 기준 카테고리 -> 정수 인덱스 맵(라벨 정보 없음, 리키지 아님)."""
    vocabs = {}
    for c in cat_cols:
        vals = sorted(df[c].astype(str).fillna("__NA__").unique())
        vocabs[c] = {v: i + 1 for i, v in enumerate(vals)}  # 0은 unseen/NaN 예약
    return vocabs


def encode_cat(df, cat_cols, vocabs):
    arr = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, c in enumerate(cat_cols):
        vmap = vocabs[c]
        arr[:, i] = df[c].astype(str).fillna("__NA__").map(vmap).fillna(0).astype(np.int64)
    return arr


def fit_eval_fold(df_tr, df_va, cat_cols, num_cols, vocabs):
    X_cat_tr = encode_cat(df_tr, cat_cols, vocabs)
    X_cat_va = encode_cat(df_va, cat_cols, vocabs)

    medians = df_tr[num_cols].median()
    means = df_tr[num_cols].fillna(medians).mean()
    stds = df_tr[num_cols].fillna(medians).std().replace(0, 1.0)

    X_num_tr = ((df_tr[num_cols].fillna(medians) - means) / stds).values.astype(np.float32)
    X_num_va = ((df_va[num_cols].fillna(medians) - means) / stds).values.astype(np.float32)

    y_tr = df_tr[TARGET_COL].values.astype(np.float32)
    y_va = df_va[TARGET_COL].values.astype(np.float32)

    ds_tr = TabDataset(X_cat_tr, X_num_tr, y_tr)
    ds_va = TabDataset(X_cat_va, X_num_va, y_va)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=65536, shuffle=False)

    cardinalities = [len(vocabs[c]) for c in cat_cols]
    model = TabMLP(cardinalities, len(num_cols)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_epoch, patience_ctr = -1.0, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb_cat, xb_num, yb in dl_tr:
            xb_cat, xb_num, yb = xb_cat.to(DEVICE), xb_num.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb_cat, xb_num)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb_cat, xb_num, yb in dl_va:
                xb_cat, xb_num = xb_cat.to(DEVICE), xb_num.to(DEVICE)
                preds.append(torch.sigmoid(model(xb_cat, xb_num)).cpu().numpy())
        preds = np.concatenate(preds)
        auc = roc_auc_score(y_va, preds)
        sched.step(auc)

        improved = auc > best_auc + 1e-5
        if improved:
            best_auc, best_epoch, patience_ctr = auc, epoch, 0
        else:
            patience_ctr += 1
        if epoch % 3 == 0 or improved:
            print(f"    epoch{epoch:02d} val_auc={auc:.5f} best={best_auc:.5f}@{best_epoch} "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if patience_ctr >= PATIENCE:
            print(f"    early stop at epoch{epoch}", flush=True)
            break

    return best_auc, best_epoch


def main():
    t_start = time.time()
    print(f"device={DEVICE}", flush=True)
    print("Load train.csv + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    cat_cols, num_cols = split_columns(df)
    print(f" n_cat={len(cat_cols)} n_num={len(num_cols)}", flush=True)
    vocabs = build_cat_vocab(df, cat_cols)

    groups = df["pitcher_id"].values
    y = df[TARGET_COL].values
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(df, y, groups)):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        n_overlap = len(set(df_tr["pitcher_id"]) & set(df_va["pitcher_id"]))
        print(f"\n[fold{fold_i}] n_train={len(df_tr)} n_valid={len(df_va)} "
              f"pitcher_overlap={n_overlap}", flush=True)
        t0 = time.time()
        auc, best_epoch = fit_eval_fold(df_tr, df_va, cat_cols, num_cols, vocabs)
        print(f"  -> best_val_auc={auc:.5f} @epoch{best_epoch} ({time.time()-t0:.1f}s)", flush=True)
        fold_aucs.append(auc)

    mean_auc = float(np.mean(fold_aucs))
    print(f"\n{'='*60}\nSUMMARY (elapsed {time.time()-t_start:.0f}s)\n{'='*60}")
    print(f"MLP(no-identity)  mean_auc={mean_auc:.5f}  per-fold={['%.5f'%a for a in fold_aucs]}")
    print("CatBoost baseline(tune_situational_te.py) per-fold=[0.57316, 0.57059, 0.57504, 0.56616, 0.57358] mean=0.57171")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_mlp_no_identity_results.json"), "w") as f:
        json.dump({"fold_aucs": fold_aucs, "mean_auc": mean_auc}, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_mlp_no_identity_results.json")


if __name__ == "__main__":
    main()
