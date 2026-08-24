"""1단계(pitcher x batter만 넣은 순수 매치업 FM)가 fold2_2024에서 완전히
무너진 건, ID 페어 단독으로는 정보가 너무 희소해서(그 페어가 처음 보는
조합이면 사실상 예측할 게 없음) 과적합했을 가능성이 큼. 상황 피처(카운트,
아웃, 주자, 좌우 매치업 등)를 같이 넣은 완전한 multi-field FM으로 확장해서
재검증 -- 페어가 희소해도 상황 정보로 보완되는지 확인.

y = w0 + sum_f b_f[cat_f] + 0.5*(||sum_f e_f||^2 - sum_f ||e_f||^2)
(표준 2-way FM의 "제곱합-합의제곱" 트릭, 필드 수와 무관하게 O(필드수 x 임베딩차원)으로 계산)
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/fm_multifield_walkforward.json"
TARGET_COL = "control_success"
SEED = 42
EMBED_DIM = 16
EPOCHS = 15
BATCH_SIZE = 4096
LR = 5e-3
WEIGHT_DECAY = 1e-5

FIELDS = [
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "base_state", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before",
    "pitcher_team_id", "batter_team_id",
]


def log(msg):
    print(msg, flush=True)


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


class MultiFieldFM(nn.Module):
    def __init__(self, cardinalities, embed_dim):
        super().__init__()
        self.fields = list(cardinalities.keys())
        self.w0 = nn.Parameter(torch.zeros(1))
        self.biases = nn.ModuleDict()
        self.embeds = nn.ModuleDict()
        for f, card in cardinalities.items():
            self.biases[f] = nn.Embedding(card + 1, 1, padding_idx=card)
            self.embeds[f] = nn.Embedding(card + 1, embed_dim, padding_idx=card)
            nn.init.zeros_(self.biases[f].weight)
            nn.init.normal_(self.embeds[f].weight, std=0.01)

    def forward(self, cat_tensors):
        linear = self.w0
        embs = []
        for f in self.fields:
            linear = linear + self.biases[f](cat_tensors[f]).squeeze(1)
            embs.append(self.embeds[f](cat_tensors[f]))
        stacked = torch.stack(embs, dim=1)  # (batch, n_fields, embed_dim)
        sum_sq = stacked.sum(dim=1).pow(2).sum(dim=1)
        sq_sum = stacked.pow(2).sum(dim=(1, 2))
        interaction = 0.5 * (sum_sq - sq_sum)
        return linear + interaction


def build_mappings(train_df):
    mappings, cardinalities = {}, {}
    for f in FIELDS:
        uniq = sorted(train_df[f].astype(str).unique())
        mappings[f] = {v: i for i, v in enumerate(uniq)}
        cardinalities[f] = len(uniq)
    return mappings, cardinalities


def encode(df, mappings):
    out = {}
    for f, m in mappings.items():
        card = len(m)
        out[f] = df[f].astype(str).map(m).fillna(card).astype(int).values
    return out


def to_tensors(enc, device):
    return {f: torch.tensor(v, dtype=torch.long, device=device) for f, v in enc.items()}


def train_fm(enc_train, y_train, cardinalities, device):
    model = MultiFieldFM(cardinalities, EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    cat_t = to_tensors(enc_train, device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(y_train)

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED + epoch)
        idx = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            batch = idx[i:i + BATCH_SIZE]
            batch_cat = {f: v[batch] for f, v in cat_t.items()}
            opt.zero_grad()
            logit = model(batch_cat)
            loss = loss_fn(logit, y_t[batch])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"    epoch {epoch+1}/{EPOCHS} loss={epoch_loss/n_batches:.5f}")
    return model


@torch.no_grad()
def predict_fm(model, enc, device):
    model.eval()
    cat_t = to_tensors(enc, device)
    logit = model(cat_t)
    return torch.sigmoid(logit).cpu().numpy()


def run_fold(tag, train_df, eval_df, device):
    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)

    mappings, cardinalities = build_mappings(train_sub)
    enc_train = encode(train_sub, mappings)
    y_train = train_sub[TARGET_COL].values.astype(np.float32)
    enc_calib = encode(calib, mappings)
    y_calib = calib[TARGET_COL].values
    enc_eval = encode(eval_df, mappings)
    y_eval = eval_df[TARGET_COL].values

    t0 = time.time()
    model = train_fm(enc_train, y_train, cardinalities, device)
    elapsed = time.time() - t0

    calib_raw = predict_fm(model, enc_calib, device)
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = predict_fm(model, enc_eval, device)
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag,
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    log(f"  [{tag}] auc={result['auc']:.4f} bss_calib={result['bss_calibrated']:.2f} ({elapsed:.0f}s)")
    return result, eval_calib, y_eval


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    cols = ["row_id", "season", TARGET_COL] + FIELDS
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=list(set(cols)))
    log(f"shape={df.shape}")

    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    baselines = {"fold0_2022": 2386.4930909141212, "fold2_2024": 832.4060438655523}
    results = {}
    for fold_name, (train_df, eval_df) in fold_specs.items():
        log(f"\n=== {fold_name} ===")
        r, eval_calib, y_eval = run_fold(fold_name, train_df, eval_df, device)
        r["baseline_calibrated"] = baselines[fold_name]
        results[fold_name] = r

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("\n=== SUMMARY (multi-field FM 단독 AUC/BSS, CatBoost baseline은 참고용) ===")
    for fn in fold_specs:
        log(f"  {fn}: FM auc={results[fn]['auc']:.4f} bss_calib={results[fn]['bss_calibrated']:.2f}  "
            f"(CatBoost baseline bss_calib={baselines[fn]:.2f}, 71피처 참고치)")
    log(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
