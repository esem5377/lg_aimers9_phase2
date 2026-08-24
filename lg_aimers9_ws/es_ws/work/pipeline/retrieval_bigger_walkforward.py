"""retrieval NCA 인코더를 더 키우면(임베딩 차원, hidden layer 확대) 개선되는지
확인 -- 단, 오늘 랜덤 carve-out으로 레시피를 잘못 판단해 리더보드 -19 회귀를
겪은 직후라 이번엔 처음부터 시간 기준 walk-forward(train<=2023 -> eval=2024)로
검증한다. 987 레시피(원재료 제거, 실측 검증된 안전한 쪽)로 고정.

비교: 현재 크기(embed=32, hidden=[256,128], v25/v26과 동일) vs 확대판(embed=64,
hidden=[512,256,128]).
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
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/retrieval_bigger_walkforward.json"

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BATCH_SIZE = 1024
NCA_EPOCHS = 20
NCA_LR = 1e-3
NCA_WEIGHT_DECAY = 1e-5
NCA_TEMP_INIT = 0.1
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000

VARIANTS = {
    "current_v26": dict(embed_dims={
        "pitcher_id": 24, "batter_id": 24, "pitcher_team_id": 6, "batter_team_id": 6,
        "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2, "base_state": 4,
    }, hidden=[256, 128], embed_out=32),
    "bigger": dict(embed_dims={
        "pitcher_id": 48, "batter_id": 48, "pitcher_team_id": 12, "batter_team_id": 12,
        "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2, "base_state": 4,
    }, hidden=[512, 256, 128], embed_out=64),
}


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


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


class RowEncoder(nn.Module):
    def __init__(self, cat_cardinalities, embed_dims, hidden, embed_out, n_numeric, dropout=0.2):
        super().__init__()
        self.embeds = nn.ModuleDict()
        embed_out_dim = 0
        for col, card in cat_cardinalities.items():
            dim = embed_dims[col]
            self.embeds[col] = nn.Embedding(card + 1, dim)
            embed_out_dim += dim
        in_dim = embed_out_dim + n_numeric
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, embed_out)]
        self.mlp = nn.Sequential(*layers)
        self.log_temp = nn.Parameter(torch.tensor(np.log(NCA_TEMP_INIT), dtype=torch.float32))

    def encode(self, cat_tensors, x_num):
        parts = [self.embeds[col](cat_tensors[col]) for col in self.embeds]
        parts.append(x_num)
        x = torch.cat(parts, dim=1)
        z = self.mlp(x)
        return nn.functional.normalize(z, dim=1)


def build_nn_cat_mappings(df):
    mappings, cardinalities = {}, {}
    for col in ALL_CAT_FOR_NN:
        uniq = sorted(df[col].astype(str).unique())
        mappings[col] = {v: i for i, v in enumerate(uniq)}
        cardinalities[col] = len(uniq)
    return mappings, cardinalities


def encode_cats(df, mappings):
    out = {}
    for col, m in mappings.items():
        oov = len(m)
        out[col] = df[col].astype(str).map(m).fillna(oov).astype(int).values
    return out


def prep_numeric_fit(df, numeric_cols):
    X = df[numeric_cols].copy()
    isna_flags = X.isna().astype(np.float32)
    isna_flags.columns = [f"{c}__isna" for c in numeric_cols]
    medians = X.median()
    X = X.fillna(medians)
    X_all = pd.concat([X, isna_flags], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all.values.astype(np.float32))
    return X_scaled.astype(np.float32), medians, scaler


def prep_numeric_transform(df, numeric_cols, medians, scaler):
    X = df[numeric_cols].copy()
    isna_flags = X.isna().astype(np.float32)
    isna_flags.columns = [f"{c}__isna" for c in numeric_cols]
    X = X.fillna(medians)
    X_all = pd.concat([X, isna_flags], axis=1)
    return scaler.transform(X_all.values.astype(np.float32)).astype(np.float32)


def make_batches(n, batch_size, shuffle=True, seed=0):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        batch = idx[i:i + batch_size]
        if len(batch) < 8:
            continue
        yield batch


def train_encoder(cat_train, x_num_train, y_train, cardinalities, n_numeric, embed_dims, hidden, embed_out, device):
    model = RowEncoder(cardinalities, embed_dims, hidden, embed_out, n_numeric).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=NCA_LR, weight_decay=NCA_WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    cat_train_t = {c: torch.tensor(v, dtype=torch.long, device=device) for c, v in cat_train.items()}
    x_num_train_t = torch.tensor(x_num_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.values.astype(np.float32), device=device)

    n = len(y_train)
    for epoch in range(NCA_EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch_idx in make_batches(n, BATCH_SIZE, shuffle=True, seed=SEED + epoch):
            cat_batch = {c: v[batch_idx] for c, v in cat_train_t.items()}
            x_num_batch = x_num_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            opt.zero_grad()
            z = model.encode(cat_batch, x_num_batch)
            sim = z @ z.T
            temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)
            sim = sim / temp
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=device)
            sim = sim.masked_fill(eye, float("-inf"))
            weights = torch.softmax(sim, dim=1)
            pred = (weights @ y_batch).clamp(1e-6, 1 - 1e-6)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"      epoch {epoch+1}/{NCA_EPOCHS} loss={epoch_loss/max(n_batches,1):.5f}")
    return model


@torch.no_grad()
def compute_embeddings(model, device, cat_dict, x_num, chunk=REFERENCE_CHUNK):
    model.eval()
    n = x_num.shape[0]
    cat_t = {c: torch.tensor(v, dtype=torch.long, device=device) for c, v in cat_dict.items()}
    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    zs = []
    for i in range(0, n, chunk):
        cat_chunk = {c: v[i:i + chunk] for c, v in cat_t.items()}
        z = model.encode(cat_chunk, x_num_t[i:i + chunk])
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


@torch.no_grad()
def retrieve_predict(model, device, cat_query, x_num_query, z_ref, y_ref):
    model.eval()
    temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0).to(device)
    z_ref = z_ref.to(device)
    y_ref = torch.tensor(y_ref, dtype=torch.float32, device=device)

    cat_query_t = {c: torch.tensor(v, dtype=torch.long, device=device) for c, v in cat_query.items()}
    x_num_query_t = torch.tensor(x_num_query, dtype=torch.float32, device=device)
    n_q = x_num_query_t.shape[0]
    preds = np.zeros(n_q, dtype=np.float64)

    for qi in range(0, n_q, QUERY_CHUNK):
        cat_q_chunk = {c: v[qi:qi + QUERY_CHUNK] for c, v in cat_query_t.items()}
        x_q_chunk = x_num_query_t[qi:qi + QUERY_CHUNK]
        z_q = model.encode(cat_q_chunk, x_q_chunk)

        running_max = torch.full((z_q.shape[0],), float("-inf"), device=device)
        running_numer = torch.zeros(z_q.shape[0], device=device)
        running_denom = torch.zeros(z_q.shape[0], device=device)
        for i in range(0, z_ref.shape[0], REFERENCE_CHUNK):
            z_ref_c = z_ref[i:i + REFERENCE_CHUNK]
            y_ref_c = y_ref[i:i + REFERENCE_CHUNK]
            sim = (z_q @ z_ref_c.T) / temp
            chunk_max = sim.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)
            scale_old = torch.exp(running_max - new_max)
            scale_old = torch.where(torch.isfinite(scale_old), scale_old, torch.zeros_like(scale_old))
            w_chunk = torch.exp(sim - new_max.unsqueeze(1))
            running_numer = running_numer * scale_old + (w_chunk * y_ref_c.unsqueeze(0)).sum(dim=1)
            running_denom = running_denom * scale_old + w_chunk.sum(dim=1)
            running_max = new_max
        pred_chunk = (running_numer / running_denom).clamp(1e-6, 1 - 1e-6)
        preds[qi:qi + QUERY_CHUNK] = pred_chunk.cpu().numpy()

    return preds


def run_variant(name, cfg, train_df, eval_df, device):
    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)
    train_sub = train_sub.reset_index(drop=True)
    calib = calib.reset_index(drop=True)

    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub)
    numeric_cols = [c for c in train_sub.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]

    cat_train = encode_cats(train_sub, nn_cat_mappings)
    cat_calib = encode_cats(calib, nn_cat_mappings)
    cat_eval = encode_cats(eval_df, nn_cat_mappings)
    x_num_train, medians, scaler = prep_numeric_fit(train_sub, numeric_cols)
    x_num_calib = prep_numeric_transform(calib, numeric_cols, medians, scaler)
    x_num_eval = prep_numeric_transform(eval_df, numeric_cols, medians, scaler)

    y_calib = calib[TARGET_COL].values
    y_eval = eval_df[TARGET_COL].values

    t0 = time.time()
    log(f"    [{name}] encoder 학습 시작 (train_sub={len(train_sub)})")
    model = train_encoder(cat_train, x_num_train, train_sub[TARGET_COL], cardinalities,
                           x_num_train.shape[1], cfg["embed_dims"], cfg["hidden"], cfg["embed_out"], device)
    elapsed_train = time.time() - t0

    t0 = time.time()
    z_ref = compute_embeddings(model, device, cat_train, x_num_train)
    y_ref = train_sub[TARGET_COL].values
    calib_raw = retrieve_predict(model, device, cat_calib, x_num_calib, z_ref, y_ref)
    eval_raw = retrieve_predict(model, device, cat_eval, x_num_eval, z_ref, y_ref)
    elapsed_infer = time.time() - t0

    a, b = fit_platt(calib_raw, y_calib)
    eval_calib_pred = apply_platt(eval_raw, a, b)

    result = {
        "variant": name,
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib_pred, y_eval),
        "train_sec": elapsed_train, "infer_sec": elapsed_infer,
    }
    log(f"    [{name}] auc={result['auc']:.4f} bss_calib={result['bss_calibrated']:.2f} "
        f"(train {elapsed_train:.0f}s, infer {elapsed_infer:.0f}s)")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df = add_risk_score_drop_ingredients(df)
    log(f"shape={df.shape}")

    fold_specs = {
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    results = {}
    for fold_name, (train_df, eval_df) in fold_specs.items():
        log(f"\n=== {fold_name} (train={len(train_df)} eval={len(eval_df)}) ===")
        results[fold_name] = {}
        for name, cfg in VARIANTS.items():
            results[fold_name][name] = run_variant(name, cfg, train_df, eval_df, device)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("\n=== SUMMARY ===")
    for fn, variants in results.items():
        for name, r in variants.items():
            log(f"  {fn}/{name}: auc={r['auc']:.4f} bss_calib={r['bss_calibrated']:.2f}")
    log(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
