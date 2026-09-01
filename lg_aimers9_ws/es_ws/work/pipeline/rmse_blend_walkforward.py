"""RMSE loss가 CatBoost 단독으로는 fold0(+14.96)/fold2(+8.65) 둘 다 Logloss보다 나았음
(rmse_loss_check.py). 실제 production 레시피는 CatBoost:retrieval=0.7:0.3 블렌드이므로,
RMSE로 바꾼 CatBoost가 블렌드 안에서도 이득이 유지되는지 fold0+fold2 양쪽에서 확인.
v13_walkforward_check.py의 retrieval 인코더 학습 로직을 재사용."""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/rmse_blend_walkforward.json"

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

CB_SEED = 42
CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
CAT_ITERATIONS = 2000
W_CATBOOST = 0.7

EMBED_DIMS = {
    "pitcher_id": 24, "batter_id": 24, "pitcher_team_id": 6, "batter_team_id": 6,
    "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2, "base_state": 4,
}
ENCODER_HIDDEN = [256, 128]
EMBED_OUT_DIM = 32
BATCH_SIZE = 1024
NCA_EPOCHS = 20
NCA_LR = 1e-3
NCA_WEIGHT_DECAY = 1e-5
NCA_TEMP_INIT = 0.1
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000

FOLDS = [
    {"name": "fold0", "train_max": 2021, "eval": 2022},
    {"name": "fold2", "train_max": 2023, "eval": 2024},
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


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


def build_id_mappings(df):
    return {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in RAW_ID_COLS}


def build_catboost_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str)
    return X


class RowEncoder(nn.Module):
    def __init__(self, cat_cardinalities, n_numeric):
        super().__init__()
        self.embeds = nn.ModuleDict()
        embed_out = 0
        for col, card in cat_cardinalities.items():
            dim = EMBED_DIMS[col]
            self.embeds[col] = nn.Embedding(card + 1, dim)
            embed_out += dim
        in_dim = embed_out + n_numeric
        layers = []
        prev = in_dim
        for h in ENCODER_HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, EMBED_OUT_DIM)]
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


def train_encoder(cat_train, x_num_train, y_train, cardinalities, n_numeric, device):
    model = RowEncoder(cardinalities, n_numeric).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=NCA_LR, weight_decay=NCA_WEIGHT_DECAY)
    loss_fn = nn.BCELoss()
    cat_train_t = {c: torch.tensor(v, dtype=torch.long, device=device) for c, v in cat_train.items()}
    x_num_train_t = torch.tensor(x_num_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.values.astype(np.float32), device=device)
    n = len(y_train)
    for epoch in range(NCA_EPOCHS):
        model.train()
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


def calib_and_score(raw_calib, y_calib, raw_eval, y_eval):
    a, b = fit_platt(raw_calib, y_calib)
    eval_pred = apply_platt(raw_eval, a, b)
    return bss_score(eval_pred, y_eval), roc_auc_score(y_eval, raw_eval)


def run_fold(df, fold, device):
    train_max, eval_season, name = fold["train_max"], fold["eval"], fold["name"]
    train_df = df[df["season"] <= train_max]
    eval_df = df[df["season"] == eval_season]
    log(f"\n### {name}: train(<={train_max})={len(train_df)} eval({eval_season})={len(eval_df)}")

    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)
    train_sub = train_sub.reset_index(drop=True)
    calib = calib.reset_index(drop=True)
    y_calib = calib[TARGET_COL].values
    y_eval = eval_df[TARGET_COL].values

    id_mappings = build_id_mappings(train_sub)
    X_train_cb = build_catboost_features(train_sub, id_mappings)
    X_calib_cb = build_catboost_features(calib, id_mappings)
    X_eval_cb = build_catboost_features(eval_df, id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]
    y_train_cb = train_sub[TARGET_COL]

    log("  --- CatBoost Logloss ---")
    t0 = time.time()
    m_ll = CatBoostClassifier(iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=CB_SEED,
                               cat_features=cat_idx, verbose=False, task_type="GPU", devices="0", **CAT_BEST_PARAMS)
    m_ll.fit(X_train_cb, y_train_cb)
    ll_calib = m_ll.predict_proba(X_calib_cb)[:, 1]
    ll_eval = m_ll.predict_proba(X_eval_cb)[:, 1]
    log(f"    done ({time.time()-t0:.0f}s)")

    log("  --- CatBoost RMSE ---")
    t0 = time.time()
    m_rmse = CatBoostRegressor(iterations=CAT_ITERATIONS, loss_function="RMSE", random_seed=CB_SEED,
                                cat_features=cat_idx, verbose=False, task_type="GPU", devices="0", **CAT_BEST_PARAMS)
    m_rmse.fit(X_train_cb, y_train_cb)
    rmse_calib = m_rmse.predict(X_calib_cb)
    rmse_eval = m_rmse.predict(X_eval_cb)
    log(f"    done ({time.time()-t0:.0f}s)")

    log("  --- retrieval encoder (fold-specific) ---")
    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub)
    numeric_cols = [c for c in train_sub.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]
    cat_train_nn = encode_cats(train_sub, nn_cat_mappings)
    cat_calib_nn = encode_cats(calib, nn_cat_mappings)
    cat_eval_nn = encode_cats(eval_df, nn_cat_mappings)
    x_num_train, medians, scaler = prep_numeric_fit(train_sub, numeric_cols)
    x_num_calib = prep_numeric_transform(calib, numeric_cols, medians, scaler)
    x_num_eval = prep_numeric_transform(eval_df, numeric_cols, medians, scaler)

    t0 = time.time()
    encoder = train_encoder(cat_train_nn, x_num_train, train_sub[TARGET_COL], cardinalities, x_num_train.shape[1], device)
    z_ref = compute_embeddings(encoder, device, cat_train_nn, x_num_train)
    y_ref = train_sub[TARGET_COL].values
    nca_calib = retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
    nca_eval = retrieve_predict(encoder, device, cat_eval_nn, x_num_eval, z_ref, y_ref)
    log(f"    done ({time.time()-t0:.0f}s)")

    log("  --- blend: (Logloss+retrieval, 0.7:0.3) vs (RMSE+retrieval, 0.7:0.3) ---")
    blend_ll_calib = W_CATBOOST * ll_calib + (1 - W_CATBOOST) * nca_calib
    blend_ll_eval = W_CATBOOST * ll_eval + (1 - W_CATBOOST) * nca_eval
    bss_ll, auc_ll = calib_and_score(blend_ll_calib, y_calib, blend_ll_eval, y_eval)

    blend_rmse_calib = W_CATBOOST * rmse_calib + (1 - W_CATBOOST) * nca_calib
    blend_rmse_eval = W_CATBOOST * rmse_eval + (1 - W_CATBOOST) * nca_eval
    bss_rmse, auc_rmse = calib_and_score(blend_rmse_calib, y_calib, blend_rmse_eval, y_eval)

    log(f"  blend(Logloss+retrieval): auc={auc_ll:.4f} bss_calib={bss_ll:.2f}")
    log(f"  blend(RMSE+retrieval):    auc={auc_rmse:.4f} bss_calib={bss_rmse:.2f}  delta={bss_rmse-bss_ll:+.2f}")

    return {
        "fold": name, "train_max": train_max, "eval_season": eval_season,
        "blend_logloss": {"auc": auc_ll, "bss_calib": bss_ll},
        "blend_rmse": {"auc": auc_rmse, "bss_calib": bss_rmse, "delta_vs_logloss_blend": bss_rmse - bss_ll},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df = add_risk_score_drop_ingredients(df)
    log(f"shape={df.shape}")

    results = [run_fold(df, fold, device) for fold in FOLDS]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nSaved: {OUT_PATH}")
    log("\n=== SUMMARY ===")
    for r in results:
        log(f"{r['fold']}: blend_delta={r['blend_rmse']['delta_vs_logloss_blend']:+.2f}")


if __name__ == "__main__":
    main()
