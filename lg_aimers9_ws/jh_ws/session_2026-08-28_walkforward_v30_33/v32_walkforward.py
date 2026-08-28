"""v32(retrieval_score를 CatBoost 피처로 추가)의 fold0_2022/fold2_2024
walk-forward 검증. 프로덕션 v32는 K=3 OOF로 train_sub 전체(140만행)에 대한
retrieval_score를 만들었지만(시간 비용 큼, encoder 3회 학습), 여기서는 시간
단축을 위해 단순화된 hold-out 방식을 쓴다:

  encoder_train(각 fold train 데이터의 45%) -> encoder 학습(리크 없음, 1회만)
  feature_pool(50%) -> CatBoost가 실제로 학습하는 데이터. retrieval_score는
    encoder_train을 참조집합으로 retrieve_predict로 계산(feature_pool 행 자신은
    참조집합에 없으므로 리크 없음).
  calib(5%) -> Platt calibration + eval용, retrieval_score도 동일하게 encoder_train
    참조로 계산.

baseline(retrieval_score 없음)도 반드시 같은 feature_pool로 재학습(공정 비교;
기존 fold 캐시는 train_sub 전체로 학습된 것이라 그대로 쓰면 데이터량이 달라
불공정 비교가 됨).

두 CatBoost(baseline/treatment) 모두 BEST_PARAMS 그대로, iterations만 데이터량
축소를 고려해 유지(2000, early stop 없음 -- 원래도 없었음).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "v32_walkforward_results.json")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BEST_PARAMS = dict(learning_rate=0.01, depth=8, l2_leaf_reg=20.0, bagging_temperature=1.0, random_strength=1.0, border_count=32)
ITERATIONS = 2000

EMBED_DIMS = {
    "pitcher_id": 24, "batter_id": 24, "pitcher_team_id": 6, "batter_team_id": 6,
    "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2, "base_state": 4,
}
ENCODER_HIDDEN = [256, 128]
EMBED_OUT_DIM = 32
DROPOUT = 0.2
BATCH_SIZE = 1024
NCA_EPOCHS = 20
NCA_LR = 1e-3
NCA_WEIGHT_DECAY = 1e-5
NCA_TEMP_INIT = 0.1
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000


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


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    df["control_risk_score_weighted"] = 0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    return df.drop(columns=INGREDIENT_COLS)


def build_catboost_id_mappings(df):
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
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
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


def train_encoder(cat_train, x_num_train, y_train, cardinalities, n_numeric, tag):
    model = RowEncoder(cardinalities, n_numeric)
    opt = torch.optim.Adam(model.parameters(), lr=NCA_LR, weight_decay=NCA_WEIGHT_DECAY)
    loss_fn = nn.BCELoss()
    cat_train_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_train.items()}
    x_num_train_t = torch.tensor(x_num_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values.astype(np.float32))
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
            eye = torch.eye(sim.shape[0], dtype=torch.bool)
            sim = sim.masked_fill(eye, float("-inf"))
            weights = torch.softmax(sim, dim=1)
            pred = (weights @ y_batch).clamp(1e-6, 1 - 1e-6)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"    [{tag}] epoch {epoch+1}/{NCA_EPOCHS} train_loss={epoch_loss/max(n_batches,1):.5f}", flush=True)
    return model


@torch.no_grad()
def compute_embeddings(model, cat_dict, x_num, chunk=REFERENCE_CHUNK):
    model.eval()
    n = x_num.shape[0]
    cat_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_dict.items()}
    x_num_t = torch.tensor(x_num, dtype=torch.float32)
    zs = []
    for i in range(0, n, chunk):
        cat_chunk = {c: v[i:i + chunk] for c, v in cat_t.items()}
        z = model.encode(cat_chunk, x_num_t[i:i + chunk])
        zs.append(z)
    return torch.cat(zs, dim=0)


@torch.no_grad()
def retrieve_predict(model, cat_query, x_num_query, z_ref, y_ref):
    model.eval()
    temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)
    y_ref = torch.tensor(y_ref, dtype=torch.float32)
    cat_query_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_query.items()}
    x_num_query_t = torch.tensor(x_num_query, dtype=torch.float32)
    n_q = x_num_query_t.shape[0]
    preds = np.zeros(n_q, dtype=np.float64)
    for qi in range(0, n_q, QUERY_CHUNK):
        cat_q_chunk = {c: v[qi:qi + QUERY_CHUNK] for c, v in cat_query_t.items()}
        x_q_chunk = x_num_query_t[qi:qi + QUERY_CHUNK]
        z_q = model.encode(cat_q_chunk, x_q_chunk)
        running_max = torch.full((z_q.shape[0],), float("-inf"))
        running_numer = torch.zeros(z_q.shape[0])
        running_denom = torch.zeros(z_q.shape[0])
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
        preds[qi:qi + QUERY_CHUNK] = pred_chunk.numpy()
    return preds


def run_fold(fold_name, train_full_df, eval_df, all_results):
    print(f"\n=== {fold_name} ===", flush=True)
    train_full_df = add_risk_score_drop_ingredients(train_full_df).reset_index(drop=True)
    eval_df = add_risk_score_drop_ingredients(eval_df).reset_index(drop=True)

    y_all = train_full_df[TARGET_COL]
    rest_df, calib_df = train_test_split(train_full_df, test_size=0.05, stratify=y_all, random_state=SEED)
    encoder_train_df, feature_pool_df = train_test_split(
        rest_df, test_size=0.5263, stratify=rest_df[TARGET_COL], random_state=SEED,
    )  # rest=95% -> encoder 45% / feature_pool 50% (0.5263*0.95≈0.50)
    encoder_train_df = encoder_train_df.reset_index(drop=True)
    feature_pool_df = feature_pool_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    print(f" encoder_train={len(encoder_train_df)} feature_pool={len(feature_pool_df)} calib={len(calib_df)} eval={len(eval_df)}", flush=True)

    numeric_cols = [c for c in encoder_train_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]

    # ---- encoder 학습 (encoder_train만, 1회) ----
    nn_cat_mappings, cardinalities = build_nn_cat_mappings(encoder_train_df)
    cat_enc_train = encode_cats(encoder_train_df, nn_cat_mappings)
    x_num_enc_train, medians, scaler = prep_numeric_fit(encoder_train_df, numeric_cols)
    t0 = time.time()
    encoder = train_encoder(cat_enc_train, x_num_enc_train, encoder_train_df[TARGET_COL], cardinalities, x_num_enc_train.shape[1], fold_name)
    print(f" encoder 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
    z_ref = compute_embeddings(encoder, cat_enc_train, x_num_enc_train)
    y_ref = encoder_train_df[TARGET_COL].values

    # ---- feature_pool/calib/eval retrieval_score (encoder_train 참조, 리크 없음) ----
    cat_fp = encode_cats(feature_pool_df, nn_cat_mappings)
    x_num_fp = prep_numeric_transform(feature_pool_df, numeric_cols, medians, scaler)
    cat_calib = encode_cats(calib_df, nn_cat_mappings)
    x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians, scaler)
    cat_eval = encode_cats(eval_df, nn_cat_mappings)
    x_num_eval = prep_numeric_transform(eval_df, numeric_cols, medians, scaler)

    t0 = time.time()
    fp_retrieval = retrieve_predict(encoder, cat_fp, x_num_fp, z_ref, y_ref)
    calib_retrieval = retrieve_predict(encoder, cat_calib, x_num_calib, z_ref, y_ref)
    eval_retrieval = retrieve_predict(encoder, cat_eval, x_num_eval, z_ref, y_ref)
    print(f" retrieval_score 계산 완료 ({time.time()-t0:.1f}s)", flush=True)

    # ---- CatBoost baseline (retrieval_score 없음, feature_pool로 학습) ----
    cb_id_mappings = build_catboost_id_mappings(feature_pool_df)
    X_fp_base = build_catboost_features(feature_pool_df, cb_id_mappings)
    X_calib_base = build_catboost_features(calib_df, cb_id_mappings)
    X_eval_base = build_catboost_features(eval_df, cb_id_mappings)
    y_fp = feature_pool_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]
    cat_idx_base = [X_fp_base.columns.get_loc(c) for c in CATBOOST_CAT_COLS]

    t0 = time.time()
    cb_base = CatBoostClassifier(iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED, cat_features=cat_idx_base, verbose=False, **BEST_PARAMS)
    cb_base.fit(X_fp_base, y_fp)
    print(f" CatBoost baseline 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
    base_calib_raw = cb_base.predict_proba(X_calib_base)[:, 1]
    base_eval_raw = cb_base.predict_proba(X_eval_base)[:, 1]
    a_b, b_b = fit_platt(base_calib_raw, y_calib)
    base_eval_calib = apply_platt(base_eval_raw, a_b, b_b)
    base_bss = bss_score(base_eval_calib, y_eval)
    base_auc = roc_auc_score(y_eval, base_eval_raw)
    print(f" [baseline] eval_auc={base_auc:.4f} eval_bss={base_bss:.2f}", flush=True)

    # ---- CatBoost + retrieval_score (v32) ----
    X_fp_v32 = X_fp_base.copy(); X_fp_v32["retrieval_score"] = fp_retrieval
    X_calib_v32 = X_calib_base.copy(); X_calib_v32["retrieval_score"] = calib_retrieval
    X_eval_v32 = X_eval_base.copy(); X_eval_v32["retrieval_score"] = eval_retrieval
    cat_idx_v32 = [X_fp_v32.columns.get_loc(c) for c in CATBOOST_CAT_COLS]

    t0 = time.time()
    cb_v32 = CatBoostClassifier(iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED, cat_features=cat_idx_v32, verbose=False, **BEST_PARAMS)
    cb_v32.fit(X_fp_v32, y_fp)
    print(f" CatBoost+retrieval 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
    v32_calib_raw = cb_v32.predict_proba(X_calib_v32)[:, 1]
    v32_eval_raw = cb_v32.predict_proba(X_eval_v32)[:, 1]
    a_v, b_v = fit_platt(v32_calib_raw, y_calib)
    v32_eval_calib = apply_platt(v32_eval_raw, a_v, b_v)
    v32_bss = bss_score(v32_eval_calib, y_eval)
    v32_auc = roc_auc_score(y_eval, v32_eval_raw)

    importances = pd.Series(cb_v32.get_feature_importance(), index=X_fp_v32.columns).sort_values(ascending=False)
    retrieval_rank = list(importances.index).index("retrieval_score") + 1

    delta = v32_bss - base_bss
    print(f" [v32] eval_auc={v32_auc:.4f} eval_bss={v32_bss:.2f} (delta {delta:+.2f}) retrieval_importance_rank={retrieval_rank}/{len(importances)}", flush=True)

    all_results[fold_name] = {
        "n_encoder_train": len(encoder_train_df), "n_feature_pool": len(feature_pool_df),
        "n_calib": len(calib_df), "n_eval": len(eval_df),
        "baseline_eval_auc": base_auc, "baseline_eval_bss": base_bss,
        "v32_eval_auc": v32_auc, "v32_eval_bss": v32_bss,
        "delta_v32_vs_baseline": delta,
        "retrieval_importance": float(importances["retrieval_score"]),
        "retrieval_importance_rank": retrieval_rank,
        "n_features": int(len(importances)),
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, encoding="utf-8") as f:
            all_results = json.load(f)

    fold_specs = [
        ("fold2_2024", df[df["season"] <= 2023], df[df["season"] == 2024]),
        ("fold0_2022", df[df["season"] <= 2021], df[df["season"] == 2022]),
    ]
    for fold_name, train_df, eval_df in fold_specs:
        if fold_name in all_results:
            print(f"\n=== {fold_name}: 이미 완료됨, 스킵 ===", flush=True)
            continue
        run_fold(fold_name, train_df, eval_df, all_results)

    print("\n=== SUMMARY ===", flush=True)
    for fold_name, r in all_results.items():
        print(f"  {fold_name}: baseline={r['baseline_eval_bss']:.2f} v32={r['v32_eval_bss']:.2f} delta={r['delta_v32_vs_baseline']:+.2f}", flush=True)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
