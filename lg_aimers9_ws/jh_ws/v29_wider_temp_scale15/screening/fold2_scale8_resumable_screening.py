"""fold2_2024, scale=8.0 전용, 청크 단위 체크포인트/진행률 로깅 버전.
run_fold의 retrieve_predict 부분을 query chunk마다 진행률 출력+체크포인트 저장하도록
재작성 -- 도구 타임아웃/환경 kill로 중단돼도 이어서 재개 가능.
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

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
CACHE_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\24e954c1-d480-4a70-9d75-dbfa46ca88d3\scratchpad"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "wider_temp_fold2_resumable_results.json")
CKPT_CALIB = os.path.join(OUT_DIR, "ckpt_fold2_calib_preds.npy")
CKPT_CALIB_IDX = os.path.join(OUT_DIR, "ckpt_fold2_calib_idx.txt")
CKPT_EVAL = os.path.join(OUT_DIR, "ckpt_fold2_eval_preds.npy")
CKPT_EVAL_IDX = os.path.join(OUT_DIR, "ckpt_fold2_eval_idx.txt")

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS

EMBED_DIMS = {
    "pitcher_id": 24, "batter_id": 24,
    "pitcher_team_id": 6, "batter_team_id": 6,
    "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2,
    "base_state": 4,
}
ENCODER_HIDDEN = [256, 128]
EMBED_OUT_DIM = 32
DROPOUT = 0.2
NCA_TEMP_INIT = 0.1
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 2000
SCALE = 8.0
FOLD_NAME = "fold2_2024"
CHECKPOINT_EVERY = 3  # query chunks


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
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df


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
        z = nn.functional.normalize(z, dim=1)
        return z


def build_cat_mappings(train_sub_df):
    mappings, cardinalities = {}, {}
    for col in ALL_CAT_FOR_NN:
        uniq = sorted(train_sub_df[col].astype(str).unique())
        mappings[col] = {v: i for i, v in enumerate(uniq)}
        cardinalities[col] = len(uniq)
    return mappings, cardinalities


def encode_cats(df, mappings):
    out = {}
    for col, m in mappings.items():
        oov = len(m)
        out[col] = df[col].astype(str).map(m).fillna(oov).astype(int).values
    return out


def prep_numeric(df, numeric_cols, medians=None, scaler=None, fit=False):
    X = df[numeric_cols].copy()
    isna_flags = X.isna().astype(np.float32)
    isna_flags.columns = [f"{c}__isna" for c in numeric_cols]
    if fit:
        medians = X.median()
    X = X.fillna(medians)
    X_all = pd.concat([X, isna_flags], axis=1)
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all.values.astype(np.float32))
    else:
        X_scaled = scaler.transform(X_all.values.astype(np.float32))
    return X_scaled.astype(np.float32), medians, scaler


@torch.no_grad()
def encode_reference_chunks(model, cat_ref, x_num_ref, y_ref):
    cat_ref_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_ref.items()}
    x_num_ref_t = torch.tensor(x_num_ref, dtype=torch.float32)
    y_ref_t = torch.tensor(y_ref.values.astype(np.float32))
    n_ref = x_num_ref_t.shape[0]
    z_chunks, y_chunks = [], []
    for i in range(0, n_ref, REFERENCE_CHUNK):
        cat_chunk = {c: v[i:i + REFERENCE_CHUNK] for c, v in cat_ref_t.items()}
        x_chunk = x_num_ref_t[i:i + REFERENCE_CHUNK]
        z_chunks.append(model.encode(cat_chunk, x_chunk))
        y_chunks.append(y_ref_t[i:i + REFERENCE_CHUNK])
    return z_chunks, y_chunks


@torch.no_grad()
def retrieve_predict_resumable(model, cat_query, x_num_query, z_ref_chunks, y_ref_chunks,
                                scale, ckpt_preds_path, ckpt_idx_path, tag):
    model.eval()
    base_temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)
    temp_s = base_temp * scale
    cat_query_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_query.items()}
    x_num_query_t = torch.tensor(x_num_query, dtype=torch.float32)
    n_q = x_num_query_t.shape[0]
    n_chunks = (n_q + QUERY_CHUNK - 1) // QUERY_CHUNK

    if os.path.exists(ckpt_preds_path) and os.path.exists(ckpt_idx_path):
        preds = np.load(ckpt_preds_path)
        with open(ckpt_idx_path) as f:
            start_chunk = int(f.read().strip())
        print(f"  [{tag}] 체크포인트에서 재개: chunk {start_chunk}/{n_chunks}", flush=True)
    else:
        preds = np.zeros(n_q, dtype=np.float64)
        start_chunk = 0

    t0 = time.time()
    for ci in range(start_chunk, n_chunks):
        qi = ci * QUERY_CHUNK
        cat_q_chunk = {c: v[qi:qi + QUERY_CHUNK] for c, v in cat_query_t.items()}
        x_q_chunk = x_num_query_t[qi:qi + QUERY_CHUNK]
        z_q = model.encode(cat_q_chunk, x_q_chunk)

        n_qc = z_q.shape[0]
        max_scaled = torch.full((n_qc,), float("-inf"))
        raw_sims = []
        for z_ref_c in z_ref_chunks:
            sim = (z_q @ z_ref_c.T) / temp_s
            raw_sims.append(sim)
            max_scaled = torch.maximum(max_scaled, sim.max(dim=1).values)
        numer = torch.zeros(n_qc)
        denom = torch.zeros(n_qc)
        for sim, y_ref_c in zip(raw_sims, y_ref_chunks):
            w = torch.exp(sim - max_scaled.unsqueeze(1))
            numer += (w * y_ref_c.unsqueeze(0)).sum(dim=1)
            denom += w.sum(dim=1)
        pred_chunk = (numer / denom).clamp(1e-6, 1 - 1e-6)
        preds[qi:qi + QUERY_CHUNK] = pred_chunk.numpy()

        if (ci + 1) % CHECKPOINT_EVERY == 0 or (ci + 1) == n_chunks:
            np.save(ckpt_preds_path, preds)
            with open(ckpt_idx_path, "w") as f:
                f.write(str(ci + 1))
            elapsed = time.time() - t0
            rate = (ci + 1 - start_chunk) / elapsed if elapsed > 0 else 0
            remaining = (n_chunks - ci - 1) / rate if rate > 0 else float("nan")
            print(
                f"  [{tag}] chunk {ci+1}/{n_chunks} ({elapsed:.1f}s elapsed, "
                f"~{remaining:.0f}s remaining, {rate:.2f} chunk/s)",
                flush=True,
            )

    return preds


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    train_df = df[df["season"] <= 2023]
    eval_df = df[df["season"] == 2024]

    train_df = add_risk_score(train_df).drop(columns=INGREDIENT_COLS)
    eval_df = add_risk_score(eval_df).drop(columns=INGREDIENT_COLS)

    y_train_full = train_df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    cache_path = os.path.join(CACHE_DIR, f"catboost_cache_{FOLD_NAME}.pkl")
    cb_result, cb_eval_raw, cb_calib_raw = joblib.load(cache_path)
    print(f"  [catboost cached] auc={cb_result['auc']:.4f} bss_calib={cb_result['bss_calibrated']:.2f}", flush=True)
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]

    cat_mappings, cardinalities = build_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]

    cat_train = encode_cats(train_sub_df, cat_mappings)
    cat_calib = encode_cats(calib_df, cat_mappings)
    cat_eval = encode_cats(eval_df, cat_mappings)

    x_num_train, medians, scaler = prep_numeric(train_sub_df, numeric_cols, fit=True)
    x_num_calib, _, _ = prep_numeric(calib_df, numeric_cols, medians=medians, scaler=scaler, fit=False)
    x_num_eval, _, _ = prep_numeric(eval_df, numeric_cols, medians=medians, scaler=scaler, fit=False)

    ckpt_path = os.path.join(OUT_DIR, f"encoder_{FOLD_NAME}.pt")
    print(f"  encoder 체크포인트 로드(재학습 없음): {ckpt_path}", flush=True)
    encoder = RowEncoder(cardinalities, x_num_train.shape[1])
    encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    encoder.eval()
    print(f"  base_temp={torch.exp(encoder.log_temp).item():.4f}", flush=True)

    print(f"  참조집합 인코딩(reference={len(train_sub_df)}행)...", flush=True)
    z_ref_chunks, y_ref_chunks = encode_reference_chunks(encoder, cat_train, x_num_train, train_sub_df[TARGET_COL])
    print(f"  참조집합 인코딩 완료, ref_chunks={len(z_ref_chunks)}개", flush=True)

    print(f"  [scale={SCALE}] calib_preds 계산 시작 (n={len(calib_df)})...", flush=True)
    calib_preds = retrieve_predict_resumable(
        encoder, cat_calib, x_num_calib, z_ref_chunks, y_ref_chunks, SCALE,
        CKPT_CALIB, CKPT_CALIB_IDX, "calib",
    )
    print(f"  [scale={SCALE}] eval_preds 계산 시작 (n={len(eval_df)})...", flush=True)
    eval_preds = retrieve_predict_resumable(
        encoder, cat_eval, x_num_eval, z_ref_chunks, y_ref_chunks, SCALE,
        CKPT_EVAL, CKPT_EVAL_IDX, "eval",
    )

    solo_auc = roc_auc_score(y_eval, eval_preds)
    a, b = fit_platt(calib_preds, y_calib)
    eval_calib_v = apply_platt(eval_preds, a, b)
    solo_bss = bss_score(eval_calib_v, y_eval)

    blend_calib_raw = (cb_calib_raw + calib_preds) / 2
    blend_eval_raw = (cb_eval_raw + eval_preds) / 2
    a_bl, b_bl = fit_platt(blend_calib_raw, y_calib)
    blend_eval_calib = apply_platt(blend_eval_raw, a_bl, b_bl)
    blend_bss = bss_score(blend_eval_calib, y_eval)
    delta = blend_bss - cb_result["bss_calibrated"]

    result = {
        "fold_name": FOLD_NAME,
        "scale": SCALE,
        "catboost_baseline_bss": cb_result["bss_calibrated"],
        "solo_auc": solo_auc,
        "solo_bss": solo_bss,
        "blend_bss": blend_bss,
        "delta_vs_catboost": delta,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print("\n=== RESULT ===", flush=True)
    print(
        f"  solo_auc={solo_auc:.4f} solo_bss={solo_bss:.2f} "
        f"blend_bss={blend_bss:.2f} (delta {delta:+.2f})",
        flush=True,
    )
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
