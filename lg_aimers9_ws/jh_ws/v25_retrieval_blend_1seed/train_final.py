"""987점(v22 drop_ingredients) 레시피 + retrieval 기반(ModernNCA 스타일) 모델을
CatBoost와 블렌드, 단일 시드로 실제 제출 검증.

배경(2026-08-23, retrieval_nca.py 로컬 스크리닝): CatBoost 단독보다는
낮지만(fold0 2082.57 vs 2368.23, fold2 653.38 vs 848.32) 이 세션에서 시도한
NN 계열 4종(entity_embed_nn v1/v2, TabM, retrieval_nca) 중 CatBoost와의
격차가 가장 작았고, 블렌드 손상도 가장 적었음(fold0 -23.86, fold2 -18.47).
사용자가 실제 제출로 검증해보자고 요청.

**서버 추론시간 제약(중요)**: retrieval 추론은 쿼리마다 참조 데이터 전체와
유사도를 계산해야 함(top-K 근사 아님, exact kernel regression). 로컬
스크리닝에서 참조 116만행 기준 추론에만 990초(16.5분)가 걸렸는데, 대회
서버 추론시간 제한이 10분이라 그대로면 타임아웃(런타임 오류, 일일 제출
횟수 차감) 위험이 있음. 안전마진을 위해 참조 데이터를 REFERENCE_SIZE(50만)
로 서브샘플링(인코더 자체는 전체 데이터로 학습, 추론 시 사용할 참조 집합만
축소) -- 대회 서버는 GPU(L4)가 있어 로컬 CPU보다 빠를 가능성이 높지만
안전을 우선함.

CAT_COLS(네이티브)/RAW_ID_COLS(label-encoded)/trackman context/BEST_PARAMS/
calibration carve-out(5%, seed=42)는 v18/v20/v22와 동일. retrieval encoder
구조는 retrieval_nca.py와 동일(NCA loss, embed_dim=32).
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed\output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
ITERATIONS = 2000

EMBED_DIMS = {
    "pitcher_id": 24, "batter_id": 24,
    "pitcher_team_id": 6, "batter_team_id": 6,
    "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2,
    "base_state": 4,
}
ENCODER_HIDDEN = [256, 128]
EMBED_OUT_DIM = 32
DROPOUT = 0.2
BATCH_SIZE = 1024
NCA_EPOCHS = 20
NCA_LR = 1e-3
NCA_WEIGHT_DECAY = 1e-5
NCA_TEMP_INIT = 0.1
REFERENCE_SIZE = None  # None=서브샘플 없이 train_sub 전체를 참조로 사용(사용자 요청, 점수 우선)
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000  # 실제 서버는 GPU 가능성 있어 로컬보다 크게 잡음


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt_scaling(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt_scaling(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


def build_catboost_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


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


def train_encoder(cat_train, x_num_train, y_train, cardinalities, n_numeric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  NCA encoder device={device}", flush=True)
    model = RowEncoder(cardinalities, n_numeric).to(device)
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
        print(f"    epoch {epoch+1}/{NCA_EPOCHS} train_loss={epoch_loss/max(n_batches,1):.5f}", flush=True)

    return model, device


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
    """참조 청크를 한 번만 순회하는 온라인(스트리밍) softmax 누적.
    이전 버전은 모든 ref 청크의 유사도 행렬을 리스트에 쌓아뒀다가 나중에
    합산했는데(쿼리 청크당 최대 수십 GB 메모리 점유, 극심한 스와핑/저속의
    원인이었음), 이 버전은 청크 하나 처리 후 즉시 버리고 누적값(런닝 max/
    numer/denom)만 유지 -- 메모리 사용량이 O(query_chunk x REFERENCE_CHUNK)
    로 고정됨(flash-attention과 동일한 원리)."""
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


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data (전체) + control_risk_score 추가, 원재료3 제거...", flush=True)
    df = load_data()
    df = add_risk_score_drop_ingredients(df)
    print(f" shape={df.shape}", flush=True)

    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        df, test_size=0.05, stratify=y_all, random_state=SEED,
    )
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    print(f" train_sub={train_sub_df.shape}  calib={calib_df.shape}", flush=True)

    # ---------- CatBoost ----------
    print("\n=== CatBoost 학습 ===", flush=True)
    cb_id_mappings = build_catboost_id_mappings(train_sub_df)
    X_train_cb = build_catboost_features(train_sub_df, cb_id_mappings)
    y_train_cb = train_sub_df[TARGET_COL]
    X_calib_cb = build_catboost_features(calib_df, cb_id_mappings)
    y_calib = calib_df[TARGET_COL]
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]

    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED,
        cat_features=cat_idx, verbose=200, **BEST_PARAMS,
    )
    cb_model.fit(X_train_cb, y_train_cb)
    cb_elapsed = time.time() - t0
    print(f" CatBoost 학습 완료 ({cb_elapsed:.1f}s)", flush=True)
    cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

    cb_model_path = os.path.join(MODEL_DIR, "catboost_seed42.cbm")
    cb_model.save_model(cb_model_path)
    print(f" saved: {cb_model_path}", flush=True)

    # ---------- Retrieval encoder ----------
    print("\n=== Retrieval encoder(NCA) 학습 ===", flush=True)
    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]
    print(f" n_numeric={len(numeric_cols)} cardinalities={cardinalities}", flush=True)

    cat_train_nn = encode_cats(train_sub_df, nn_cat_mappings)
    cat_calib_nn = encode_cats(calib_df, nn_cat_mappings)
    x_num_train, medians, scaler = prep_numeric_fit(train_sub_df, numeric_cols)
    x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians, scaler)

    t0 = time.time()
    encoder, device = train_encoder(
        cat_train_nn, x_num_train, train_sub_df[TARGET_COL], cardinalities, x_num_train.shape[1],
    )
    nca_elapsed = time.time() - t0
    print(f" encoder 학습 완료 ({nca_elapsed:.1f}s)", flush=True)

    # 참조 집합: 사용자 요청으로 서브샘플 없이 train_sub 전체 사용
    rng = np.random.RandomState(SEED)
    n_train_sub = len(train_sub_df)
    ref_size = n_train_sub if REFERENCE_SIZE is None else min(REFERENCE_SIZE, n_train_sub)
    ref_idx = rng.choice(n_train_sub, size=ref_size, replace=False)
    print(f" 참조 집합: {ref_size}/{n_train_sub} (서브샘플 없음)", flush=True)

    cat_ref_nn = {c: v[ref_idx] for c, v in cat_train_nn.items()}
    x_num_ref = x_num_train[ref_idx]
    y_ref = train_sub_df[TARGET_COL].values[ref_idx]

    t0 = time.time()
    z_ref = compute_embeddings(encoder, device, cat_ref_nn, x_num_ref)
    print(f" 참조 임베딩 계산 완료 ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    nca_calib_raw = retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
    print(f" calib retrieval 추론 완료 ({time.time()-t0:.1f}s)", flush=True)

    # ---------- 블렌드 + calibration ----------
    print("\n=== 블렌드 + Platt calibration ===", flush=True)
    blend_calib_raw = (cb_calib_raw + nca_calib_raw) / 2
    a_final, b_final = fit_platt_scaling(blend_calib_raw, y_calib)
    blend_calib_pred = apply_platt_scaling(blend_calib_raw, a_final, b_final)

    metrics = {
        "seed": SEED,
        "cb_elapsed_sec": cb_elapsed,
        "nca_elapsed_sec": nca_elapsed,
        "reference_size": ref_size,
        "carveout_bss_cb_only": bss_score(cb_calib_raw, y_calib),
        "carveout_bss_nca_only": bss_score(nca_calib_raw, y_calib),
        "carveout_bss_blend_raw": bss_score(blend_calib_raw, y_calib),
        "carveout_bss_blend_calibrated": bss_score(blend_calib_pred, y_calib),
    }
    print(f" carve-out BSS: cb_only={metrics['carveout_bss_cb_only']:.2f}  "
          f"nca_only={metrics['carveout_bss_nca_only']:.2f}  "
          f"blend_calibrated={metrics['carveout_bss_blend_calibrated']:.2f}", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v25.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---------- 저장 ----------
    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "retrieval_encoder.pt"))
    np.save(os.path.join(MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy().astype(np.float32))
    np.save(os.path.join(MODEL_DIR, "reference_labels.npy"), y_ref.astype(np.float32))
    joblib.dump({"medians": medians, "scaler": scaler}, os.path.join(MODEL_DIR, "numeric_prep.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "catboost": {
                "columns": list(X_train_cb.columns),
                "cat_cols": CATBOOST_CAT_COLS,
                "raw_id_cols": RAW_ID_COLS,
                "id_mappings": cb_id_mappings,
            },
            "retrieval": {
                "numeric_cols": numeric_cols,
                "cat_cols": ALL_CAT_FOR_NN,
                "cat_mappings": nn_cat_mappings,
                "cardinalities": cardinalities,
                "embed_dims": EMBED_DIMS,
                "encoder_hidden": ENCODER_HIDDEN,
                "embed_out_dim": EMBED_OUT_DIM,
            },
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all artifacts to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
