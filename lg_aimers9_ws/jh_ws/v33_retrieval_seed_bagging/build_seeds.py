"""retrieval 다중 시드 확장 -- v26(CatBoost:retrieval=0.7:0.3, 1023점)에 이 프로젝트
유일의 검증된 성공 메커니즘(시드 배깅, 새 정보 무추가+순수 분산감소)을 적용.
8/21 CatBoost 단독 배깅(3시드 974.9->979, 6시드 979->982) 때와 같은 시드값
[42,7,123,1,99,777] 사용, 그때처럼 seed=42는 이미 있는 v26 모델(CatBoost+
encoder+참조임베딩)을 재학습 없이 재사용하고 새 시드 5개(7/123/1/99/777)만
학습.

버그 수정(중요): v25/v26의 train_encoder()는 배치 셔플 시드로 전역 상수 SEED
(=42 고정)를 그대로 썼음(`make_batches(..., seed=SEED+epoch)`) -- 시드
배깅의 취지(각 시드마다 진짜 다른 랜덤성)를 살리려면 배치 순서도 시드별로
달라야 하므로, 이번엔 seed 파라미터를 train_encoder에 명시적으로 전달해서
가중치 초기화(torch.manual_seed)와 배치 셔플 둘 다 해당 시드를 쓰도록 수정.

시드별 체크포인트: CatBoost/encoder/참조임베딩을 시드 완료마다 저장, 중간에
끊겨도 이어서 진행 가능.
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

DATA_SPLIT_SEED = 42  # calib carve-out 분할은 항상 고정(시드 배깅 대상 아님)

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v33_retrieval_seed_bagging\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v33_retrieval_seed_bagging\output"
CKPT_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\31c37839-97fb-4f8c-b428-a9f2da4f79c5\scratchpad\v33_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

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
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000

ALL_SEEDS = [42, 7, 123, 1, 99, 777]
NEW_SEEDS = [7, 123, 1, 99, 777]  # 42는 v26 재사용(재학습 없음)
W_CATBOOST = 0.7  # v26과 동일 블렌드 가중치


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


def train_encoder(cat_train, x_num_train, y_train, cardinalities, n_numeric, seed):
    """seed 파라미터로 가중치 초기화 + 배치 셔플 둘 다 제어(버그 수정, 위 docstring 참고)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        for batch_idx in make_batches(n, BATCH_SIZE, shuffle=True, seed=seed + epoch):
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
        print(f"      epoch {epoch+1}/{NCA_EPOCHS} train_loss={epoch_loss/max(n_batches,1):.5f}", flush=True)

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
    """v25/v26과 동일한 메모리 안전 온라인 스트리밍 버전."""
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
    print("Load train data + control_risk_score(원재료 제거)...", flush=True)
    df = load_data()
    df = add_risk_score_drop_ingredients(df)
    print(f" shape={df.shape}", flush=True)

    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        df, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[TARGET_COL]
    print(f" train_sub={train_sub_df.shape}  calib={calib_df.shape} (v26과 동일 split)", flush=True)

    cb_id_mappings = build_catboost_id_mappings(train_sub_df)
    X_train_cb = build_catboost_features(train_sub_df, cb_id_mappings)
    y_train_cb = train_sub_df[TARGET_COL]
    X_calib_cb = build_catboost_features(calib_df, cb_id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]

    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]
    cat_train_nn = encode_cats(train_sub_df, nn_cat_mappings)
    cat_calib_nn = encode_cats(calib_df, nn_cat_mappings)
    x_num_train, medians, scaler = prep_numeric_fit(train_sub_df, numeric_cols)
    x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians, scaler)

    progress_path = os.path.join(CKPT_DIR, "seed_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
    else:
        progress = {"done_seeds": []}

    for seed in NEW_SEEDS:
        if seed in progress["done_seeds"]:
            print(f"\n=== seed={seed}: 이미 완료됨, 스킵 ===", flush=True)
            continue
        print(f"\n########## seed={seed} 학습 ##########", flush=True)

        t0 = time.time()
        cb_model = CatBoostClassifier(
            iterations=ITERATIONS, loss_function="Logloss", random_seed=seed,
            cat_features=cat_idx, verbose=200, **BEST_PARAMS,
        )
        cb_model.fit(X_train_cb, y_train_cb)
        print(f" CatBoost(seed={seed}) 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
        cb_model.save_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))

        t0 = time.time()
        encoder, device = train_encoder(
            cat_train_nn, x_num_train, train_sub_df[TARGET_COL], cardinalities, x_num_train.shape[1], seed,
        )
        print(f" encoder(seed={seed}) 학습 완료 ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        z_ref = compute_embeddings(encoder, device, cat_train_nn, x_num_train)
        print(f" 참조 임베딩 계산 완료 ({time.time()-t0:.1f}s)", flush=True)

        torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, f"retrieval_encoder_seed{seed}.pt"))
        np.save(os.path.join(MODEL_DIR, f"reference_embeddings_seed{seed}.npy"), z_ref.numpy().astype(np.float32))

        progress["done_seeds"].append(seed)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f)
        print(f" 체크포인트 저장 완료 (seed={seed})", flush=True)

    print("\n########## 전체 시드 완료. calib 예측 계산 + 블렌드 + calibration ##########", flush=True)
    y_ref_full = train_sub_df[TARGET_COL].values

    cb_calib_raws, retr_calib_raws = [], []
    for seed in ALL_SEEDS:
        cb_model = CatBoostClassifier()
        cb_model.load_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cb_calib_raws.append(cb_model.predict_proba(X_calib_cb)[:, 1])

        encoder = RowEncoder(cardinalities, x_num_train.shape[1])
        encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"retrieval_encoder_seed{seed}.pt"), map_location="cpu"))
        encoder.eval()
        z_ref = torch.tensor(np.load(os.path.join(MODEL_DIR, f"reference_embeddings_seed{seed}.npy")), dtype=torch.float32)
        device = torch.device("cpu")
        retr_calib_raws.append(retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref_full))
        print(f" seed={seed} calib 예측 완료", flush=True)

    cb_avg_raw = np.mean(cb_calib_raws, axis=0)
    retr_avg_raw = np.mean(retr_calib_raws, axis=0)
    blend_raw = W_CATBOOST * cb_avg_raw + (1 - W_CATBOOST) * retr_avg_raw
    a_final, b_final = fit_platt_scaling(blend_raw, y_calib)
    blend_calib_pred = apply_platt_scaling(blend_raw, a_final, b_final)

    metrics = {
        "seeds": ALL_SEEDS,
        "w_catboost": W_CATBOOST,
        "carveout_bss_cb_avg_only": bss_score(cb_avg_raw, y_calib),
        "carveout_bss_retr_avg_only": bss_score(retr_avg_raw, y_calib),
        "carveout_bss_blend_raw": bss_score(blend_raw, y_calib),
        "carveout_bss_blend_calibrated": bss_score(blend_calib_pred, y_calib),
    }
    print(f"\n carve-out BSS: cb_avg={metrics['carveout_bss_cb_avg_only']:.2f}  "
          f"retr_avg={metrics['carveout_bss_retr_avg_only']:.2f}  "
          f"blend_calibrated={metrics['carveout_bss_blend_calibrated']:.2f}", flush=True)
    print(f" (참고: v26 1시드 동일 방식 carve-out은 2082.78)", flush=True)

    with open(os.path.join(OUT_DIR, "metrics_v33.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

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
            "seeds": ALL_SEEDS,
            "blend_weight_catboost": W_CATBOOST,
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved feature_meta.json to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
