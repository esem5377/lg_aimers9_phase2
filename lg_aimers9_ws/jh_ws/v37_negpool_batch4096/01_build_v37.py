"""
v26(CatBoost:retrieval=0.7:0.3, 팀 최고였던 조합)의 CatBoost는 그대로
재사용(재학습 없음)하고, retrieval encoder만 "in-batch negative pool 확대"
버전(BATCH_SIZE 1024->4096, 나머지 용량/구조는 v25/v26과 동일 -- embed_out=32,
encoder_hidden=[256,128], id embed 24/team 6)으로 전체 프로덕션 데이터
(2019~2024, 140만행)에서 처음부터 재학습.

배경(2026-08-28): "cosine similarity로 전환"은 이미 v25부터 encode()가
L2 정규화 후 z@z.T로 cosine similarity를 쓰고 있어 델타가 없는 실험으로
확인(코드 재확인 후 기각). 대신 아직 안 건드린 negative sampling 쪽 --
현재는 배치 내 나머지 전부(1023개)를 소프트 이웃/네거티브로 쓰는데,
사용자가 이 풀을 4배(4096, 네거티브 4095개)로 늘려보는 실험을 선택.

v35(용량 확대: embed_out 32->64, hidden [256,128]->[512,256])는 carve-out
calib_bss가 2082.07로 v26(2082.78) 대비 사실상 그대로였음(용량은 병목이
아니었다는 신호) -- 그래서 이번엔 용량은 그대로 두고 negative pool
크기만 바꿔서 순수하게 이 축의 효과를 분리해서 본다.

**중요**: fold0/fold2 walk-forward 검증(session_2026-08-28_walkforward_v37_negpool/)이
아직 끝나지 않은 상태에서 사용자가 검증 결과를 기다리지 않고 바로 프로덕션
빌드+제출 준비를 요청해 진행. lg_aimers9_walkforward_methodology 규칙
("fold0/fold2 둘 다 양수여야 실제 제출 후보") 우회는 v36 사례처럼 실제
리더보드 하락으로 이어질 수 있음 -- 사용자가 명시적으로 감수하기로 한
리스크. walk-forward 결과가 나중에 나오면 대조해서 기록할 것.

CatBoost/전처리(numeric_prep은 새로 fit, risk_score 레시피는 v22/v25/v26과
동일)는 기존 v26 자산을 최대한 재사용. retrieval 블렌드 가중치(w_cb)는
v26의 0.7을 그대로 쓰지 않고, 새 encoder의 실력 변화를 반영해 전체 이력
calib carve-out(7.4만행, v34/v35와 동일 split)에서 다시 그리드서치(1자유도).
script.py는 v26/v35 것을 변경 없이 그대로 재사용(RowEncoder 아키텍처를
feature_meta.json의 retrieval.embed_dims/encoder_hidden/embed_out_dim에서
읽어오도록 이미 일반화돼 있음 -- BATCH_SIZE는 학습 시에만 쓰이는
하이퍼파라미터라 추론 코드/저장되는 아키텍처에는 영향 없음).

체크포인트: train_encoder가 에폭마다 ckpt_path에 저장하고(model/opt state,
best_state, patience_ctr 포함) 재시작 시 이어서 학습. encoder 학습이 아예
끝나 trained_encoder_path가 있으면 그 이후 단계(참조 임베딩/그리드서치)만
다시 수행 -- 중단돼도 처음부터 다시 하지 않음.
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

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
V37_DIR = os.path.dirname(__file__)
V37_MODEL_DIR = os.path.join(V37_DIR, "model")
V37_OUT_DIR = os.path.join(V37_DIR, "output")

# ---- v25/train_final.py 함수 재사용(finalize_w07.py/01_build_v34.py/01_build_v35.py와 동일 방식) ----
g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "finalize"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)
TARGET_COL, ID_COL, SEED = g["TARGET_COL"], g["ID_COL"], g["SEED"]
CAT_COLS, TEAM_COLS = g["CAT_COLS"], g["TEAM_COLS"]
ALL_CAT_FOR_NN = g["ALL_CAT_FOR_NN"]

# ---- v25/v26과 동일한 표준 용량(용량은 v35에서 이미 효과 없음을 확인) ----
EMBED_DIMS = {
    "pitcher_id": 24, "batter_id": 24,
    "pitcher_team_id": 6, "batter_team_id": 6,
    "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2,
    "base_state": 4,
}
ENCODER_HIDDEN = [256, 128]
EMBED_OUT_DIM = 32
DROPOUT = 0.2
BATCH_SIZE = 4096  # v25/v26/v35는 1024 -- in-batch negative pool 확대 실험(4배)
MAX_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 4
NCA_TEMP_INIT = 0.1
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 2000

WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


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


def make_batches(n, batch_size, shuffle=True, seed=0):
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def train_encoder(cat_train, x_num_train, y_train, cat_val, x_num_val, y_val, cardinalities, n_numeric,
                   ckpt_path=None):
    model = RowEncoder(cardinalities, n_numeric)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터 수: {n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    cat_train_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_train.items()}
    x_num_train_t = torch.tensor(x_num_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values.astype(np.float32))

    cat_val_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_val.items()}
    x_num_val_t = torch.tensor(x_num_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values.astype(np.float32))

    n = len(y_train)
    start_epoch = 0
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    loss_fn = nn.BCELoss()

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"  체크포인트 발견, 재개: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        best_state = ckpt["best_state"]
        patience_ctr = ckpt["patience_ctr"]
        print(f"  epoch {start_epoch}부터 재개 (best_val_loss={best_val_loss:.5f}, patience_ctr={patience_ctr})",
              flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
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
            pred = weights @ y_batch
            pred = pred.clamp(1e-6, 1 - 1e-6)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            ref_size = min(20000, n)
            z_train_sample_idx = np.random.RandomState(SEED).choice(n, size=ref_size, replace=False)
            cat_ref = {c: v[z_train_sample_idx] for c, v in cat_train_t.items()}
            z_ref = model.encode(cat_ref, x_num_train_t[z_train_sample_idx])
            y_ref = y_train_t[z_train_sample_idx]
            temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)

            VAL_QUERY_CHUNK = 5000
            n_val = x_num_val_t.shape[0]
            pred_chunks = []
            for qi in range(0, n_val, VAL_QUERY_CHUNK):
                cat_val_chunk = {c: v[qi:qi + VAL_QUERY_CHUNK] for c, v in cat_val_t.items()}
                x_val_chunk = x_num_val_t[qi:qi + VAL_QUERY_CHUNK]
                z_val_chunk = model.encode(cat_val_chunk, x_val_chunk)
                sim_chunk = (z_val_chunk @ z_ref.T) / temp
                weights_chunk = torch.softmax(sim_chunk, dim=1)
                pred_chunks.append((weights_chunk @ y_ref).clamp(1e-6, 1 - 1e-6))
            pred = torch.cat(pred_chunks, dim=0)
            val_loss = loss_fn(pred, y_val_t).item()

        print(f"    epoch {epoch+1}/{MAX_EPOCHS} train_loss={epoch_loss/max(n_batches,1):.5f} "
              f"val_loss(ref={ref_size})={val_loss:.5f} temp={temp.item():.4f} n_batches={n_batches}", flush=True)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if ckpt_path is not None:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
                "best_val_loss": best_val_loss, "best_state": best_state, "patience_ctr": patience_ctr,
            }, ckpt_path)

        if patience_ctr >= PATIENCE:
            print(f"    early stop at epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return model


@torch.no_grad()
def encode_reference_chunks(model, cat_ref, x_num_ref, y_ref):
    cat_ref_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_ref.items()}
    x_num_ref_t = torch.tensor(x_num_ref, dtype=torch.float32)
    y_ref_t = torch.tensor(y_ref.values.astype(np.float32))
    n_ref = x_num_ref_t.shape[0]
    z_list = []
    for i in range(0, n_ref, REFERENCE_CHUNK):
        cat_chunk = {c: v[i:i + REFERENCE_CHUNK] for c, v in cat_ref_t.items()}
        x_chunk = x_num_ref_t[i:i + REFERENCE_CHUNK]
        z_list.append(model.encode(cat_chunk, x_chunk))
    z_all = torch.cat(z_list, dim=0)
    return z_all, y_ref_t


@torch.no_grad()
def retrieve_predict(model, cat_query, x_num_query, z_ref, y_ref):
    model.eval()
    base_temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)
    cat_query_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_query.items()}
    x_num_query_t = torch.tensor(x_num_query, dtype=torch.float32)
    n_q = x_num_query_t.shape[0]
    preds = np.zeros(n_q, dtype=np.float64)
    for qi in range(0, n_q, QUERY_CHUNK):
        cat_q_chunk = {c: v[qi:qi + QUERY_CHUNK] for c, v in cat_query_t.items()}
        x_q_chunk = x_num_query_t[qi:qi + QUERY_CHUNK]
        z_q = model.encode(cat_q_chunk, x_q_chunk)

        max_scaled = torch.full((z_q.shape[0],), float("-inf"))
        for i in range(0, z_ref.shape[0], REFERENCE_CHUNK):
            z_ref_c = z_ref[i:i + REFERENCE_CHUNK]
            sim = (z_q @ z_ref_c.T) / base_temp
            max_scaled = torch.maximum(max_scaled, sim.max(dim=1).values)

        numer = torch.zeros(z_q.shape[0])
        denom = torch.zeros(z_q.shape[0])
        for i in range(0, z_ref.shape[0], REFERENCE_CHUNK):
            z_ref_c = z_ref[i:i + REFERENCE_CHUNK]
            y_ref_c = y_ref[i:i + REFERENCE_CHUNK]
            sim = (z_q @ z_ref_c.T) / base_temp
            w = torch.exp(sim - max_scaled.unsqueeze(1))
            numer += (w * y_ref_c.unsqueeze(0)).sum(dim=1)
            denom += w.sum(dim=1)

        pred_chunk = (numer / denom).clamp(1e-6, 1 - 1e-6)
        preds[qi:qi + QUERY_CHUNK] = pred_chunk.numpy()
    return preds


def main():
    print("=== v37: CatBoost(v26 재사용) + retrieval(negative pool 4배 확대, 전체 재학습) ===", flush=True)
    print("Load data(전체 이력 2019~2024) + split(재현, v26/v34/v35와 동일 split)...", flush=True)
    df = g["load_data"]()
    df = g["add_risk_score_drop_ingredients"](df)
    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[TARGET_COL].to_numpy(dtype=float)
    print(f" train_sub={len(train_sub_df)} calib={len(calib_df)}", flush=True)

    print("\nCatBoost raw 예측(calib, v26 모델 재사용, 재학습 없음)...", flush=True)
    cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
    X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
    cb_model = CatBoostClassifier()
    cb_model.load_model(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"))
    cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

    print("\nnegative pool 확대 retrieval encoder 학습(BATCH_SIZE=4096, 전체 프로덕션 데이터, 처음부터)...", flush=True)
    cat_mappings, cardinalities = build_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]
    print(f" n_numeric={len(numeric_cols)} cardinalities={cardinalities}", flush=True)

    cat_train = encode_cats(train_sub_df, cat_mappings)
    cat_calib = encode_cats(calib_df, cat_mappings)
    x_num_train, medians, scaler = prep_numeric(train_sub_df, numeric_cols, fit=True)
    x_num_calib, _, _ = prep_numeric(calib_df, numeric_cols, medians=medians, scaler=scaler, fit=False)

    ckpt_path = os.path.join(V37_OUT_DIR, "encoder_train_checkpoint.pt")
    trained_encoder_path = os.path.join(V37_OUT_DIR, "encoder_trained_final.pt")
    os.makedirs(V37_OUT_DIR, exist_ok=True)
    t0 = time.time()
    if os.path.exists(trained_encoder_path):
        print(f" 학습 완료된 encoder 발견, 재학습 건너뜀: {trained_encoder_path}", flush=True)
        encoder = RowEncoder(cardinalities, x_num_train.shape[1])
        encoder.load_state_dict(torch.load(trained_encoder_path, map_location="cpu"))
        encoder.eval()
    else:
        encoder = train_encoder(
            cat_train, x_num_train, train_sub_df[TARGET_COL],
            cat_calib, x_num_calib, calib_df[TARGET_COL],
            cardinalities, x_num_train.shape[1],
            ckpt_path=ckpt_path,
        )
        torch.save(encoder.state_dict(), trained_encoder_path)
    train_elapsed = time.time() - t0
    print(f" encoder 학습 완료 ({train_elapsed:.1f}s)", flush=True)

    t0 = time.time()
    z_ref, y_ref = encode_reference_chunks(encoder, cat_train, x_num_train, train_sub_df[TARGET_COL])
    print(f" 참조집합 인코딩 완료(reference={len(train_sub_df)}행, {time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    nca_calib_raw = retrieve_predict(encoder, cat_calib, x_num_calib, z_ref, y_ref)
    print(f" retrieval calib 예측 완료({time.time()-t0:.1f}s)", flush=True)

    print("\n저자유도 그리드서치(CatBoost:retrieval 가중치, raw 가중합->calibration 1회)...", flush=True)
    best = {"calib_bss": -1}
    grid_log = []
    for w in WEIGHT_GRID:
        blend_raw = w * cb_calib_raw + (1 - w) * nca_calib_raw
        a, b = fit_platt(blend_raw, y_calib)
        s = bss_score(apply_platt(blend_raw, a, b), y_calib)
        grid_log.append({"w_catboost": float(w), "calib_bss": float(s)})
        if s > best["calib_bss"]:
            best = {"w_catboost": float(w), "calib_bss": float(s), "a": a, "b": b}
    print(f" best_w_catboost={best['w_catboost']:.2f}  calib_bss={best['calib_bss']:.2f}  "
          f"(참고: v26 w=0.70 calib_bss=2082.78, v35 w=0.70 calib_bss=2082.07)", flush=True)

    # ---- 모델 아티팩트 저장 ----
    os.makedirs(V37_MODEL_DIR, exist_ok=True)
    os.makedirs(V37_OUT_DIR, exist_ok=True)

    torch.save(encoder.state_dict(), os.path.join(V37_MODEL_DIR, "retrieval_encoder.pt"))
    np.save(os.path.join(V37_MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy())
    np.save(os.path.join(V37_MODEL_DIR, "reference_labels.npy"), y_ref.numpy())
    joblib.dump({"medians": medians, "scaler": scaler}, os.path.join(V37_MODEL_DIR, "numeric_prep.pkl"))

    import shutil
    shutil.copy(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"), os.path.join(V37_MODEL_DIR, "catboost_seed42.cbm"))
    shutil.copy(os.path.join(V26_MODEL_DIR, "trackman_context.pkl"), os.path.join(V37_MODEL_DIR, "trackman_context.pkl"))

    with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["retrieval"]["numeric_cols"] = numeric_cols
    meta["retrieval"]["cat_cols"] = CAT_COLS
    meta["retrieval"]["cat_mappings"] = cat_mappings
    meta["retrieval"]["cardinalities"] = cardinalities
    meta["retrieval"]["embed_dims"] = EMBED_DIMS
    meta["retrieval"]["encoder_hidden"] = ENCODER_HIDDEN
    meta["retrieval"]["embed_out_dim"] = EMBED_OUT_DIM
    meta["blend_weight_catboost"] = best["w_catboost"]
    meta["calibration"] = {"method": "platt_sigmoid", "a": best["a"], "b": best["b"]}
    with open(os.path.join(V37_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(os.path.join(V37_OUT_DIR, "metrics_v37.json"), "w", encoding="utf-8") as f:
        json.dump({
            "batch_size": BATCH_SIZE,
            "best_w_catboost": best["w_catboost"],
            "carveout_calib_bss": best["calib_bss"],
            "calibration_a": best["a"], "calibration_b": best["b"],
            "grid_log": grid_log,
            "encoder_train_elapsed_sec": train_elapsed,
            "n_train_sub": len(train_sub_df), "n_calib": len(calib_df),
            "encoder_param_count": sum(p.numel() for p in encoder.parameters()),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved model artifacts to {V37_MODEL_DIR}", flush=True)
    print(f"Saved metrics to {V37_OUT_DIR}\\metrics_v37.json", flush=True)


if __name__ == "__main__":
    main()
