"""jh_ws v26(1023점) 레시피 위에서 세 가지를 동시에 확장:
  1) CatBoost 1시드 -> 6시드(42/7/123/1/99/777) 배깅 (이 프로젝트에서 유일하게
     로컬<->실측 전이가 신뢰됐던 축)
  2) LGBM 3시드를 새 축으로 추가해 CatBoost+LGBM+Retrieval 3-way 블렌드로 확장
     (es_ws v9에서 아키텍처 블렌드가 982->986 냈던 전례, retrieval 조합에서는
     미검증)
  3) REFERENCE_SIZE는 이미 v25/v26에서 서브샘플 없이 전체(133만행)를 쓰고
     있어 추가로 키울 여지 없음 -- 이 스크립트에서는 그대로 유지

레시피는 v22/v25/v26과 동일(control_risk_score 추가 + 원재료 3종 제거).
calibration carve-out(5%, stratify, random_state=42)도 동일 -> v26의
carveout_bss_blend_calibrated=2082.78과 apples-to-apples 비교 가능.

CatBoost/Retrieval encoder는 GPU(RTX 4060, task_type=GPU)로 학습해 6시드
배깅에도 시간 여유를 확보.
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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../es_ws/work
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
TRACKMAN_CONTEXT_PATH = os.path.join(_BASE, "model", "trackman_context.pkl")
MODEL_DIR = os.path.join(_BASE, "model_retrieval_blend_v3")
OUT_DIR = os.path.join(_BASE, "output")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

CB_SEEDS = [42, 7, 123, 1, 99, 777]
LGB_SEEDS = [42, 7, 123]

CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
CAT_ITERATIONS = 2000
CAT_TASK_TYPE = "GPU"

LGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.02, num_leaves=63,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, verbosity=-1,
)

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
REFERENCE_SIZE = None  # v25/v26과 동일: 서브샘플 없이 train_sub 전체 참조
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000


def log(msg):
    print(msg, flush=True)


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


def build_id_mappings(df):
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


def build_lgb_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str).astype("category")
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
    log(f"  NCA encoder device={device}")
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
        log(f"    epoch {epoch+1}/{NCA_EPOCHS} train_loss={epoch_loss/max(n_batches,1):.5f}")

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


def grid_search_3way(cb_r, lgb_r, nca_r, y, step=0.05):
    best = None
    n = int(round(1 / step))
    for i in range(n + 1):
        w_nca = i * step
        for j in range(n + 1 - i):
            w_lgb = j * step
            w_cb = 1.0 - w_nca - w_lgb
            if w_cb < -1e-9:
                continue
            raw = w_cb * cb_r + w_lgb * lgb_r + w_nca * nca_r
            score = bss_score(raw, y)
            if best is None or score > best[0]:
                best = (score, w_cb, w_lgb, w_nca)
    return best


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    log("Load train data + control_risk_score 추가, 원재료3 제거 (v22/v25/v26과 동일 레시피)...")
    df = load_data()
    df = add_risk_score_drop_ingredients(df)
    log(f" shape={df.shape}  ({time.time()-t0:.0f}s)")

    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        df, test_size=0.05, stratify=y_all, random_state=SEED,
    )
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    log(f" train_sub={train_sub_df.shape}  calib={calib_df.shape}  ({time.time()-t0:.0f}s)")

    id_mappings = build_id_mappings(train_sub_df)
    y_train = train_sub_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]

    # ---------- CatBoost 6시드 배깅 ----------
    log(f"\n=== CatBoost {len(CB_SEEDS)}시드 학습 (GPU, iterations={CAT_ITERATIONS}) ===")
    X_train_cb = build_catboost_features(train_sub_df, id_mappings)
    X_calib_cb = build_catboost_features(calib_df, id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]

    cb_calib_raws = []
    for seed in CB_SEEDS:
        ts = time.time()
        log(f" --- CatBoost seed={seed} 시작 ({time.time()-t0:.0f}s elapsed) ---")
        try:
            cb_model = CatBoostClassifier(
                iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=seed,
                cat_features=cat_idx, verbose=200, task_type=CAT_TASK_TYPE, devices="0",
                **CAT_BEST_PARAMS,
            )
            cb_model.fit(X_train_cb, y_train)
        except Exception as e:
            log(f"   GPU 학습 실패({e}), CPU로 재시도")
            cb_model = CatBoostClassifier(
                iterations=CAT_ITERATIONS, loss_function="Logloss", random_seed=seed,
                cat_features=cat_idx, verbose=200, **CAT_BEST_PARAMS,
            )
            cb_model.fit(X_train_cb, y_train)
        cb_model.save_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cb_calib_raws.append(cb_model.predict_proba(X_calib_cb)[:, 1])
        log(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    cb_calib_raw = np.mean(cb_calib_raws, axis=0)
    log(f" CatBoost {len(CB_SEEDS)}시드 평균 carve-out BSS(raw)={bss_score(cb_calib_raw, y_calib):.2f}")

    # ---------- LGBM 3시드 배깅 (신규 3rd 축) ----------
    log(f"\n=== LightGBM {len(LGB_SEEDS)}시드 학습 ===")
    X_train_lgb = build_lgb_features(train_sub_df, id_mappings)
    X_calib_lgb = build_lgb_features(calib_df, id_mappings)

    lgb_calib_raws = []
    for seed in LGB_SEEDS:
        ts = time.time()
        m = LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m.fit(X_train_lgb, y_train, categorical_feature=CATBOOST_CAT_COLS)
        m.booster_.save_model(os.path.join(MODEL_DIR, f"lgb_seed{seed}.txt"))
        lgb_calib_raws.append(m.predict_proba(X_calib_lgb)[:, 1])
        log(f" seed={seed} 완료 ({time.time()-ts:.0f}s)")
    lgb_calib_raw = np.mean(lgb_calib_raws, axis=0)
    log(f" LGBM {len(LGB_SEEDS)}시드 평균 carve-out BSS(raw)={bss_score(lgb_calib_raw, y_calib):.2f}")

    # ---------- Retrieval encoder (1회, 전체 참조) ----------
    log("\n=== Retrieval encoder(NCA) 학습 ===")
    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]
    log(f" n_numeric={len(numeric_cols)} cardinalities={cardinalities}")

    cat_train_nn = encode_cats(train_sub_df, nn_cat_mappings)
    cat_calib_nn = encode_cats(calib_df, nn_cat_mappings)
    x_num_train, medians, scaler = prep_numeric_fit(train_sub_df, numeric_cols)
    x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians, scaler)

    ts = time.time()
    encoder, device = train_encoder(
        cat_train_nn, x_num_train, train_sub_df[TARGET_COL], cardinalities, x_num_train.shape[1],
    )
    log(f" encoder 학습 완료 ({time.time()-ts:.0f}s)")

    rng = np.random.RandomState(SEED)
    n_train_sub = len(train_sub_df)
    ref_size = n_train_sub if REFERENCE_SIZE is None else min(REFERENCE_SIZE, n_train_sub)
    ref_idx = rng.choice(n_train_sub, size=ref_size, replace=False)
    log(f" 참조 집합: {ref_size}/{n_train_sub} (서브샘플 없음, v25/v26과 동일)")

    cat_ref_nn = {c: v[ref_idx] for c, v in cat_train_nn.items()}
    x_num_ref = x_num_train[ref_idx]
    y_ref = train_sub_df[TARGET_COL].values[ref_idx]

    ts = time.time()
    z_ref = compute_embeddings(encoder, device, cat_ref_nn, x_num_ref)
    log(f" 참조 임베딩 계산 완료 ({time.time()-ts:.0f}s)")

    ts = time.time()
    nca_calib_raw = retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
    log(f" calib retrieval 추론 완료 ({time.time()-ts:.0f}s)")
    log(f" Retrieval 단독 carve-out BSS(raw)={bss_score(nca_calib_raw, y_calib):.2f}")

    # ---------- 2-way(CB+NCA) 재현 확인 + 3-way(CB+LGB+NCA) 그리드서치 ----------
    log("\n=== 블렌드 가중치 탐색 + Platt calibration ===")
    blend2_raw = 0.7 * cb_calib_raw + 0.3 * nca_calib_raw
    a2, b2 = fit_platt_scaling(blend2_raw, y_calib)
    blend2_calib = apply_platt_scaling(blend2_raw, a2, b2)
    log(f" [재현] CB(6시드avg):NCA=0.7:0.3 calibrated BSS={bss_score(blend2_calib, y_calib):.2f} "
        f"(v26 1시드 기준 2082.78과 비교)")

    best_score, w_cb, w_lgb, w_nca = grid_search_3way(cb_calib_raw, lgb_calib_raw, nca_calib_raw, y_calib, step=0.05)
    log(f" [그리드서치] best raw BSS={best_score:.2f} at w_cb={w_cb:.2f} w_lgb={w_lgb:.2f} w_nca={w_nca:.2f}")

    blend3_raw = w_cb * cb_calib_raw + w_lgb * lgb_calib_raw + w_nca * nca_calib_raw
    a3, b3 = fit_platt_scaling(blend3_raw, y_calib)
    blend3_calib = apply_platt_scaling(blend3_raw, a3, b3)
    log(f" [3-way 최종] calibrated BSS={bss_score(blend3_calib, y_calib):.2f}")

    metrics = {
        "cb_seeds": CB_SEEDS, "lgb_seeds": LGB_SEEDS,
        "reference_size": ref_size,
        "carveout_bss_cb6_only": bss_score(cb_calib_raw, y_calib),
        "carveout_bss_lgb3_only": bss_score(lgb_calib_raw, y_calib),
        "carveout_bss_nca_only": bss_score(nca_calib_raw, y_calib),
        "carveout_bss_2way_cb_nca_07_03_calibrated": bss_score(blend2_calib, y_calib),
        "grid_search_3way": {"w_cb": w_cb, "w_lgb": w_lgb, "w_nca": w_nca, "raw_bss": best_score},
        "carveout_bss_3way_calibrated": bss_score(blend3_calib, y_calib),
        "reference_v26_1seed_2way_calibrated": 2082.7818915293506,
    }
    log(f"\n요약: v26(1시드,2way)=2082.78  ->  v3 2way(6시드)={metrics['carveout_bss_2way_cb_nca_07_03_calibrated']:.2f}"
        f"  ->  v3 3way(6시드+lgb3)={metrics['carveout_bss_3way_calibrated']:.2f}")
    with open(os.path.join(OUT_DIR, "metrics_retrieval_blend_v3.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---------- 저장 ----------
    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "retrieval_encoder.pt"))
    np.save(os.path.join(MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy().astype(np.float32))
    np.save(os.path.join(MODEL_DIR, "reference_labels.npy"), y_ref.astype(np.float32))
    joblib.dump({"medians": medians, "scaler": scaler}, os.path.join(MODEL_DIR, "numeric_prep.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "catboost": {
                "columns": list(X_train_cb.columns), "cat_cols": CATBOOST_CAT_COLS,
                "raw_id_cols": RAW_ID_COLS, "id_mappings": id_mappings, "seeds": CB_SEEDS,
            },
            "lgb": {
                "columns": list(X_train_lgb.columns), "cat_cols": CATBOOST_CAT_COLS,
                "raw_id_cols": RAW_ID_COLS, "id_mappings": id_mappings, "seeds": LGB_SEEDS,
            },
            "retrieval": {
                "numeric_cols": numeric_cols, "cat_cols": ALL_CAT_FOR_NN,
                "cat_mappings": nn_cat_mappings, "cardinalities": cardinalities,
                "embed_dims": EMBED_DIMS, "encoder_hidden": ENCODER_HIDDEN, "embed_out_dim": EMBED_OUT_DIM,
            },
            "blend_weights": {"w_cb": w_cb, "w_lgb": w_lgb, "w_nca": w_nca},
            "calibration": {"method": "platt_sigmoid", "a": a3, "b": b3},
        }, f, indent=2, ensure_ascii=False)
    log(f"\nSaved all artifacts to {MODEL_DIR}  (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
