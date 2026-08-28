"""retrieval(NCA) 점수를 CatBoost와 사후 블렌드하는 대신, CatBoost의 새 입력
피처로 추가. teacher-student distillation(8/24, 소프트라벨로 타겟 자체를
전달 -> 신호 대부분 소실)과는 다른 방향(피처로 전달, CatBoost가 트리 분기로
비선형 결합 가능) -- 사용자 요청으로 fold0/fold2 스크리닝 생략, 바로 프로덕션
빌드+제출 준비.

핵심 설계(리크 방지):
  - calib_df/test 시점 retrieval_score: v26의 기존 encoder+참조집합(train_sub
    전체 140만행)을 그대로 재사용 -- calib_df/test는 애초에 그 참조집합에
    없으므로 리크 없음, 재학습 불필요.
  - train_sub_df(CatBoost가 실제로 학습할 140만행) 자신의 retrieval_score만
    K-fold OOF로 생성(사용자 지시로 K=3, 시간 단축) -- 안 그러면 그 행 자신이
    참조집합에 포함돼 CatBoost가 "정답을 미리 본 초강력 피처"를 학습해버림.

체크포인트: fold별 OOF 예측을 매번 저장(중간에 끊겨도 이어서 진행 가능,
8/25 여러 차례 겪은 background kill 대응).

retrieve_predict/train_encoder는 v25/v26 train_final.py에서 그대로 가져옴
(이미 메모리 안전한 온라인 스트리밍 버전, early stopping 없이 고정 20에폭).
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v32_retrieval_as_feature\model"
OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v32_retrieval_as_feature\output"
CKPT_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER\31c37839-97fb-4f8c-b428-a9f2da4f79c5\scratchpad\v32_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

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

K_FOLDS = 3  # 사용자 지시로 5 -> 3 축소(시간 단축, OOF 품질은 약간 낮아지지만 합리적 타협)


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
    """v25/v26 train_final.py와 동일한 메모리 안전 온라인 스트리밍 버전
    (청크 하나 처리 후 즉시 버리고 누적 max/numer/denom만 유지)."""
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


def generate_oof_retrieval_scores(train_sub_df, numeric_cols):
    """train_sub_df(140만행) 자신에 대한 K-fold OOF retrieval_score.
    fold별 체크포인트 저장/재사용."""
    oof_path = os.path.join(CKPT_DIR, "oof_retrieval_score.npy")
    progress_path = os.path.join(CKPT_DIR, "oof_progress.json")

    n = len(train_sub_df)
    if os.path.exists(oof_path) and os.path.exists(progress_path):
        oof = np.load(oof_path)
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        print(f" 기존 OOF 체크포인트 재사용: 완료된 fold={progress['done_folds']}", flush=True)
    else:
        oof = np.full(n, np.nan, dtype=np.float64)
        progress = {"done_folds": []}

    y_all = train_sub_df[TARGET_COL].values
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.arange(n), y_all))

    for fold_i, (tr_idx, oof_idx) in enumerate(splits):
        if fold_i in progress["done_folds"]:
            print(f"\n=== OOF fold {fold_i+1}/{K_FOLDS}: 이미 완료됨, 스킵 ===", flush=True)
            continue
        print(f"\n=== OOF fold {fold_i+1}/{K_FOLDS}: train={len(tr_idx)}  oof_query={len(oof_idx)} ===", flush=True)
        tr_fold_df = train_sub_df.iloc[tr_idx].reset_index(drop=True)
        oof_fold_df = train_sub_df.iloc[oof_idx].reset_index(drop=True)

        nn_cat_mappings, cardinalities = build_nn_cat_mappings(tr_fold_df)
        cat_train_nn = encode_cats(tr_fold_df, nn_cat_mappings)
        cat_oof_nn = encode_cats(oof_fold_df, nn_cat_mappings)
        x_num_train, medians, scaler = prep_numeric_fit(tr_fold_df, numeric_cols)
        x_num_oof = prep_numeric_transform(oof_fold_df, numeric_cols, medians, scaler)

        print(f"   encoder 학습...", flush=True)
        t0 = time.time()
        encoder, device = train_encoder(
            cat_train_nn, x_num_train, tr_fold_df[TARGET_COL], cardinalities, x_num_train.shape[1],
        )
        print(f"   encoder 학습 완료 ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        z_ref = compute_embeddings(encoder, device, cat_train_nn, x_num_train)
        y_ref = tr_fold_df[TARGET_COL].values
        print(f"   참조 임베딩 계산 완료 ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        oof_pred = retrieve_predict(encoder, device, cat_oof_nn, x_num_oof, z_ref, y_ref)
        print(f"   OOF 추론 완료 ({time.time()-t0:.1f}s)", flush=True)

        oof[oof_idx] = oof_pred
        progress["done_folds"].append(fold_i)
        np.save(oof_path, oof)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f)
        print(f"   체크포인트 저장 완료 (fold {fold_i+1}/{K_FOLDS})", flush=True)

    assert not np.isnan(oof).any(), "OOF 생성 누락된 행이 있음"
    return oof


def main():
    print("Load train data + control_risk_score(원재료 제거)...", flush=True)
    df = load_data()
    df = add_risk_score_drop_ingredients(df)
    print(f" shape={df.shape}", flush=True)

    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(
        df, test_size=0.05, stratify=y_all, random_state=SEED,
    )
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    print(f" train_sub={train_sub_df.shape}  calib={calib_df.shape} (v26과 동일 split)", flush=True)

    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]

    # ---------- 1) train_sub 자신의 retrieval_score: K-fold OOF ----------
    print(f"\n########## PHASE 1: train_sub OOF retrieval_score (K={K_FOLDS}) ##########", flush=True)
    oof_retrieval_score = generate_oof_retrieval_scores(train_sub_df, numeric_cols)
    print(f" OOF 생성 완료. mean={oof_retrieval_score.mean():.4f} std={oof_retrieval_score.std():.4f}", flush=True)

    # ---------- 2) calib_df의 retrieval_score: v26 기존 encoder+참조집합 재사용(재학습 없음) ----------
    calib_score_path = os.path.join(CKPT_DIR, "calib_retrieval_score.npy")
    if os.path.exists(calib_score_path):
        print("\n########## PHASE 2: calib retrieval_score - 체크포인트 재사용 ##########", flush=True)
        calib_retrieval_score = np.load(calib_score_path)
    else:
        print("\n########## PHASE 2: calib retrieval_score (v26 encoder+참조집합 재사용) ##########", flush=True)
        nn_cat_mappings_v26 = None
        with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
            v26_meta = json.load(f)
        nn_cat_mappings_v26 = v26_meta["retrieval"]["cat_mappings"]
        cardinalities_v26 = v26_meta["retrieval"]["cardinalities"]
        numeric_cols_v26 = v26_meta["retrieval"]["numeric_cols"]
        assert numeric_cols_v26 == numeric_cols, "numeric_cols가 v26과 다름 -- 재현 오류"

        numeric_prep = joblib.load(os.path.join(V26_MODEL_DIR, "numeric_prep.pkl"))
        medians_v26, scaler_v26 = numeric_prep["medians"], numeric_prep["scaler"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        v26_encoder = RowEncoder(cardinalities_v26, len(numeric_cols) * 2).to(device)
        v26_encoder.load_state_dict(torch.load(os.path.join(V26_MODEL_DIR, "retrieval_encoder.pt"), map_location=device))
        v26_encoder.eval()
        z_ref_v26 = torch.tensor(np.load(os.path.join(V26_MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
        y_ref_v26 = np.load(os.path.join(V26_MODEL_DIR, "reference_labels.npy"))

        cat_calib_nn = encode_cats(calib_df, nn_cat_mappings_v26)
        x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians_v26, scaler_v26)
        t0 = time.time()
        calib_retrieval_score = retrieve_predict(v26_encoder, device, cat_calib_nn, x_num_calib, z_ref_v26, y_ref_v26)
        print(f" calib retrieval 추론 완료 ({time.time()-t0:.1f}s)", flush=True)
        np.save(calib_score_path, calib_retrieval_score)

    # ---------- 3) CatBoost 재학습 (retrieval_score 새 피처 추가) ----------
    print("\n########## PHASE 3: CatBoost 재학습 (retrieval_score 피처 추가) ##########", flush=True)
    cb_id_mappings = build_catboost_id_mappings(train_sub_df)
    X_train_cb = build_catboost_features(train_sub_df, cb_id_mappings)
    X_train_cb["retrieval_score"] = oof_retrieval_score
    y_train_cb = train_sub_df[TARGET_COL]

    X_calib_cb = build_catboost_features(calib_df, cb_id_mappings)
    X_calib_cb["retrieval_score"] = calib_retrieval_score
    y_calib = calib_df[TARGET_COL]

    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]
    print(f" n_features={X_train_cb.shape[1]} (기존 70 + retrieval_score 1)", flush=True)

    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED,
        cat_features=cat_idx, verbose=200, **BEST_PARAMS,
    )
    cb_model.fit(X_train_cb, y_train_cb)
    cb_elapsed = time.time() - t0
    print(f" CatBoost 학습 완료 ({cb_elapsed:.1f}s)", flush=True)

    cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]
    a_final, b_final = fit_platt_scaling(cb_calib_raw, y_calib)
    cb_calib_pred = apply_platt_scaling(cb_calib_raw, a_final, b_final)

    # feature importance로 retrieval_score 기여도 확인(참고용)
    importances = pd.Series(cb_model.get_feature_importance(), index=X_train_cb.columns).sort_values(ascending=False)
    retrieval_rank = list(importances.index).index("retrieval_score") + 1
    print(f"\n retrieval_score importance={importances['retrieval_score']:.4f}  "
          f"(전체 {len(importances)}개 피처 중 {retrieval_rank}위)", flush=True)
    print(f" 상위 10개 피처:\n{importances.head(10)}", flush=True)

    metrics = {
        "k_folds": K_FOLDS,
        "cb_elapsed_sec": cb_elapsed,
        "carveout_bss_raw": bss_score(cb_calib_raw, y_calib),
        "carveout_bss_calibrated": bss_score(cb_calib_pred, y_calib),
        "retrieval_score_importance": float(importances["retrieval_score"]),
        "retrieval_score_importance_rank": retrieval_rank,
        "n_features": int(X_train_cb.shape[1]),
    }
    print(f"\n carve-out BSS: raw={metrics['carveout_bss_raw']:.2f}  "
          f"calibrated={metrics['carveout_bss_calibrated']:.2f}", flush=True)
    print(f" (참고: v22(987, retrieval_score 없음) 동일 방식 carve-out은 이전 세션에서 2057.20)", flush=True)

    with open(os.path.join(OUT_DIR, "metrics_v32.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cb_model.save_model(os.path.join(MODEL_DIR, "catboost_seed42.cbm"))

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_train_cb.columns),
            "cat_cols": CATBOOST_CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": cb_id_mappings,
            "retrieval_feature": {
                "numeric_cols": numeric_cols,
                "cat_cols": ALL_CAT_FOR_NN,
                "embed_dims": EMBED_DIMS,
                "encoder_hidden": ENCODER_HIDDEN,
                "embed_out_dim": EMBED_OUT_DIM,
                "source": "v26_encoder_and_reference_reused",
            },
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved model + feature_meta.json to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
