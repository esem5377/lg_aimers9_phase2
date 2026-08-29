"""v37 (retrieval encoder in-batch negative pool 확대: BATCH_SIZE 1024->4096)의
fold0_2022/fold2_2024 walk-forward 검증.

배경(2026-08-28): cosine similarity는 이미 v25부터 encode()에서 L2 정규화 후
z@z.T로 적용돼 있었음(RowEncoder.encode 마지막 줄 nn.functional.normalize) --
"cosine으로 전환" 요청은 델타가 없는 실험이라 기각, 대신 아직 안 건드린
negative sampling 쪽(현재는 배치 내 나머지 전부를 소프트 이웃/네거티브로
사용, BATCH_SIZE=1024)에서 in-batch negative pool을 4배(4096)로 늘려
학습 신호가 개선되는지 검증.

v32_walkforward.py와 달리 이건 "retrieval_score를 CatBoost 피처로 추가"가
아니라 v25/v26/v34 프로덕션과 동일한 **블렌드 구조**(CatBoost raw와 retrieval
raw를 0.7:0.3으로 섞은 뒤 1회 Platt calibration)를 그대로 씀. CatBoost는
encoder batch size와 무관하므로 fold당 1회만 학습해 재사용(baseline/treatment
공통) -- v32_walkforward 대비 CatBoost fit 비용 절반.

split: train_full(해당 fold의 시즌 <= 컷오프) -> train_sub 95% / calib 5%
(stratified, seed=42). eval은 다음 시즌 전체(완전 미관측, 진짜 walk-forward).
encoder/CatBoost 모두 train_sub 전체로 학습(REFERENCE_SIZE=None, v25/v26과
동일 -- reference subsampling 없음). calib_df로 Platt fit, eval_df로 BSS 계산.

프로젝트 규칙(lg_aimers9_walkforward_methodology 메모리): fold0/fold2 둘 다
양의 델타가 나와야 실제 제출 후보로 취급.
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "v37_walkforward_results.json")

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
NCA_EPOCHS = 20
NCA_LR = 1e-3
NCA_WEIGHT_DECAY = 1e-5
NCA_TEMP_INIT = 0.1
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000
W_CATBOOST = 0.7

BATCH_VARIANTS = [("batch1024_baseline", 1024), ("batch4096_treatment", 4096)]
BATCH_VARIANT_NAMES = [name for name, _ in BATCH_VARIANTS]


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
    from sklearn.preprocessing import StandardScaler
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


def train_encoder(cat_train, x_num_train, y_train, cardinalities, n_numeric, batch_size, tag, ckpt_path=None):
    """epoch 단위 체크포인트 지원(ckpt_path가 주어지면 매 epoch 끝에 model/opt
    state 저장, 재시작 시 이어서 학습 -- 배치 배치 크기가 클수록(4096) 한 epoch
    자체가 길어 epoch 도중 죽으면 이전엔 통째로 날아갔음(2026-08-28 fold2
    batch1024 19/20 epoch에서 원인 불명으로 프로세스 종료된 사례로 발견)."""
    model = RowEncoder(cardinalities, n_numeric)
    opt = torch.optim.Adam(model.parameters(), lr=NCA_LR, weight_decay=NCA_WEIGHT_DECAY)
    loss_fn = nn.BCELoss()
    cat_train_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_train.items()}
    x_num_train_t = torch.tensor(x_num_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values.astype(np.float32))
    n = len(y_train)

    start_epoch = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"    [{tag}] 체크포인트 발견, epoch {start_epoch}부터 재개: {ckpt_path}", flush=True)

    for epoch in range(start_epoch, NCA_EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch_idx in make_batches(n, batch_size, shuffle=True, seed=SEED + epoch):
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
        print(f"    [{tag}] epoch {epoch+1}/{NCA_EPOCHS} train_loss={epoch_loss/max(n_batches,1):.5f} n_batches={n_batches}", flush=True)
        if ckpt_path is not None:
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict()}, ckpt_path)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
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
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints", fold_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_full_df = add_risk_score_drop_ingredients(train_full_df).reset_index(drop=True)
    eval_df = add_risk_score_drop_ingredients(eval_df).reset_index(drop=True)

    y_all = train_full_df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(train_full_df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    print(f" train_sub={len(train_sub_df)} calib={len(calib_df)} eval={len(eval_df)}", flush=True)

    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]

    # ---- CatBoost (encoder batch size와 무관, fold당 1회만 학습해 재사용) ----
    # 체크포인트: 재시작 시 이미 학습된 모델/raw 예측이 있으면 재학습 없이 로드
    cb_model_path = os.path.join(ckpt_dir, "cb_model.cbm")
    cb_calib_raw_path = os.path.join(ckpt_dir, "cb_calib_raw.npy")
    cb_eval_raw_path = os.path.join(ckpt_dir, "cb_eval_raw.npy")

    cb_id_mappings = build_catboost_id_mappings(train_sub_df)
    X_calib_cb = build_catboost_features(calib_df, cb_id_mappings)
    X_eval_cb = build_catboost_features(eval_df, cb_id_mappings)
    y_train_sub = train_sub_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]

    if os.path.exists(cb_model_path) and os.path.exists(cb_calib_raw_path) and os.path.exists(cb_eval_raw_path):
        print(f" CatBoost 체크포인트 발견, 재학습 스킵 ({cb_model_path})", flush=True)
        cb_calib_raw = np.load(cb_calib_raw_path)
        cb_eval_raw = np.load(cb_eval_raw_path)
    else:
        X_train_cb = build_catboost_features(train_sub_df, cb_id_mappings)
        cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]
        t0 = time.time()
        cb_model = CatBoostClassifier(iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED, cat_features=cat_idx, verbose=False, **BEST_PARAMS)
        cb_model.fit(X_train_cb, y_train_sub)
        print(f" CatBoost 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
        cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]
        cb_eval_raw = cb_model.predict_proba(X_eval_cb)[:, 1]
        cb_model.save_model(cb_model_path)
        np.save(cb_calib_raw_path, cb_calib_raw)
        np.save(cb_eval_raw_path, cb_eval_raw)
        print(f" CatBoost 체크포인트 저장 완료 ({cb_model_path})", flush=True)

    # ---- NN용 인코딩 (인코더 배치 변형과 무관, 1회만) ----
    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub_df)
    cat_train_nn = encode_cats(train_sub_df, nn_cat_mappings)
    x_num_train, medians, scaler = prep_numeric_fit(train_sub_df, numeric_cols)
    cat_calib_nn = encode_cats(calib_df, nn_cat_mappings)
    x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians, scaler)
    cat_eval_nn = encode_cats(eval_df, nn_cat_mappings)
    x_num_eval = prep_numeric_transform(eval_df, numeric_cols, medians, scaler)
    y_ref = train_sub_df[TARGET_COL].values

    # 폴드가 이전 실행에서 일부 variant까지만 끝난 채 중단됐을 수 있음 -> 이어서 진행
    fold_result = all_results.get(fold_name, {})
    fold_result["n_train_sub"] = len(train_sub_df)
    fold_result["n_calib"] = len(calib_df)
    fold_result["n_eval"] = len(eval_df)
    all_results[fold_name] = fold_result
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    for variant_name, batch_size in BATCH_VARIANTS:
        if variant_name in fold_result:
            print(f"\n --- {fold_name} / {variant_name}: 이미 완료됨(eval_bss={fold_result[variant_name]['eval_bss']:.2f}), 스킵 ---", flush=True)
            continue

        print(f"\n --- {fold_name} / {variant_name} (batch_size={batch_size}) ---", flush=True)
        encoder_ckpt_path = os.path.join(ckpt_dir, f"encoder_{variant_name}.pt")  # 학습 완료된 최종 state
        encoder_train_ckpt_path = os.path.join(ckpt_dir, f"encoder_train_ckpt_{variant_name}.pt")  # epoch 단위 재개용
        n_numeric = x_num_train.shape[1]

        if os.path.exists(encoder_ckpt_path):
            print(f"  encoder 체크포인트 발견, 재학습 스킵 ({encoder_ckpt_path})", flush=True)
            encoder = RowEncoder(cardinalities, n_numeric)
            encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
        else:
            t0 = time.time()
            encoder = train_encoder(cat_train_nn, x_num_train, y_train_sub, cardinalities, n_numeric, batch_size, f"{fold_name}/{variant_name}", ckpt_path=encoder_train_ckpt_path)
            print(f"  encoder 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
            torch.save(encoder.state_dict(), encoder_ckpt_path)
            print(f"  encoder 체크포인트 저장 완료 ({encoder_ckpt_path})", flush=True)

        nca_calib_path = os.path.join(ckpt_dir, f"nca_calib_raw_{variant_name}.npy")
        nca_eval_path = os.path.join(ckpt_dir, f"nca_eval_raw_{variant_name}.npy")
        if os.path.exists(nca_calib_path) and os.path.exists(nca_eval_path):
            print(f"  retrieval 추론 체크포인트 발견, 재계산 스킵", flush=True)
            nca_calib_raw = np.load(nca_calib_path)
            nca_eval_raw = np.load(nca_eval_path)
        else:
            t0 = time.time()
            z_ref = compute_embeddings(encoder, cat_train_nn, x_num_train)
            nca_calib_raw = retrieve_predict(encoder, cat_calib_nn, x_num_calib, z_ref, y_ref)
            nca_eval_raw = retrieve_predict(encoder, cat_eval_nn, x_num_eval, z_ref, y_ref)
            print(f"  retrieval 추론 완료 ({time.time()-t0:.1f}s)", flush=True)
            np.save(nca_calib_path, nca_calib_raw)
            np.save(nca_eval_path, nca_eval_raw)

        blend_calib_raw = W_CATBOOST * cb_calib_raw + (1 - W_CATBOOST) * nca_calib_raw
        blend_eval_raw = W_CATBOOST * cb_eval_raw + (1 - W_CATBOOST) * nca_eval_raw
        a, b = fit_platt(blend_calib_raw, y_calib)
        blend_eval_calib = apply_platt(blend_eval_raw, a, b)
        eval_bss = bss_score(blend_eval_calib, y_eval)
        print(f"  [{variant_name}] eval_bss={eval_bss:.2f}", flush=True)

        fold_result[variant_name] = {"batch_size": batch_size, "eval_bss": eval_bss}
        # variant 하나 끝날 때마다 즉시 저장 -> 중단돼도 다음 variant부터 재개
        all_results[fold_name] = fold_result
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    if "batch1024_baseline" in fold_result and "batch4096_treatment" in fold_result:
        delta = fold_result["batch4096_treatment"]["eval_bss"] - fold_result["batch1024_baseline"]["eval_bss"]
        fold_result["delta_treatment_vs_baseline"] = delta
        print(f" [{fold_name}] delta(4096-1024) = {delta:+.2f}", flush=True)

    all_results[fold_name] = fold_result
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
        if "delta_treatment_vs_baseline" in all_results.get(fold_name, {}):
            print(f"\n=== {fold_name}: 이미 완료됨(두 variant 모두), 스킵 ===", flush=True)
            continue
        run_fold(fold_name, train_df, eval_df, all_results)

    print("\n=== SUMMARY ===", flush=True)
    for fold_name, r in all_results.items():
        if "delta_treatment_vs_baseline" not in r:
            print(f"  {fold_name}: 미완료 (변형 결과: {[k for k in BATCH_VARIANT_NAMES if k in r]})", flush=True)
            continue
        b = r["batch1024_baseline"]["eval_bss"]
        t = r["batch4096_treatment"]["eval_bss"]
        print(f"  {fold_name}: baseline(1024)={b:.2f} treatment(4096)={t:.2f} delta={r['delta_treatment_vs_baseline']:+.2f}", flush=True)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
