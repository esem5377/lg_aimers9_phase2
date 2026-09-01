"""fold0(+11.28)/fold2(+5.28) walk-forward에서 CatBoost RMSE loss(대신 Logloss)가
CatBoost+retrieval(0.7:0.3) 블렌드 안에서도 일관되게 개선을 보인 결과를 바탕으로,
실제 제출용 프로덕션 패키지를 만든다.

- CatBoost: 987 레시피(원재료 drop) 그대로, 전체 데이터(train_sub, 95%)로 새로 학습.
  단 loss_function만 Logloss -> RMSE(CatBoostRegressor).
- retrieval: v26의 검증된 encoder를 그대로 재사용(재학습 없음, 8/24에 이미 bit-identical
  재현 확인된 기법) -- reference_embeddings.npy는 encoder 순전파로 재계산.
- 블렌드 가중치: w_catboost=0.7 고정(v26/walk-forward 테스트와 동일 구성).
- Platt calibration은 calib(5%)에서 새로 fit.
"""
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../es_ws/work
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
TRACKMAN_CONTEXT_PATH = os.path.join(_BASE, "model", "trackman_context.pkl")
V26_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(_BASE)), "jh_ws", "v26_retrieval_blend_w07", "model")
OUT_MODEL_DIR = os.path.join(_BASE, "model_rmse_retrieval")
OUT_DIR = os.path.join(_BASE, "output")
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

SEED = 42
ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]
W_CATBOOST = 0.7
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000

CAT_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
CAT_ITERATIONS = 2000


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


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


def build_catboost_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def encode_cats(df, mappings):
    out = {}
    for col, m in mappings.items():
        oov = len(m)
        out[col] = df[col].astype(str).map(m).fillna(oov).astype(int).values
    return out


def prep_numeric_transform(df, numeric_cols, medians, scaler):
    X = df[numeric_cols].copy()
    isna_flags = X.isna().astype(np.float32)
    isna_flags.columns = [f"{c}__isna" for c in numeric_cols]
    X = X.fillna(medians)
    X_all = pd.concat([X, isna_flags], axis=1)
    return scaler.transform(X_all.values.astype(np.float32)).astype(np.float32)


class RowEncoder(nn.Module):
    def __init__(self, cat_cardinalities, embed_dims, encoder_hidden, embed_out_dim, dropout, n_numeric):
        super().__init__()
        self.embeds = nn.ModuleDict()
        embed_out = 0
        for col, card in cat_cardinalities.items():
            dim = embed_dims[col]
            self.embeds[col] = nn.Embedding(card + 1, dim)
            embed_out += dim
        in_dim = embed_out + n_numeric
        layers = []
        prev = in_dim
        for h in encoder_hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, embed_out_dim)]
        self.mlp = nn.Sequential(*layers)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def encode(self, cat_tensors, x_num):
        parts = [self.embeds[col](cat_tensors[col]) for col in self.embeds]
        parts.append(x_num)
        x = torch.cat(parts, dim=1)
        z = self.mlp(x)
        return nn.functional.normalize(z, dim=1)


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


def main():
    log("Load train data (987 레시피, 원재료 drop)...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df = add_risk_score_drop_ingredients(df)
    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[TARGET_COL]
    log(f" train_sub={train_sub_df.shape}  calib={calib_df.shape}")

    id_mappings = {c: {v: i for i, v in enumerate(sorted(train_sub_df[c].astype(str).unique()))} for c in RAW_ID_COLS}
    X_train_cb = build_catboost_features(train_sub_df, id_mappings)
    X_calib_cb = build_catboost_features(calib_df, id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]
    y_train_cb = train_sub_df[TARGET_COL]

    log("\n=== CatBoost RMSE 전체 데이터 학습 (프로덕션) ===")
    cb_model = CatBoostRegressor(
        iterations=CAT_ITERATIONS, loss_function="RMSE", random_seed=SEED,
        cat_features=cat_idx, verbose=False, task_type="GPU", devices="0",
        **CAT_BEST_PARAMS,
    )
    cb_model.fit(X_train_cb, y_train_cb)
    cb_calib_raw = cb_model.predict(X_calib_cb)
    log(f" CatBoost RMSE calib carve-out BSS(raw)={bss_score(cb_calib_raw, y_calib):.2f}")
    cb_model.save_model(os.path.join(OUT_MODEL_DIR, "catboost_seed42.cbm"))

    log("\n=== v26 retrieval_encoder.pt 재사용 (재학습 없음) + 임베딩 재계산 ===")
    with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        v26_meta = json.load(f)
    meta_rt = v26_meta["retrieval"]
    numeric_prep = joblib.load(os.path.join(V26_MODEL_DIR, "numeric_prep.pkl"))
    medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_numeric = len(meta_rt["numeric_cols"]) * 2
    encoder = RowEncoder(
        meta_rt["cardinalities"], meta_rt["embed_dims"], meta_rt["encoder_hidden"],
        meta_rt["embed_out_dim"], 0.0, n_numeric,
    ).to(device)
    state_dict = torch.load(os.path.join(V26_MODEL_DIR, "retrieval_encoder.pt"), map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    cat_train_nn = encode_cats(train_sub_df, meta_rt["cat_mappings"])
    cat_calib_nn = encode_cats(calib_df, meta_rt["cat_mappings"])
    x_num_train = prep_numeric_transform(train_sub_df, meta_rt["numeric_cols"], medians, scaler)
    x_num_calib = prep_numeric_transform(calib_df, meta_rt["numeric_cols"], medians, scaler)

    z_ref = compute_embeddings(encoder, device, cat_train_nn, x_num_train)
    y_ref = train_sub_df[TARGET_COL].values
    nca_calib_raw = retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
    nca_bss = bss_score(nca_calib_raw, y_calib)
    log(f" Retrieval(v26 encoder 재사용) carve-out BSS(raw)={nca_bss:.2f} "
        f"(원본 v25 단독 1896.79와 비교, 재현 확인용)")

    log(f"\n=== 블렌드 (w_catboost={W_CATBOOST}, walk-forward 검증과 동일 구성) ===")
    raw_blend = W_CATBOOST * cb_calib_raw + (1 - W_CATBOOST) * nca_calib_raw
    a, b = fit_platt_scaling(raw_blend, y_calib)
    calib_blend = apply_platt_scaling(raw_blend, a, b)
    calib_bss = bss_score(calib_blend, y_calib)
    log(f" calibrated carve-out BSS = {calib_bss:.2f}  (참고: v26 동일 carve-out 2082.78)")
    log(f" delta(RMSE - v26) on carve-out = {calib_bss - 2082.7818915293506:+.2f}")

    log("\n=== 모델 아티팩트 저장 ===")
    np.save(os.path.join(OUT_MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy())
    np.save(os.path.join(OUT_MODEL_DIR, "reference_labels.npy"), y_ref)
    shutil.copy(os.path.join(V26_MODEL_DIR, "retrieval_encoder.pt"), os.path.join(OUT_MODEL_DIR, "retrieval_encoder.pt"))
    shutil.copy(os.path.join(V26_MODEL_DIR, "numeric_prep.pkl"), os.path.join(OUT_MODEL_DIR, "numeric_prep.pkl"))
    shutil.copy(TRACKMAN_CONTEXT_PATH, os.path.join(OUT_MODEL_DIR, "trackman_context.pkl"))

    meta = {
        "catboost": {
            "columns": list(X_train_cb.columns),
            "cat_cols": CATBOOST_CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "loss_function": "RMSE",
        },
        "retrieval": meta_rt,
        "calibration": {"method": "platt_sigmoid", "a": a, "b": b},
        "blend_weight_catboost": W_CATBOOST,
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    result = {
        "catboost_rmse_calib_raw_bss": bss_score(cb_calib_raw, y_calib),
        "retrieval_calib_raw_bss": nca_bss,
        "blend_calibrated_bss": calib_bss,
        "delta_vs_v26_carveout": calib_bss - 2082.7818915293506,
        "platt": {"a": a, "b": b},
        "walkforward_evidence": {
            "fold0_blend_delta_vs_logloss": 11.28,
            "fold2_blend_delta_vs_logloss": 5.28,
        },
    }
    with open(os.path.join(OUT_DIR, "metrics_rmse_retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"\nSaved model dir: {OUT_MODEL_DIR}")
    log(f"Saved metrics: {OUT_DIR}/metrics_rmse_retrieval.json")


if __name__ == "__main__":
    main()
