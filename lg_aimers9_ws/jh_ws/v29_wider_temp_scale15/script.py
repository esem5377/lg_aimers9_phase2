# script.py
# v26(1023점, CatBoost:retrieval=0.7:0.3)의 CatBoost/encoder/참조 임베딩을
# 그대로 재사용(재학습 없음) -- retrieval 추론 온도(temperature)만 학습된
# 값의 15배로 넓힘(fold0/fold2 계절분리 walk-forward에서 검증, fold0 delta
# +21.94, fold2(production weight 0.7 기준) delta +5.54). 블렌드 가중치는
# v26과 동일 0.7 유지(랜덤 carve-out 그리드서치는 이 프로젝트에서 신뢰 안 함,
# walk-forward 결과만 근거). scale은 feature_meta.json의 retrieval_temp_scale
# 필드에서 읽음(없으면 1.0, 구버전 호환).
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier

ID_COL = "row_id"
TARGET_COL = "control_success"
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 4000


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def attach_trackman_context(df, context):
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


def build_catboost_features(df, meta_cb):
    X = df.drop(columns=[ID_COL])
    for c in meta_cb["raw_id_cols"]:
        mapping = meta_cb["id_mappings"][c]
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in meta_cb["cat_cols"]:
        X[c] = X[c].astype(str)
    return X[meta_cb["columns"]]


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
def retrieve_predict(model, device, cat_query, x_num_query, z_ref, y_ref, temp_scale=1.0):
    """참조 청크를 한 번만 순회하는 온라인(스트리밍) softmax 누적
    (메모리 O(query_chunk x REFERENCE_CHUNK) 고정, flash-attention과 동일 원리).
    temp_scale: 학습된 온도(log_temp)에 곱하는 배수 -- 1.0보다 크면 더 넓은
    이웃을 평균내 노이즈를 줄인다(walk-forward로 검증됨, v29 참고)."""
    model.eval()
    temp = (torch.exp(model.log_temp).clamp(min=1e-3, max=10.0) * temp_scale).to(device)
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


def apply_calibration(raw_p, calib):
    if calib is None:
        return raw_p
    a, b = calib["a"], calib["b"]
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
    CONTEXT_PATH = os.path.join(MODEL_DIR, "trackman_context.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("Load model + meta...")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    meta_cb = meta["catboost"]
    meta_rt = meta["retrieval"]
    calib = meta.get("calibration")
    w_cb = meta.get("blend_weight_catboost", 0.5)
    retrieval_temp_scale = meta.get("retrieval_temp_scale", 1.0)

    cb_model = CatBoostClassifier()
    cb_model.load_model(os.path.join(MODEL_DIR, "catboost_seed42.cbm"))

    numeric_prep = joblib.load(os.path.join(MODEL_DIR, "numeric_prep.pkl"))
    medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]

    n_numeric = len(meta_rt["numeric_cols"]) * 2  # 원본 + isna 플래그
    encoder = RowEncoder(
        meta_rt["cardinalities"], meta_rt["embed_dims"], meta_rt["encoder_hidden"],
        meta_rt["embed_out_dim"], 0.0, n_numeric,
    ).to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR, "retrieval_encoder.pt"), map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    z_ref = torch.tensor(np.load(os.path.join(MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
    y_ref = np.load(os.path.join(MODEL_DIR, "reference_labels.npy"))

    context = joblib.load(CONTEXT_PATH)
    print(f" OK. reference_size={len(y_ref)}  calibration={calib}  blend_weight_catboost={w_cb}"
          f"  retrieval_temp_scale={retrieval_temp_scale}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    test = attach_trackman_context(test, context)
    test = add_risk_score_drop_ingredients(test)
    ids = test[ID_COL].tolist()

    print("CatBoost inference...")
    X_cb = build_catboost_features(test, meta_cb)
    n_oov_pitcher = int((X_cb["pitcher_id"] == -1).sum())
    n_oov_batter = int((X_cb["batter_id"] == -1).sum())
    print(f" features={X_cb.shape[1]}  OOV pitcher_id={n_oov_pitcher}  OOV batter_id={n_oov_batter}")
    cb_raw = cb_model.predict_proba(X_cb)[:, 1] if len(X_cb) else np.array([])

    print("Retrieval inference...")
    if len(test):
        cat_query = encode_cats(test, meta_rt["cat_mappings"])
        x_num_query = prep_numeric_transform(test, meta_rt["numeric_cols"], medians, scaler)
        nca_raw = retrieve_predict(encoder, device, cat_query, x_num_query, z_ref, y_ref, retrieval_temp_scale)
    else:
        nca_raw = np.array([])

    print("Build submission...")
    if len(test):
        blend_raw = w_cb * cb_raw + (1 - w_cb) * nca_raw
        preds = apply_calibration(blend_raw, calib)
    else:
        preds = []
    print(f" preds={len(preds)}")

    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
