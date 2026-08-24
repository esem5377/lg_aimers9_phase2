"""v26(CatBoost:retrieval=0.7:0.3, calibrated BSS 2082.78)의 이미 학습된 자산
(CatBoost/encoder/참조임베딩)을 그대로 재사용 -- 재학습 없음.

배경(2026-08-25): retrieval의 추론 온도(scale)를 학습된 값보다 넓게(8~15배)
키우면 fold0/fold2(계절 분리 walk-forward) 둘 다 강한 양성 신호(fold0 scale=15
delta +21.94, fold2 scale=15 delta +8.60, 50:50 블렌드 기준)를 보였음. 이걸
실제 프로덕션(0.7:0.3 가중치)에 반영하되, retrieval의 raw 예측 분포가 scale로
바뀌었으니 블렌드 가중치도 다시 그리드서치(calibrated BSS 기준, 8/22 세션
확정 원칙)해서 재검증.

retrieve_predict만 temp*RETRIEVAL_TEMP_SCALE로 수정, 나머지는 finalize_w07.py와
동일 패턴.
"""
import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
V29_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v29_wider_temp_scale15\model"
V29_OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v29_wider_temp_scale15\output"
os.makedirs(V29_MODEL_DIR, exist_ok=True)
os.makedirs(V29_OUT_DIR, exist_ok=True)

RETRIEVAL_TEMP_SCALE = 15.0
W_CANDIDATES = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "finalize"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


@torch.no_grad()
def retrieve_predict_scaled(model, device, cat_query, x_num_query, z_ref, y_ref, scale):
    """train_final.py의 retrieve_predict와 동일(온라인 softmax 누적)하되
    temp에 scale을 곱해서 더 넓게 평균낸다."""
    REFERENCE_CHUNK = g["REFERENCE_CHUNK"]
    QUERY_CHUNK = g["QUERY_CHUNK"]
    model.eval()
    base_temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)
    temp = (base_temp * scale).to(device)
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


print("Load data + split (재현, v25/v26과 동일 seed/split)...", flush=True)
df = g["load_data"]()
df = g["add_risk_score_drop_ingredients"](df)
y_all = df[g["TARGET_COL"]]
train_sub_df, calib_df = train_test_split(
    df, test_size=0.05, stratify=y_all, random_state=g["SEED"],
)
train_sub_df = train_sub_df.reset_index(drop=True)
calib_df = calib_df.reset_index(drop=True)
y_calib = calib_df[g["TARGET_COL"]]
print(f" calib_df={calib_df.shape}", flush=True)

print("CatBoost 예측 계산 (v26 모델 재사용, 변경 없음)...", flush=True)
cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"))
cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]
print(f" cb_only bss={bss_score(cb_calib_raw, y_calib):.2f}", flush=True)

print(f"\nRetrieval 예측 계산 (v26 encoder+참조임베딩 재사용, scale={RETRIEVAL_TEMP_SCALE})...", flush=True)
nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
numeric_cols = [c for c in train_sub_df.columns if c not in [g["ID_COL"], g["TARGET_COL"]] + g["ALL_CAT_FOR_NN"]]
cat_calib_nn = g["encode_cats"](calib_df, nn_cat_mappings)
numeric_prep = joblib.load(os.path.join(V26_MODEL_DIR, "numeric_prep.pkl"))
medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]
x_num_calib = g["prep_numeric_transform"](calib_df, numeric_cols, medians, scaler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" device={device}", flush=True)
encoder = g["RowEncoder"](cardinalities, len(numeric_cols) * 2).to(device)
encoder.load_state_dict(torch.load(os.path.join(V26_MODEL_DIR, "retrieval_encoder.pt"), map_location=device))
encoder.eval()
z_ref = torch.tensor(np.load(os.path.join(V26_MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
y_ref = np.load(os.path.join(V26_MODEL_DIR, "reference_labels.npy"))
print(f" reference_size={len(y_ref)}", flush=True)

import time
t0 = time.time()
nca_calib_raw = retrieve_predict_scaled(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref, RETRIEVAL_TEMP_SCALE)
print(f" retrieval 추론 완료 ({time.time()-t0:.1f}s), nca_only bss={bss_score(nca_calib_raw, y_calib):.2f}", flush=True)

print(f"\n블렌드 가중치 그리드서치 (calibrated BSS 기준, candidates={W_CANDIDATES})...", flush=True)
best_w, best_bss, best_ab = None, -1, None
grid_results = {}
for w in W_CANDIDATES:
    blend_raw = w * cb_calib_raw + (1 - w) * nca_calib_raw
    a, b = g["fit_platt_scaling"](blend_raw, y_calib)
    pred = g["apply_platt_scaling"](blend_raw, a, b)
    bss = bss_score(pred, y_calib)
    grid_results[str(w)] = bss
    print(f"  w_catboost={w}: calibrated BSS={bss:.2f}", flush=True)
    if bss > best_bss:
        best_w, best_bss, best_ab = w, bss, (a, b)

print(f"\n최적: w_catboost={best_w}  calibrated BSS={best_bss:.2f}  (v26 기존={2082.78})", flush=True)
a_final, b_final = best_ab

with open(os.path.join(V29_OUT_DIR, "metrics_v29.json"), "w", encoding="utf-8") as f:
    json.dump({
        "retrieval_temp_scale": RETRIEVAL_TEMP_SCALE,
        "w_catboost": best_w,
        "grid_results": grid_results,
        "carveout_bss_cb_only": bss_score(cb_calib_raw, y_calib),
        "carveout_bss_nca_only": bss_score(nca_calib_raw, y_calib),
        "carveout_bss_blend_calibrated": best_bss,
        "calibration_a": a_final,
        "calibration_b": b_final,
        "v26_reference_bss": 2082.78,
    }, f, indent=2, ensure_ascii=False)

with open(os.path.join(V25_DIR, "model", "feature_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
meta["blend_weight_catboost"] = best_w
meta["calibration"] = {"method": "platt_sigmoid", "a": a_final, "b": b_final}
meta["retrieval_temp_scale"] = RETRIEVAL_TEMP_SCALE

with open(os.path.join(V29_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nSaved feature_meta.json to {V29_MODEL_DIR}", flush=True)
