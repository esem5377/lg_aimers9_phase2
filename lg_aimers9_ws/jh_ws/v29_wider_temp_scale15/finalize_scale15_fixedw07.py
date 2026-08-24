"""grid search 결과(w=0.5)가 랜덤 carve-out 기반이라 이 프로젝트의 확립된 원칙
("무작위 carve-out으로 가중치/레시피를 고르지 말 것, walk-forward만 신뢰")을
위반함 -- fold2 walk-forward 재확인 결과 w_catboost=0.7(v26과 동일 가중치)에서도
scale=15가 여전히 양성(+5.54)이었으므로, carve-out grid search 대신 v26의
walk-forward 검증된 가중치 0.7을 그대로 고정하고 calibration만 다시 fit한다.
"""
import json
import os
import time

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
V29_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v29_wider_temp_scale15\model"
V29_OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v29_wider_temp_scale15\output"

RETRIEVAL_TEMP_SCALE = 15.0
W_CATBOOST = 0.7  # v26과 동일(walk-forward 검증됨), carve-out grid search 결과(0.5)는 채택 안 함

g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "finalize"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


@torch.no_grad()
def retrieve_predict_scaled(model, device, cat_query, x_num_query, z_ref, y_ref, scale):
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


print("Load data + split (재현)...", flush=True)
df = g["load_data"]()
df = g["add_risk_score_drop_ingredients"](df)
y_all = df[g["TARGET_COL"]]
train_sub_df, calib_df = train_test_split(
    df, test_size=0.05, stratify=y_all, random_state=g["SEED"],
)
train_sub_df = train_sub_df.reset_index(drop=True)
calib_df = calib_df.reset_index(drop=True)
y_calib = calib_df[g["TARGET_COL"]]

print("CatBoost 예측 계산...", flush=True)
cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"))
cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

print(f"Retrieval 예측 계산 (scale={RETRIEVAL_TEMP_SCALE})...", flush=True)
nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
numeric_cols = [c for c in train_sub_df.columns if c not in [g["ID_COL"], g["TARGET_COL"]] + g["ALL_CAT_FOR_NN"]]
cat_calib_nn = g["encode_cats"](calib_df, nn_cat_mappings)
numeric_prep = joblib.load(os.path.join(V26_MODEL_DIR, "numeric_prep.pkl"))
medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]
x_num_calib = g["prep_numeric_transform"](calib_df, numeric_cols, medians, scaler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = g["RowEncoder"](cardinalities, len(numeric_cols) * 2).to(device)
encoder.load_state_dict(torch.load(os.path.join(V26_MODEL_DIR, "retrieval_encoder.pt"), map_location=device))
encoder.eval()
z_ref = torch.tensor(np.load(os.path.join(V26_MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
y_ref = np.load(os.path.join(V26_MODEL_DIR, "reference_labels.npy"))

t0 = time.time()
nca_calib_raw = retrieve_predict_scaled(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref, RETRIEVAL_TEMP_SCALE)
print(f" retrieval 추론 완료 ({time.time()-t0:.1f}s)", flush=True)

blend_raw = W_CATBOOST * cb_calib_raw + (1 - W_CATBOOST) * nca_calib_raw
a_final, b_final = g["fit_platt_scaling"](blend_raw, y_calib)
blend_pred = g["apply_platt_scaling"](blend_raw, a_final, b_final)
bss = bss_score(blend_pred, y_calib)
print(f"\ncarve-out calibrated BSS(w={W_CATBOOST}, scale={RETRIEVAL_TEMP_SCALE})={bss:.2f}  "
      f"a={a_final}  b={b_final}", flush=True)
print(" (참고용 숫자일 뿐 -- 채택 근거는 fold2 walk-forward w=0.7 delta=+5.54)", flush=True)

with open(os.path.join(V29_OUT_DIR, "metrics_v29_final.json"), "w", encoding="utf-8") as f:
    json.dump({
        "retrieval_temp_scale": RETRIEVAL_TEMP_SCALE,
        "w_catboost": W_CATBOOST,
        "w_catboost_source": "v26 walk-forward validated weight (NOT random carve-out grid search)",
        "carveout_bss_blend_calibrated": bss,
        "calibration_a": a_final,
        "calibration_b": b_final,
        "walkforward_fold2_delta_at_w07": 5.54,
        "walkforward_fold0_delta_at_50_50": 21.94,
        "walkforward_fold2_delta_at_50_50": 8.60,
        "v26_reference_carveout_bss": 2082.78,
    }, f, indent=2, ensure_ascii=False)

with open(os.path.join(V25_DIR, "model", "feature_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
meta["blend_weight_catboost"] = W_CATBOOST
meta["calibration"] = {"method": "platt_sigmoid", "a": a_final, "b": b_final}
meta["retrieval_temp_scale"] = RETRIEVAL_TEMP_SCALE

with open(os.path.join(V29_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nSaved feature_meta.json to {V29_MODEL_DIR}", flush=True)
