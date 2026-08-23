"""v25(CatBoost+retrieval 50:50)의 이미 학습된 모델(CatBoost/encoder/참조
임베딩)을 그대로 재사용, 블렌드 가중치만 0.7:0.3(CatBoost:retrieval)로
바꿔 calibration을 새로 fit. 재학습 없음(blend_weight_search.py 그리드서치
결과 w=0.7이 최적, calibrated BSS 2071.52 -> 2082.78, +11.26).
"""
import json
import os

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
V26_OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\output"
os.makedirs(V26_OUT_DIR, exist_ok=True)

W_CATBOOST = 0.7

g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "finalize"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)

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

print("Retrieval 예측 계산...", flush=True)
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
nca_calib_raw = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)

print(f"\n블렌드(w_catboost={W_CATBOOST}) + Platt calibration...", flush=True)
blend_raw = W_CATBOOST * cb_calib_raw + (1 - W_CATBOOST) * nca_calib_raw
a_final, b_final = g["fit_platt_scaling"](blend_raw, y_calib)
blend_pred = g["apply_platt_scaling"](blend_raw, a_final, b_final)
bss = g["bss_score"](blend_pred, y_calib)
print(f" carve-out calibrated BSS(w={W_CATBOOST})={bss:.2f}  a={a_final}  b={b_final}", flush=True)

with open(os.path.join(V26_OUT_DIR, "metrics_v26.json"), "w", encoding="utf-8") as f:
    json.dump({
        "w_catboost": W_CATBOOST,
        "carveout_bss_blend_calibrated": bss,
        "calibration_a": a_final,
        "calibration_b": b_final,
    }, f, indent=2, ensure_ascii=False)

with open(os.path.join(V25_DIR, "model", "feature_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
meta["blend_weight_catboost"] = W_CATBOOST
meta["calibration"] = {"method": "platt_sigmoid", "a": a_final, "b": b_final}

with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nSaved feature_meta.json (blend_weight_catboost={W_CATBOOST}) to {V26_MODEL_DIR}", flush=True)
