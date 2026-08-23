"""이미 학습된 CatBoost + retrieval encoder를 재학습 없이 재사용,
블렌드 가중치(w*catboost + (1-w)*retrieval)만 그리드서치.
calib carve-out(같은 SEED=42 split이라 train_final.py/resume_train.py와
동일한 calib_df 재현됨)에서 raw 예측 재계산 -> 가중치별 Platt fit -> BSS 비교.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

g = {"__file__": "train_final.py", "__name__": "blendsearch"}
exec(open("train_final.py", encoding="utf-8").read().split("def main()")[0], g)

MODEL_DIR = g["MODEL_DIR"]

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
print(f" calib={calib_df.shape}", flush=True)

print("CatBoost 예측 계산...", flush=True)
cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(MODEL_DIR, "catboost_seed42.cbm"))
cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

print("Retrieval 예측 계산(저장된 encoder+참조 임베딩 재사용)...", flush=True)
nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
numeric_cols = [c for c in train_sub_df.columns if c not in [g["ID_COL"], g["TARGET_COL"]] + g["ALL_CAT_FOR_NN"]]
cat_calib_nn = g["encode_cats"](calib_df, nn_cat_mappings)
numeric_prep = joblib.load(os.path.join(MODEL_DIR, "numeric_prep.pkl"))
medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]
x_num_calib = g["prep_numeric_transform"](calib_df, numeric_cols, medians, scaler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = g["RowEncoder"](cardinalities, len(numeric_cols) * 2).to(device)
encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, "retrieval_encoder.pt"), map_location=device))
encoder.eval()

z_ref = torch.tensor(np.load(os.path.join(MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
y_ref = np.load(os.path.join(MODEL_DIR, "reference_labels.npy"))

nca_calib_raw = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
print(" 완료", flush=True)

print("\n=== 블렌드 가중치 그리드서치 (w*catboost + (1-w)*retrieval) ===", flush=True)
results = []
for w in np.arange(0.0, 1.01, 0.1):
    blend_raw = w * cb_calib_raw + (1 - w) * nca_calib_raw
    a, b = g["fit_platt_scaling"](blend_raw, y_calib)
    blend_pred = g["apply_platt_scaling"](blend_raw, a, b)
    bss = g["bss_score"](blend_pred, y_calib)
    results.append((round(w, 1), bss))
    print(f"  w={w:.1f}  (catboost={w:.1f} / retrieval={1-w:.1f})  calibrated BSS={bss:.2f}", flush=True)

best_w, best_bss = max(results, key=lambda r: r[1])
print(f"\n최적 가중치: w={best_w} (catboost={best_w}/retrieval={1-best_w}), BSS={best_bss:.2f}", flush=True)
print(f"참고: w=0.5(기존 v25) BSS={dict(results)[0.5]:.2f}, w=1.0(catboost단독) BSS={dict(results)[1.0]:.2f}, w=0.0(retrieval단독) BSS={dict(results)[0.0]:.2f}", flush=True)

with open("blend_weight_search_results.json", "w", encoding="utf-8") as f:
    json.dump({"results": results, "best_w": best_w, "best_bss": best_bss}, f, indent=2, ensure_ascii=False)
print("Saved: blend_weight_search_results.json", flush=True)
