"""v26의 이미 학습된 모델(CatBoost/encoder/참조 임베딩)을 그대로 재사용,
calibration 방식만 교체: "블렌드 후 Platt 1회"(v26) -> "CatBoost/retrieval
각각 isotonic 개별 보정 후 0.7:0.3 블렌드"(v28). 재학습 없음.
(calibration_compare.py에서 20% 정직한 holdout으로 4가지 방식을 비교해
이 방식이 +15.23으로 최고였음을 확인 후 production 규모로 재적용.)

Platt처럼 (a,b) 스칼라 2개로 재현 불가능한 isotonic이라, sklearn 객체를
그대로 pickle하지 않고(8/16 sklearn 버전 크로스 호환 사고 전례) X/y
threshold 배열만 JSON으로 저장 -- script.py에서는 sklearn 없이 순수
numpy(np.interp, sklearn의 out_of_bounds='clip'과 동일한 선형보간+클리핑)로
재현.
"""
import json
import os

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V28_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v28_isotonic_per_model\model"
V28_OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v28_isotonic_per_model\output"
os.makedirs(V28_OUT_DIR, exist_ok=True)

W_CATBOOST = 0.7


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


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
cb_model.load_model(os.path.join(V28_MODEL_DIR, "catboost_seed42.cbm"))
cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

print("Retrieval 예측 계산...", flush=True)
nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
numeric_cols = [c for c in train_sub_df.columns if c not in [g["ID_COL"], g["TARGET_COL"]] + g["ALL_CAT_FOR_NN"]]
cat_calib_nn = g["encode_cats"](calib_df, nn_cat_mappings)
numeric_prep = joblib.load(os.path.join(V28_MODEL_DIR, "numeric_prep.pkl"))
medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]
x_num_calib = g["prep_numeric_transform"](calib_df, numeric_cols, medians, scaler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = g["RowEncoder"](cardinalities, len(numeric_cols) * 2).to(device)
encoder.load_state_dict(torch.load(os.path.join(V28_MODEL_DIR, "retrieval_encoder.pt"), map_location=device))
encoder.eval()
z_ref = torch.tensor(np.load(os.path.join(V28_MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
y_ref = np.load(os.path.join(V28_MODEL_DIR, "reference_labels.npy"))
nca_calib_raw = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)

print(f"\nCatBoost/retrieval 각각 isotonic 개별 fit (calib_df 전체 {len(calib_df)}행)...", flush=True)
iso_cb = IsotonicRegression(out_of_bounds="clip")
iso_cb.fit(cb_calib_raw, y_calib)
iso_nca = IsotonicRegression(out_of_bounds="clip")
iso_nca.fit(nca_calib_raw, y_calib)

cb_calib_cal = iso_cb.predict(cb_calib_raw)
nca_calib_cal = iso_nca.predict(nca_calib_raw)
blend_pred = W_CATBOOST * cb_calib_cal + (1 - W_CATBOOST) * nca_calib_cal
bss = bss_score(blend_pred, y_calib)
print(f" carve-out isotonic-per-model blend BSS(같은 calib_df로 fit+eval, v25/v26과 동일 관례)={bss:.2f}", flush=True)

with open(os.path.join(V28_OUT_DIR, "metrics_v28.json"), "w", encoding="utf-8") as f:
    json.dump({
        "w_catboost": W_CATBOOST,
        "carveout_bss_blend_isotonic_per_model": bss,
        "n_calib": len(calib_df),
    }, f, indent=2, ensure_ascii=False)

with open(os.path.join(V25_DIR, "model", "feature_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
meta["blend_weight_catboost"] = W_CATBOOST
meta["calibration"] = None  # 기존 platt-on-blend 구조 사용 안 함
meta["calibration_isotonic_per_model"] = {
    "catboost": {
        "x_thresholds": iso_cb.X_thresholds_.tolist(),
        "y_thresholds": iso_cb.y_thresholds_.tolist(),
    },
    "retrieval": {
        "x_thresholds": iso_nca.X_thresholds_.tolist(),
        "y_thresholds": iso_nca.y_thresholds_.tolist(),
    },
}

with open(os.path.join(V28_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nSaved feature_meta.json (isotonic per-model, blend_weight_catboost={W_CATBOOST}) to {V28_MODEL_DIR}", flush=True)
