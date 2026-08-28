"""v26(1023점, CatBoost:retrieval=0.7:0.3)의 이미 학습된 모델(CatBoost/encoder/
참조임베딩)을 그대로 재사용(재학습 없음), calibration만 Platt(2파라미터)에서
Beta calibration(3파라미터, logit = a*ln(s) - b*ln(1-s) + c)으로 교체.

배경: CatBoost 단독 fold0/fold2 스크리닝에서 Beta가 Platt보다 소폭 나빴음
(fold0 -5.38/fold2 -0.92, 둘 다 음수) -- 다만 이건 블렌드가 아니라 CatBoost
단독 기준이었고, 사용자가 재검증 생략하고 실제 v26 블렌드(1023점)에 바로
적용해서 실제 제출까지 검증하기로 함(2026-08-25).
"""
import json
import os

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

V26_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07"
V31_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v31_beta_calibration_w07\model"
V31_OUT_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v31_beta_calibration_w07\output"
os.makedirs(V31_OUT_DIR, exist_ok=True)

W_CATBOOST = 0.7  # v26과 동일, 블렌드 가중치는 안 건드림(calibration만 교체)

g = {"__file__": os.path.join(V26_DIR, "..", "v25_retrieval_blend_1seed", "train_final.py"), "__name__": "refit"}
V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)


def fit_beta(raw_p, y, eps=1e-6):
    s = np.clip(np.asarray(raw_p), eps, 1 - eps)
    X = np.column_stack([np.log(s), -np.log(1 - s)])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(X, y)
    a, b = lr.coef_[0]
    c = lr.intercept_[0]
    return float(a), float(b), float(c)


def apply_beta(raw_p, a, b, c, eps=1e-6):
    s = np.clip(np.asarray(raw_p), eps, 1 - eps)
    logit = a * np.log(s) - b * np.log(1 - s) + c
    return 1.0 / (1.0 + np.exp(-logit))


print("Load data + split (v25/v26와 동일 split 재현)...", flush=True)
df = g["load_data"]()
df = g["add_risk_score_drop_ingredients"](df)
y_all = df[g["TARGET_COL"]]
train_sub_df, calib_df = train_test_split(
    df, test_size=0.05, stratify=y_all, random_state=g["SEED"],
)
train_sub_df = train_sub_df.reset_index(drop=True)
calib_df = calib_df.reset_index(drop=True)
y_calib = calib_df[g["TARGET_COL"]]

print("CatBoost 예측 계산 (v26 모델 그대로 로드, 재학습 없음)...", flush=True)
cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(V31_MODEL_DIR, "catboost_seed42.cbm"))
cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

print("Retrieval 예측 계산 (v26 encoder+참조임베딩 그대로, 재학습 없음)...", flush=True)
nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
numeric_cols = [c for c in train_sub_df.columns if c not in [g["ID_COL"], g["TARGET_COL"]] + g["ALL_CAT_FOR_NN"]]
cat_calib_nn = g["encode_cats"](calib_df, nn_cat_mappings)
numeric_prep = joblib.load(os.path.join(V31_MODEL_DIR, "numeric_prep.pkl"))
medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]
x_num_calib = g["prep_numeric_transform"](calib_df, numeric_cols, medians, scaler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = g["RowEncoder"](cardinalities, len(numeric_cols) * 2).to(device)
encoder.load_state_dict(torch.load(os.path.join(V31_MODEL_DIR, "retrieval_encoder.pt"), map_location=device))
encoder.eval()
z_ref = torch.tensor(np.load(os.path.join(V31_MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
y_ref = np.load(os.path.join(V31_MODEL_DIR, "reference_labels.npy"))
nca_calib_raw = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)

print(f"\n블렌드(w_catboost={W_CATBOOST})...", flush=True)
blend_raw = W_CATBOOST * cb_calib_raw + (1 - W_CATBOOST) * nca_calib_raw

# Platt sanity check (v26의 기존 a/b: 4.774794369719763 / -2.40192142901094와 일치해야 함)
a_platt, b_platt = g["fit_platt_scaling"](blend_raw, y_calib)
platt_pred = g["apply_platt_scaling"](blend_raw, a_platt, b_platt)
platt_bss = g["bss_score"](platt_pred, y_calib)
print(f" [sanity check] Platt: a={a_platt}  b={b_platt}  carve-out bss={platt_bss:.2f}", flush=True)
print(f" (v26 기존 값과 비교: a=4.774794369719763 b=-2.40192142901094)", flush=True)

# Beta calibration
a_beta, b_beta, c_beta = fit_beta(blend_raw, y_calib)
beta_pred = apply_beta(blend_raw, a_beta, b_beta, c_beta)
beta_bss = g["bss_score"](beta_pred, y_calib)
print(f" Beta : a={a_beta:.4f}  b={b_beta:.4f}  c={c_beta:.4f}  carve-out bss={beta_bss:.2f}", flush=True)
print(f" delta(Beta - Platt) on carve-out = {beta_bss - platt_bss:+.2f}", flush=True)

with open(os.path.join(V31_OUT_DIR, "metrics_v31_beta.json"), "w", encoding="utf-8") as f:
    json.dump({
        "w_catboost": W_CATBOOST,
        "platt_carveout_bss": platt_bss,
        "platt_params": {"a": a_platt, "b": b_platt},
        "beta_carveout_bss": beta_bss,
        "beta_params": {"a": a_beta, "b": b_beta, "c": c_beta},
        "delta_beta_vs_platt_carveout": beta_bss - platt_bss,
    }, f, indent=2, ensure_ascii=False)

with open(os.path.join(V31_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
meta["calibration"] = {"method": "beta", "a": a_beta, "b": b_beta, "c": c_beta}
with open(os.path.join(V31_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nSaved feature_meta.json (calibration=beta) to {V31_MODEL_DIR}", flush=True)
