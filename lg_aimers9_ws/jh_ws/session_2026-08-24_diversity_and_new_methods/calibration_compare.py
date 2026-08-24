"""v26(987+CatBoost:retrieval=0.7:0.3) 재학습 없이, calibration 방법만
비교. 8/18 발견("sigmoid calibration만 추가해도 실제 리더보드 +70")이
근거 -- BSS(Brier)가 calibration 품질에 극도로 민감했던 전례를 이어서,
지금 production의 Platt(sigmoid, 파라미터 2개)를 더 유연한 Isotonic으로
바꾸면 이득이 있는지 확인.

방법론: calib_df(73,755행, production 5% carve-out)를 다시 80:20으로
쪼개 calib_fit(calibration 학습용)/calib_holdout(정직한 평가용)으로 분리.
같은 데이터로 fit&eval하면 Isotonic이 유연해서 유리하게 왜곡될 수 있어
반드시 분리 -- Platt/Isotonic 둘 다 이 규칙을 동일하게 적용해 공정 비교.

비교 대상:
  A. Platt-on-blend(현재 production 방식): blend_raw에 Platt 1회
  B. Isotonic-on-blend: blend_raw에 Isotonic 1회
  C. Platt-per-model then blend: CatBoost/retrieval 각각 Platt 후 blend
  D. Isotonic-per-model then blend: CatBoost/retrieval 각각 Isotonic 후 blend
"""
import json
import os

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
OUT_PATH = os.path.join(os.path.dirname(__file__), "calibration_compare_results.json")

W_CATBOOST = 0.7
SEED = 42


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
y_calib_full = calib_df[g["TARGET_COL"]]

print("CatBoost 예측 계산(calib 전체)...", flush=True)
cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"))
cb_raw_full = cb_model.predict_proba(X_calib_cb)[:, 1]

print("Retrieval 예측 계산(calib 전체, 몇 분 소요)...", flush=True)
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
nca_raw_full = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)

print("\ncalib_df를 fit/holdout(80:20)으로 분리...", flush=True)
idx_all = np.arange(len(calib_df))
idx_fit, idx_hold = train_test_split(idx_all, test_size=0.20, stratify=y_calib_full, random_state=SEED)

cb_fit, cb_hold = cb_raw_full[idx_fit], cb_raw_full[idx_hold]
nca_fit, nca_hold = nca_raw_full[idx_fit], nca_raw_full[idx_hold]
y_fit = y_calib_full.iloc[idx_fit].values
y_hold = y_calib_full.iloc[idx_hold].values
print(f" fit={len(idx_fit)}  holdout={len(idx_hold)}", flush=True)

blend_fit = W_CATBOOST * cb_fit + (1 - W_CATBOOST) * nca_fit
blend_hold = W_CATBOOST * cb_hold + (1 - W_CATBOOST) * nca_hold

results = {}

print("\n[A] Platt-on-blend (현재 production 방식)...", flush=True)
a, b = fit_platt(blend_fit, y_fit)
pred_a = apply_platt(blend_hold, a, b)
results["A_platt_on_blend"] = bss_score(pred_a, y_hold)
print(f"  holdout bss={results['A_platt_on_blend']:.2f}", flush=True)

print("[B] Isotonic-on-blend...", flush=True)
iso_blend = IsotonicRegression(out_of_bounds="clip")
iso_blend.fit(blend_fit, y_fit)
pred_b = iso_blend.predict(blend_hold)
results["B_isotonic_on_blend"] = bss_score(pred_b, y_hold)
print(f"  holdout bss={results['B_isotonic_on_blend']:.2f}", flush=True)

print("[C] Platt-per-model then blend...", flush=True)
a_cb, b_cb = fit_platt(cb_fit, y_fit)
a_nca, b_nca = fit_platt(nca_fit, y_fit)
cb_hold_cal = apply_platt(cb_hold, a_cb, b_cb)
nca_hold_cal = apply_platt(nca_hold, a_nca, b_nca)
pred_c = W_CATBOOST * cb_hold_cal + (1 - W_CATBOOST) * nca_hold_cal
results["C_platt_per_model"] = bss_score(pred_c, y_hold)
print(f"  holdout bss={results['C_platt_per_model']:.2f}", flush=True)

print("[D] Isotonic-per-model then blend...", flush=True)
iso_cb = IsotonicRegression(out_of_bounds="clip")
iso_cb.fit(cb_fit, y_fit)
iso_nca = IsotonicRegression(out_of_bounds="clip")
iso_nca.fit(nca_fit, y_fit)
cb_hold_iso = iso_cb.predict(cb_hold)
nca_hold_iso = iso_nca.predict(nca_hold)
pred_d = W_CATBOOST * cb_hold_iso + (1 - W_CATBOOST) * nca_hold_iso
results["D_isotonic_per_model"] = bss_score(pred_d, y_hold)
print(f"  holdout bss={results['D_isotonic_per_model']:.2f}", flush=True)

print("\n=== SUMMARY (calib_holdout bss, 20%, w_catboost=0.7 고정) ===", flush=True)
for k, v in results.items():
    print(f"  {k}: {v:.2f}  (delta vs A: {v - results['A_platt_on_blend']:+.2f})", flush=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT_PATH}", flush=True)
