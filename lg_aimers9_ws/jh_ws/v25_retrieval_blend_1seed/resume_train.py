"""retrieve_predict의 메모리 버그(청크별 유사도 행렬을 전부 쌓아뒀다가 합산
-> 쿼리 청크당 최대 22GB+ 점유, 극심한 스와핑)를 온라인 softmax 방식으로
고친 뒤 재개. CatBoost는 이미 학습/저장 완료(model/catboost_seed42.cbm,
train_final.py 첫 실행에서 완료됨, 데이터/split/seed 전부 동일하므로 재사용
안전) -- CatBoost 재학습(30분) 생략하고 encoder부터 다시 진행."""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

g = {"__file__": "train_final.py", "__name__": "resume"}
exec(open("train_final.py", encoding="utf-8").read().split("def main()")[0], g)

MODEL_DIR = g["MODEL_DIR"]
OUT_DIR = g["OUT_DIR"]
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("Load train data (전체) + control_risk_score 추가, 원재료3 제거...", flush=True)
df = g["load_data"]()
df = g["add_risk_score_drop_ingredients"](df)
print(f" shape={df.shape}", flush=True)

y_all = df[g["TARGET_COL"]]
train_sub_df, calib_df = train_test_split(
    df, test_size=0.05, stratify=y_all, random_state=g["SEED"],
)
train_sub_df = train_sub_df.reset_index(drop=True)
calib_df = calib_df.reset_index(drop=True)
print(f" train_sub={train_sub_df.shape}  calib={calib_df.shape}", flush=True)

# ---------- CatBoost: 이미 저장된 모델 로드 ----------
print("\n=== CatBoost 모델 로드(재학습 생략) ===", flush=True)
cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
y_calib = calib_df[g["TARGET_COL"]]
cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(MODEL_DIR, "catboost_seed42.cbm"))
cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]
print(" CatBoost 로드 완료, calib 예측 계산됨", flush=True)

# feature columns 재구성(저장용)
train_sub_cb_cols = list(g["build_catboost_features"](train_sub_df.head(1), cb_id_mappings).columns)

# ---------- Retrieval encoder ----------
print("\n=== Retrieval encoder(NCA) 학습 ===", flush=True)
nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
numeric_cols = [c for c in train_sub_df.columns if c not in [g["ID_COL"], g["TARGET_COL"]] + g["ALL_CAT_FOR_NN"]]
print(f" n_numeric={len(numeric_cols)} cardinalities={cardinalities}", flush=True)

cat_train_nn = g["encode_cats"](train_sub_df, nn_cat_mappings)
cat_calib_nn = g["encode_cats"](calib_df, nn_cat_mappings)
x_num_train, medians, scaler = g["prep_numeric_fit"](train_sub_df, numeric_cols)
x_num_calib = g["prep_numeric_transform"](calib_df, numeric_cols, medians, scaler)

t0 = time.time()
encoder, device = g["train_encoder"](
    cat_train_nn, x_num_train, train_sub_df[g["TARGET_COL"]], cardinalities, x_num_train.shape[1],
)
nca_elapsed = time.time() - t0
print(f" encoder 학습 완료 ({nca_elapsed:.1f}s)", flush=True)

rng = np.random.RandomState(g["SEED"])
n_train_sub = len(train_sub_df)
ref_size = n_train_sub  # REFERENCE_SIZE=None -> 전체 사용(사용자 요청)
ref_idx = rng.choice(n_train_sub, size=ref_size, replace=False)
print(f" 참조 집합: {ref_size}/{n_train_sub} (서브샘플 없음)", flush=True)

cat_ref_nn = {c: v[ref_idx] for c, v in cat_train_nn.items()}
x_num_ref = x_num_train[ref_idx]
y_ref = train_sub_df[g["TARGET_COL"]].values[ref_idx]

t0 = time.time()
z_ref = g["compute_embeddings"](encoder, device, cat_ref_nn, x_num_ref)
print(f" 참조 임베딩 계산 완료 ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
nca_calib_raw = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
print(f" calib retrieval 추론 완료 ({time.time()-t0:.1f}s)  <- 메모리 버그 수정 후 속도 확인용", flush=True)

# ---------- 블렌드 + calibration ----------
print("\n=== 블렌드 + Platt calibration ===", flush=True)
blend_calib_raw = (cb_calib_raw + nca_calib_raw) / 2
a_final, b_final = g["fit_platt_scaling"](blend_calib_raw, y_calib)
blend_calib_pred = g["apply_platt_scaling"](blend_calib_raw, a_final, b_final)

metrics = {
    "seed": g["SEED"],
    "nca_elapsed_sec": nca_elapsed,
    "reference_size": ref_size,
    "carveout_bss_cb_only": g["bss_score"](cb_calib_raw, y_calib),
    "carveout_bss_nca_only": g["bss_score"](nca_calib_raw, y_calib),
    "carveout_bss_blend_raw": g["bss_score"](blend_calib_raw, y_calib),
    "carveout_bss_blend_calibrated": g["bss_score"](blend_calib_pred, y_calib),
}
print(f" carve-out BSS: cb_only={metrics['carveout_bss_cb_only']:.2f}  "
      f"nca_only={metrics['carveout_bss_nca_only']:.2f}  "
      f"blend_calibrated={metrics['carveout_bss_blend_calibrated']:.2f}", flush=True)
with open(os.path.join(OUT_DIR, "metrics_v25.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "retrieval_encoder.pt"))
np.save(os.path.join(MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy().astype(np.float32))
np.save(os.path.join(MODEL_DIR, "reference_labels.npy"), y_ref.astype(np.float32))
joblib.dump({"medians": medians, "scaler": scaler}, os.path.join(MODEL_DIR, "numeric_prep.pkl"))

with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
    json.dump({
        "catboost": {
            "columns": train_sub_cb_cols,
            "cat_cols": g["CATBOOST_CAT_COLS"],
            "raw_id_cols": g["RAW_ID_COLS"],
            "id_mappings": cb_id_mappings,
        },
        "retrieval": {
            "numeric_cols": numeric_cols,
            "cat_cols": g["ALL_CAT_FOR_NN"],
            "cat_mappings": nn_cat_mappings,
            "cardinalities": cardinalities,
            "embed_dims": g["EMBED_DIMS"],
            "encoder_hidden": g["ENCODER_HIDDEN"],
            "embed_out_dim": g["EMBED_OUT_DIM"],
        },
        "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved all artifacts to {MODEL_DIR}", flush=True)
