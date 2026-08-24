"""v4 = jh_ws v26(1023점, 1시드 CB + retrieval encoder, w_cb=0.7) 레시피에서
CatBoost만 6시드 배깅으로 교체(retrieval encoder는 검증된 v26 것 그대로 재사용,
재학습 안 함). carve-out calibrated BSS: 2082.78(v26) -> 2085.15(v4, w_cb=0.70
그대로 최적으로 재확인, +2.37). 최종 제출 패키지를 조립한다.
"""
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import reuse_v26_retrieval_with_cb6 as m

SEED = 42
W_CB = 0.70
OUT_MODEL_DIR = os.path.join(m._BASE, "model_v4_cb6_retrieval")
SUB_DIR = os.path.join(os.path.dirname(m._BASE), "work", "submissions", "submit_v13_cb6_retrieval")


def log(msg):
    print(msg, flush=True)


def main():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(SUB_DIR, "model"), exist_ok=True)

    log("Load train data...")
    df = m.load_data()
    df = m.add_risk_score_drop_ingredients(df)
    y_all = df[m.TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[m.TARGET_COL]

    id_mappings = {c: {v: i for i, v in enumerate(sorted(train_sub_df[c].astype(str).unique()))} for c in m.RAW_ID_COLS}
    X_calib_cb = m.build_catboost_features(calib_df, id_mappings)

    log("Load CatBoost 6시드...")
    cb_calib_raws = []
    for seed in m.CB_SEEDS:
        mdl = CatBoostClassifier()
        mdl.load_model(os.path.join(m.CB6_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cb_calib_raws.append(mdl.predict_proba(X_calib_cb)[:, 1])
    cb_calib_raw = np.mean(cb_calib_raws, axis=0)

    log("Load v26 retrieval encoder + 임베딩 재계산...")
    with open(os.path.join(m.V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        v26_meta = json.load(f)
    meta_rt = v26_meta["retrieval"]
    numeric_prep = joblib.load(os.path.join(m.V26_MODEL_DIR, "numeric_prep.pkl"))
    medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_numeric = len(meta_rt["numeric_cols"]) * 2
    encoder = m.RowEncoder(
        meta_rt["cardinalities"], meta_rt["embed_dims"], meta_rt["encoder_hidden"],
        meta_rt["embed_out_dim"], 0.0, n_numeric,
    ).to(device)
    encoder.load_state_dict(torch.load(os.path.join(m.V26_MODEL_DIR, "retrieval_encoder.pt"), map_location=device, weights_only=True))
    encoder.eval()

    cat_train_nn = m.encode_cats(train_sub_df, meta_rt["cat_mappings"])
    cat_calib_nn = m.encode_cats(calib_df, meta_rt["cat_mappings"])
    x_num_train = m.prep_numeric_transform(train_sub_df, meta_rt["numeric_cols"], medians, scaler)
    x_num_calib = m.prep_numeric_transform(calib_df, meta_rt["numeric_cols"], medians, scaler)

    z_ref = m.compute_embeddings(encoder, device, cat_train_nn, x_num_train)
    y_ref = train_sub_df[m.TARGET_COL].values.astype(np.float32)
    nca_calib_raw = m.retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)

    log("Fit calibration @ w_cb=0.70 (grid search로 확인된 최적값)...")
    raw = W_CB * cb_calib_raw + (1 - W_CB) * nca_calib_raw
    a, b = m.fit_platt_scaling(raw, y_calib)
    calib_pred = m.apply_platt_scaling(raw, a, b)
    final_bss = m.bss_score(calib_pred, y_calib)
    log(f" carve-out calibrated BSS = {final_bss:.2f}  (v26 대비 {final_bss-2082.7818915293506:+.2f})")

    log("모델 아티팩트 저장...")
    for seed in m.CB_SEEDS:
        shutil.copy(os.path.join(m.CB6_MODEL_DIR, f"catboost_seed{seed}.cbm"),
                    os.path.join(OUT_MODEL_DIR, f"catboost_seed{seed}.cbm"))
    torch.save(encoder.state_dict(), os.path.join(OUT_MODEL_DIR, "retrieval_encoder.pt"))
    np.save(os.path.join(OUT_MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy().astype(np.float32))
    np.save(os.path.join(OUT_MODEL_DIR, "reference_labels.npy"), y_ref)
    joblib.dump({"medians": medians, "scaler": scaler}, os.path.join(OUT_MODEL_DIR, "numeric_prep.pkl"))
    shutil.copy(m.TRACKMAN_CONTEXT_PATH, os.path.join(OUT_MODEL_DIR, "trackman_context.pkl"))

    feature_meta = {
        "catboost": {
            "columns": list(X_calib_cb.columns), "cat_cols": m.CATBOOST_CAT_COLS,
            "raw_id_cols": m.RAW_ID_COLS, "id_mappings": id_mappings, "seeds": m.CB_SEEDS,
        },
        "retrieval": meta_rt,
        "blend_weight_catboost": W_CB,
        "calibration": {"method": "platt_sigmoid", "a": a, "b": b},
        "carveout_bss_calibrated": final_bss,
        "carveout_bss_reference_v26_1seed": 2082.7818915293506,
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)

    log(f"Saved to {OUT_MODEL_DIR}")

    # ---------- 제출 패키지 조립 ----------
    log("제출 패키지 조립...")
    for fname in os.listdir(OUT_MODEL_DIR):
        shutil.copy(os.path.join(OUT_MODEL_DIR, fname), os.path.join(SUB_DIR, "model", fname))
    log(f"Saved submission package to {SUB_DIR}")


if __name__ == "__main__":
    main()
