"""retrieval NCA 인코더를 3시드(42/7/123)로 앙상블해서 CB6 블렌드에 추가
개선 여지가 있는지 확인. 결정적 모드(cudnn deterministic)로 학습해 GPU
재현성 문제를 우회.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

import reuse_v26_retrieval_with_cb6 as m
import train_retrieval_blend_v3 as v3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NCA_SEEDS = [42, 7, 123]


def log(msg):
    print(msg, flush=True)


def main():
    df = m.load_data()
    df = m.add_risk_score_drop_ingredients(df)
    y_all = df[m.TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=m.SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[m.TARGET_COL]

    id_mappings = {c: {v: i for i, v in enumerate(sorted(train_sub_df[c].astype(str).unique()))} for c in m.RAW_ID_COLS}
    X_calib_cb = m.build_catboost_features(calib_df, id_mappings)
    cb_calib_raws = []
    for seed in m.CB_SEEDS:
        mdl = CatBoostClassifier()
        mdl.load_model(os.path.join(m.CB6_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cb_calib_raws.append(mdl.predict_proba(X_calib_cb)[:, 1])
    cb_calib_raw = np.mean(cb_calib_raws, axis=0)
    log(f"CB6 raw BSS={m.bss_score(cb_calib_raw, y_calib):.2f}")

    nn_cat_mappings, cardinalities = v3.build_nn_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [v3.ID_COL, v3.TARGET_COL] + v3.ALL_CAT_FOR_NN]
    cat_train_nn = v3.encode_cats(train_sub_df, nn_cat_mappings)
    cat_calib_nn = v3.encode_cats(calib_df, nn_cat_mappings)
    x_num_train, medians, scaler = v3.prep_numeric_fit(train_sub_df, numeric_cols)
    x_num_calib = v3.prep_numeric_transform(calib_df, numeric_cols, medians, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nca_calib_preds = []
    for seed in NCA_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        v3.SEED = seed
        encoder, device = v3.train_encoder(cat_train_nn, x_num_train, train_sub_df[v3.TARGET_COL], cardinalities, x_num_train.shape[1])
        z_ref = v3.compute_embeddings(encoder, device, cat_train_nn, x_num_train)
        y_ref = train_sub_df[v3.TARGET_COL].values
        nca_raw = v3.retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
        bss = m.bss_score(nca_raw, y_calib)
        log(f" NCA seed={seed} raw BSS={bss:.2f} ({time.time()-t0:.0f}s)")
        nca_calib_preds.append(nca_raw)

    nca_ens = np.mean(nca_calib_preds, axis=0)
    log(f"NCA 3시드 앙상블 raw BSS={m.bss_score(nca_ens, y_calib):.2f}  (원본 1시드=1896.79, 재학습단일 1845.04)")

    log("\n=== w_cb 재탐색 (calibrated BSS 기준) ===")
    best = None
    for i in range(21):
        w_cb = i * 0.05
        raw = w_cb * cb_calib_raw + (1 - w_cb) * nca_ens
        a, b = m.fit_platt_scaling(raw, y_calib)
        calib = m.apply_platt_scaling(raw, a, b)
        cbss = m.bss_score(calib, y_calib)
        if best is None or cbss > best[0]:
            best = (cbss, w_cb, a, b)
        log(f"  w_cb={w_cb:.2f}  calibrated_bss={cbss:.2f}")
    log(f"\nBEST: w_cb={best[1]:.2f}  calibrated_bss={best[0]:.2f}")
    log(f"현재 채택안(v13, CB6+v26단일encoder)=2085.15  ->  이번(CB6+NCA3시드앙상블)={best[0]:.2f}  delta={best[0]-2085.15:+.2f}")


if __name__ == "__main__":
    main()
