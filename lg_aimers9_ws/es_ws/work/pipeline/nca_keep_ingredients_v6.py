"""retrieval NCA 인코더를 원재료(reverse/middle/ball_rate) 유지 피처셋으로
재학습(3시드 앙상블, 결정적 모드) -- CatBoost가 원재료 유지시 더 강했던 것과
같은 효과가 retrieval에도 나타나는지 확인. CB6(keep, jh_ws v20 자산 재사용)과
블렌드해 v14(2090.34)를 넘는지 검증.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

import reuse_v26_retrieval_with_cb6 as m
import train_retrieval_blend_v3 as v3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NCA_SEEDS = [42, 7, 123]


def log(msg):
    print(msg, flush=True)


def main():
    df = pd.read_csv(os.path.join(m.DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(m.TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df["control_risk_score"] = df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    df["control_risk_score_weighted"] = 0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]

    y_all = df[m.TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=m.SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[m.TARGET_COL]

    with open("/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/jh_ws/v20_control_risk_score/model/feature_meta.json", encoding="utf-8") as f:
        v20_meta = json.load(f)

    def build_cb(dfx):
        X = dfx.drop(columns=[m.ID_COL, m.TARGET_COL])
        for c in v20_meta["raw_id_cols"]:
            X[c] = X[c].astype(str).map(v20_meta["id_mappings"][c]).fillna(-1).astype(int)
        for c in v20_meta["cat_cols"]:
            X[c] = X[c].astype(str)
        return X[v20_meta["columns"]]

    X_calib_cb = build_cb(calib_df)
    raws = []
    for seed in v20_meta["seeds"]:
        mdl = CatBoostClassifier()
        mdl.load_model(f"/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/jh_ws/v20_control_risk_score/model/catboost_seed{seed}.cbm")
        raws.append(mdl.predict_proba(X_calib_cb)[:, 1])
    cb_calib_raw = np.mean(raws, axis=0)
    log(f"CB6(keep) raw BSS={m.bss_score(cb_calib_raw, y_calib):.2f}")

    # retrieval: 원재료 유지 피처셋 (INGREDIENT_COLS를 drop하지 않음)
    nn_cat_mappings, cardinalities = v3.build_nn_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [v3.ID_COL, v3.TARGET_COL] + v3.ALL_CAT_FOR_NN]
    log(f"n_numeric(keep)={len(numeric_cols)}  (drop 버전은 61)")
    cat_train_nn = v3.encode_cats(train_sub_df, nn_cat_mappings)
    cat_calib_nn = v3.encode_cats(calib_df, nn_cat_mappings)
    x_num_train, medians, scaler = v3.prep_numeric_fit(train_sub_df, numeric_cols)
    x_num_calib = v3.prep_numeric_transform(calib_df, numeric_cols, medians, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nca_calib_preds = []
    encoders = []
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
        log(f" NCA(keep) seed={seed} raw BSS={bss:.2f} ({time.time()-t0:.0f}s)")
        nca_calib_preds.append(nca_raw)
        encoders.append((encoder, z_ref, y_ref))

    nca_ens = np.mean(nca_calib_preds, axis=0)
    log(f"NCA(keep) 3시드 앙상블 raw BSS={m.bss_score(nca_ens, y_calib):.2f}  (drop버전 원본=1896.79, drop버전 3시드앙상블=1920.92)")

    log("\n=== w_cb 재탐색 (CB6-keep + NCA-keep 앙상블, calibrated 기준) ===")
    best = None
    for i in range(21):
        w_cb = i * 0.05
        raw = w_cb * cb_calib_raw + (1 - w_cb) * nca_ens
        a, b = m.fit_platt_scaling(raw, y_calib)
        calib = m.apply_platt_scaling(raw, a, b)
        cbss = m.bss_score(calib, y_calib)
        if best is None or cbss > best[0]:
            best = (cbss, w_cb, a, b)
    log(f"BEST: w_cb={best[1]:.2f}  calibrated_bss={best[0]:.2f}")
    log(f"v14(2090.34) 대비 delta = {best[0]-2090.34:+.2f}")
    log(f"v14+LGB(2090.92) 대비 delta = {best[0]-2090.92:+.2f}")


if __name__ == "__main__":
    main()
