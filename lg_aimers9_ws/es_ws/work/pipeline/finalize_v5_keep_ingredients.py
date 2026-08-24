"""v5 = CatBoost는 원재료(reverse/middle/ball_rate) 유지 버전(jh_ws v20, 6시드,
재학습 안 하고 기존 자산 재사용) + retrieval은 v26 인코더 그대로 재사용.
같은 carve-out에서 calibrated BSS 2090.34 (v13=2085.15 대비 +5.19,
v26 원본=2082.78 대비 +7.56) 확인 -> 최종 채택.
"""
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import reuse_v26_retrieval_with_cb6 as m

SEED = 42
W_CB = 0.70
V20_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(m._BASE)), "jh_ws", "v20_control_risk_score", "model")
OUT_MODEL_DIR = os.path.join(m._BASE, "model_v5_keep_ing_retrieval")
SUB_DIR = os.path.join(m._BASE, "submissions", "submit_v14_keep_ing_retrieval")


def log(msg):
    print(msg, flush=True)


def main():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(SUB_DIR, "model"), exist_ok=True)

    log("Load train data...")
    df = pd.read_csv(os.path.join(m.DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(m.TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    df["control_risk_score"] = df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    df["control_risk_score_weighted"] = 0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]

    y_all = df[m.TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[m.TARGET_COL]

    with open(os.path.join(V20_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        v20_meta = json.load(f)

    from catboost import CatBoostClassifier
    X_calib_cb = calib_df.drop(columns=[m.ID_COL, m.TARGET_COL])
    for c in v20_meta["raw_id_cols"]:
        X_calib_cb[c] = X_calib_cb[c].astype(str).map(v20_meta["id_mappings"][c]).fillna(-1).astype(int)
    for c in v20_meta["cat_cols"]:
        X_calib_cb[c] = X_calib_cb[c].astype(str)
    X_calib_cb = X_calib_cb[v20_meta["columns"]]

    cb_calib_raws = []
    for seed in v20_meta["seeds"]:
        mdl = CatBoostClassifier()
        mdl.load_model(os.path.join(V20_MODEL_DIR, f"catboost_seed{seed}.cbm"))
        cb_calib_raws.append(mdl.predict_proba(X_calib_cb)[:, 1])
    cb_calib_raw = np.mean(cb_calib_raws, axis=0)
    log(f" CB6(원재료 유지) raw BSS={m.bss_score(cb_calib_raw, y_calib):.2f}")

    log("Load v26 retrieval encoder(재학습 안 함) + 임베딩 재계산 (원재료 drop 피처셋 사용)...")
    train_sub_dropped = train_sub_df.drop(columns=m.INGREDIENT_COLS)
    calib_dropped = calib_df.drop(columns=m.INGREDIENT_COLS)

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

    cat_train_nn = m.encode_cats(train_sub_dropped, meta_rt["cat_mappings"])
    cat_calib_nn = m.encode_cats(calib_dropped, meta_rt["cat_mappings"])
    x_num_train = m.prep_numeric_transform(train_sub_dropped, meta_rt["numeric_cols"], medians, scaler)
    x_num_calib = m.prep_numeric_transform(calib_dropped, meta_rt["numeric_cols"], medians, scaler)

    z_ref = m.compute_embeddings(encoder, device, cat_train_nn, x_num_train)
    y_ref = train_sub_dropped[m.TARGET_COL].values.astype(np.float32)
    nca_calib_raw = m.retrieve_predict(encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
    log(f" Retrieval raw BSS={m.bss_score(nca_calib_raw, y_calib):.2f}  (기대 1896.79)")

    raw = W_CB * cb_calib_raw + (1 - W_CB) * nca_calib_raw
    a, b = m.fit_platt_scaling(raw, y_calib)
    calib_pred = m.apply_platt_scaling(raw, a, b)
    final_bss = m.bss_score(calib_pred, y_calib)
    log(f" 최종 calibrated BSS(w_cb={W_CB})={final_bss:.2f}  (v13=2085.15 대비 {final_bss-2085.15:+.2f}, v26=2082.78 대비 {final_bss-2082.78:+.2f})")

    log("모델 아티팩트 저장...")
    for seed in v20_meta["seeds"]:
        shutil.copy(os.path.join(V20_MODEL_DIR, f"catboost_seed{seed}.cbm"),
                    os.path.join(OUT_MODEL_DIR, f"catboost_seed{seed}.cbm"))
    torch.save(encoder.state_dict(), os.path.join(OUT_MODEL_DIR, "retrieval_encoder.pt"))
    np.save(os.path.join(OUT_MODEL_DIR, "reference_embeddings.npy"), z_ref.numpy().astype(np.float32))
    np.save(os.path.join(OUT_MODEL_DIR, "reference_labels.npy"), y_ref)
    joblib.dump({"medians": medians, "scaler": scaler}, os.path.join(OUT_MODEL_DIR, "numeric_prep.pkl"))
    shutil.copy(m.TRACKMAN_CONTEXT_PATH, os.path.join(OUT_MODEL_DIR, "trackman_context.pkl"))

    feature_meta = {
        "catboost": {
            "columns": v20_meta["columns"], "cat_cols": v20_meta["cat_cols"],
            "raw_id_cols": v20_meta["raw_id_cols"], "id_mappings": v20_meta["id_mappings"],
            "seeds": v20_meta["seeds"], "keeps_ingredients": True,
        },
        "retrieval": meta_rt,
        "blend_weight_catboost": W_CB,
        "calibration": {"method": "platt_sigmoid", "a": a, "b": b},
        "carveout_bss_calibrated": final_bss,
        "carveout_bss_reference_v13_drop_ing": 2085.15,
        "carveout_bss_reference_v26_1seed": 2082.7818915293506,
    }
    with open(os.path.join(OUT_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)
    log(f"Saved to {OUT_MODEL_DIR}")

    for fname in os.listdir(OUT_MODEL_DIR):
        shutil.copy(os.path.join(OUT_MODEL_DIR, fname), os.path.join(SUB_DIR, "model", fname))
    log(f"Saved submission package to {SUB_DIR}")


if __name__ == "__main__":
    main()
