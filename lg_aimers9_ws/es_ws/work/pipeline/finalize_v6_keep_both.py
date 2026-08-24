"""v6(v15) = CatBoost 6시드(원재료 유지, jh_ws v20 자산 재사용) + retrieval도
원재료 유지 피처셋으로 재학습(단일 시드=7, 결정적 모드). carve-out calibrated
BSS: v26(1023 실측, 1시드 CB+drop retrieval)=2082.78 -> v13(6시드 CB+drop
retrieval)=2085.15 -> v14(6시드 CB-keep+drop retrieval)=2090.34 -> v15(6시드
CB-keep + retrieval-keep 단일시드)=2098.71.
scratchpad에 미리 저장해둔 seed=7 인코더/임베딩을 재사용(재학습 안 함).
"""
import json
import os
import shutil

import joblib
import numpy as np
import torch

import reuse_v26_retrieval_with_cb6 as m
import train_retrieval_blend_v3 as v3

SCRATCH = "/tmp/claude-1000/-home-esem5377-lg-aimers9-ws/6231e794-c099-49d2-b0e1-a611de390643/scratchpad"
V20_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(m._BASE)), "jh_ws", "v20_control_risk_score", "model")
OUT_MODEL_DIR = os.path.join(m._BASE, "model_v6_keep_both")
SUB_DIR = os.path.join(m._BASE, "submissions", "submit_v15_keep_both")

W_CB = 0.65
A, B = 4.815300779346502, -2.4224834246877687


def log(msg):
    print(msg, flush=True)


def main():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(SUB_DIR, "model"), exist_ok=True)

    with open(os.path.join(V20_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        v20_meta = json.load(f)

    for seed in v20_meta["seeds"]:
        shutil.copy(os.path.join(V20_MODEL_DIR, f"catboost_seed{seed}.cbm"),
                    os.path.join(OUT_MODEL_DIR, f"catboost_seed{seed}.cbm"))

    shutil.copy(os.path.join(SCRATCH, "nca_keep_seed7_encoder.pt"), os.path.join(OUT_MODEL_DIR, "retrieval_encoder.pt"))
    shutil.copy(os.path.join(SCRATCH, "nca_keep_seed7_zref.npy"), os.path.join(OUT_MODEL_DIR, "reference_embeddings.npy"))
    shutil.copy(os.path.join(SCRATCH, "nca_keep_seed7_yref.npy"), os.path.join(OUT_MODEL_DIR, "reference_labels.npy"))
    rt_meta = joblib.load(os.path.join(SCRATCH, "nca_keep_seed7_meta.pkl"))
    joblib.dump({"medians": rt_meta["medians"], "scaler": rt_meta["scaler"]}, os.path.join(OUT_MODEL_DIR, "numeric_prep.pkl"))
    shutil.copy(m.TRACKMAN_CONTEXT_PATH, os.path.join(OUT_MODEL_DIR, "trackman_context.pkl"))

    feature_meta = {
        "catboost": {
            "columns": v20_meta["columns"], "cat_cols": v20_meta["cat_cols"],
            "raw_id_cols": v20_meta["raw_id_cols"], "id_mappings": v20_meta["id_mappings"],
            "seeds": v20_meta["seeds"], "keeps_ingredients": True,
        },
        "retrieval": {
            "numeric_cols": rt_meta["numeric_cols"], "cat_cols": v3.ALL_CAT_FOR_NN,
            "cat_mappings": rt_meta["cat_mappings"], "cardinalities": rt_meta["cardinalities"],
            "embed_dims": {
                "pitcher_id": 24, "batter_id": 24, "pitcher_team_id": 6, "batter_team_id": 6,
                "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2, "base_state": 4,
            },
            "encoder_hidden": [256, 128], "embed_out_dim": 32,
            "keeps_ingredients": True,
        },
        "blend_weight_catboost": W_CB,
        "calibration": {"method": "platt_sigmoid", "a": A, "b": B},
        "carveout_bss_calibrated": 2098.71,
        "carveout_bss_reference_v14": 2090.34,
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
