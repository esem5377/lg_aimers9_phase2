"""v38: confidence-weighted sample_weight로 CatBoost만 재학습(retrieval encoder/
EB-GLMM 구조는 v34와 동일, 재사용). "그거 다 해봤잖아 다른 방향 모색해"
(2026-08-29)에 대한 응답으로 조사한 미시도 축 -- asof_pitcher_n/asof_batter_n
(사전 이력 표본 수)로 CatBoost 학습 시 row별 신뢰도 가중치를 줘서, cold-start에
가까운(이력 얕은) 행의 라벨 노이즈 영향을 줄인다.

핵심 판단: 이 프로젝트에서 반복 검증된 패턴은 "기존 정보의 재구성/분산감소"
(control_risk_score, 시드 배깅)이고, "새 정보를 학습"하는 축은 반복 실패했다.
confidence-weighted sample_weight는 새 피처를 추가하는 게 아니라 기존 정보
(asof_*_n, 이미 X_train_cb에 피처로도 들어있음)의 신뢰도로 기존 라벨의 학습
기여도를 재조정하는 것 -- "재구성" 계열에 해당한다고 보고 시도.

가중치 공식(EB-GLMM의 n/(n+c) shrinkage 관례와 동일한 형태 재사용):
  w_p = asof_pitcher_n / (asof_pitcher_n + K),  K=100
  w_b = asof_batter_n / (asof_batter_n + K)
  sample_weight = 0.3 + 0.7 * sqrt(w_p * w_b)   (범위 [0.3, 1.0], cold-start도
  완전히 죽이지 않고 최소 0.3은 유지 -- test에도 cold-start 행이 있으므로)

사용자 지시(2026-08-29): walk-forward 게이트를 거치지 않고 바로 실제 제출로
진행(명시적 우회). 재현성/재개 가능성을 위해 CatBoost 네이티브 스냅샷
(save_snapshot=True)을 사용 -- 중간에 프로세스가 죽어도 동일 커맨드 재실행 시
스냅샷에서 이어서 학습됨. 스크립트 자체도 최종 모델 파일이 이미 있으면 그
단계를 건너뛰는 방식으로 멱등적으로 설계.

베이스: v34(현재 팀 최고 1024점)의 구조를 그대로 따름 -- CatBoost:retrieval=
0.7:0.3 고정 후 EB-GLMM을 저자유도(w_ebglmm 1개) 그리드서치로 블렌드.
retrieval encoder/참조임베딩은 v26에서 재사용(변경 없음, sample_weight는
CatBoost 학습에만 적용). EB-GLMM은 동일 데이터 split이라 재학습해도 v34와
사실상 동일하지만, 저장된 클로저 상태를 역직렬화하는 것보다 안전하기 위해
새로 학습(4~5초 수준, 저비용).
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

V25_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v25_retrieval_blend_1seed"
V34_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v34_threeway_ebglmm_blend"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
V38_DIR = os.path.dirname(os.path.abspath(__file__))
V38_MODEL_DIR = os.path.join(V38_DIR, "model")
V38_OUT_DIR = os.path.join(V38_DIR, "output")

CONFIDENCE_K = 100.0
CONFIDENCE_FLOOR = 0.3
SNAPSHOT_PATH = os.path.join(V38_DIR, "catboost_snapshot.bin")
CB_MODEL_PATH = os.path.join(V38_MODEL_DIR, "catboost_seed42.cbm")  # script.py가 이 이름으로 로드
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
W_CATBOOST_RETRIEVAL = 0.7
EBGLMM_SEASON_WINDOW = (2024, 2024)

# ---- v25/train_final.py 함수/상수 재사용 ----
g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "v38_reuse_v25"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main():")[0], g)
TARGET_COL, ID_COL, SEED = g["TARGET_COL"], g["ID_COL"], g["SEED"]
CATBOOST_CAT_COLS, RAW_ID_COLS = g["CATBOOST_CAT_COLS"], g["RAW_ID_COLS"]
BEST_PARAMS, ITERATIONS = g["BEST_PARAMS"], g["ITERATIONS"]

# ---- v34/01_build_v34.py의 EB-GLMM 구현 재사용 ----
eb = {"__file__": os.path.join(V34_DIR, "01_build_v34.py"), "__name__": "v38_reuse_v34"}
exec(open(os.path.join(V34_DIR, "01_build_v34.py"), encoding="utf-8").read().split("def main():")[0], eb)


def confidence_sample_weight(df):
    w_p = df["asof_pitcher_n"] / (df["asof_pitcher_n"] + CONFIDENCE_K)
    w_b = df["asof_batter_n"] / (df["asof_batter_n"] + CONFIDENCE_K)
    raw_w = np.sqrt(w_p * w_b)
    return (CONFIDENCE_FLOOR + (1 - CONFIDENCE_FLOOR) * raw_w).to_numpy(dtype=float)


def main():
    os.makedirs(V38_MODEL_DIR, exist_ok=True)
    os.makedirs(V38_OUT_DIR, exist_ok=True)

    print("Load data(전체 이력 2019~2024) + split(v26/v34와 동일 split)...", flush=True)
    df = g["load_data"]()
    df = g["add_risk_score_drop_ingredients"](df)
    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[TARGET_COL].to_numpy(dtype=float)
    print(f" train_sub={len(train_sub_df)} calib={len(calib_df)}", flush=True)

    sw_train = confidence_sample_weight(train_sub_df)
    print(f" sample_weight: min={sw_train.min():.3f} p10={np.percentile(sw_train,10):.3f} "
          f"median={np.median(sw_train):.3f} max={sw_train.max():.3f}", flush=True)

    # ---------- CatBoost(confidence-weighted, 재개 가능) ----------
    cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
    X_train_cb = g["build_catboost_features"](train_sub_df, cb_id_mappings)
    y_train_cb = train_sub_df[TARGET_COL]
    X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
    cat_idx = [X_train_cb.columns.get_loc(c) for c in CATBOOST_CAT_COLS]

    if os.path.exists(CB_MODEL_PATH):
        print(f"\n기존 학습 완료 모델 발견, 재학습 건너뜀: {CB_MODEL_PATH}", flush=True)
        cb_model = CatBoostClassifier()
        cb_model.load_model(CB_MODEL_PATH)
    else:
        print("\n=== CatBoost 학습(confidence-weighted, snapshot 재개 가능) ===", flush=True)
        t0 = time.time()
        cb_model = CatBoostClassifier(
            iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED,
            cat_features=cat_idx, verbose=200, **BEST_PARAMS,
            save_snapshot=True, snapshot_file=SNAPSHOT_PATH, snapshot_interval=30,
        )
        cb_model.fit(X_train_cb, y_train_cb, sample_weight=sw_train)
        print(f" CatBoost 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
        cb_model.save_model(CB_MODEL_PATH)
        print(f" saved: {CB_MODEL_PATH}", flush=True)

    cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

    # ---------- Retrieval(v26 재사용, 재학습 없음) ----------
    print("\nRetrieval raw 예측(calib, v26 encoder 재사용)...", flush=True)
    t0 = time.time()
    nn_cat_mappings, cardinalities = g["build_nn_cat_mappings"](train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + g["ALL_CAT_FOR_NN"]]
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
    nca_calib_raw = g["retrieve_predict"](encoder, device, cat_calib_nn, x_num_calib, z_ref, y_ref)
    print(f" retrieval calib 예측 완료({time.time()-t0:.1f}s)", flush=True)

    # ---------- EB-GLMM(v34와 동일 방식, 재학습) ----------
    print("\nEB-GLMM(last1season=2024) 학습...", flush=True)
    t0 = time.time()
    predict_raw_fn, ebglmm_state = eb["train_ebglmm_production"](train_sub_df, EBGLMM_SEASON_WINDOW)
    eb_calib_raw = predict_raw_fn(eb["add_count_base"](calib_df))
    print(f" EB-GLMM 학습+calib예측 완료({time.time()-t0:.1f}s)", flush=True)

    np.save(os.path.join(V38_OUT_DIR, "cb_calib_raw.npy"), cb_calib_raw)
    np.save(os.path.join(V38_OUT_DIR, "nca_calib_raw.npy"), nca_calib_raw)
    np.save(os.path.join(V38_OUT_DIR, "eb_calib_raw.npy"), eb_calib_raw)
    np.save(os.path.join(V38_OUT_DIR, "y_calib.npy"), y_calib)
    joblib.dump(ebglmm_state, os.path.join(V38_MODEL_DIR, "ebglmm_state.pkl"))

    # ---------- 저자유도 그리드서치(v34와 동일 절차) ----------
    print("\n저자유도 그리드서치(CatBoost:retrieval=0.7:0.3 고정, w_ebglmm만 탐색)...", flush=True)
    base_calib_raw = W_CATBOOST_RETRIEVAL * cb_calib_raw + (1 - W_CATBOOST_RETRIEVAL) * nca_calib_raw

    best = {"calib_bss": -1}
    grid_log = []
    for w in WEIGHT_GRID:
        blend_raw = (1 - w) * base_calib_raw + w * eb_calib_raw
        a, b = g["fit_platt_scaling"](blend_raw, y_calib)
        s = g["bss_score"](g["apply_platt_scaling"](blend_raw, a, b), y_calib)
        grid_log.append({"w_ebglmm": float(w), "calib_bss": float(s)})
        if s > best["calib_bss"]:
            best = {"w_ebglmm": float(w), "calib_bss": float(s), "a": a, "b": b}

    ref_no_ebglmm = grid_log[0]["calib_bss"]
    print(f" best_w_ebglmm={best['w_ebglmm']:.2f}  calib_bss={best['calib_bss']:.2f}  "
          f"(w_ebglmm=0.00일 때(=confidence-weighted CB+retrieval 순수 2way) calib_bss={ref_no_ebglmm:.2f})",
          flush=True)
    print(f" 참고 -- v34(비가중 CatBoost) 동일 지표: w_ebglmm=0.00일 때 calib_bss=2082.78", flush=True)
    print(f" delta vs v34 2way(2082.78) = {ref_no_ebglmm - 2082.78:+.2f}", flush=True)

    with open(os.path.join(V38_OUT_DIR, "metrics_v38.json"), "w", encoding="utf-8") as f:
        json.dump({
            "confidence_k": CONFIDENCE_K, "confidence_floor": CONFIDENCE_FLOOR,
            "w_catboost_retrieval": W_CATBOOST_RETRIEVAL,
            "best_w_ebglmm": best["w_ebglmm"], "carveout_calib_bss": best["calib_bss"],
            "calib_bss_no_ebglmm_reference": ref_no_ebglmm,
            "calib_bss_v34_unweighted_reference": 2082.78,
            "delta_vs_v34_2way": ref_no_ebglmm - 2082.78,
            "calibration_a": best["a"], "calibration_b": best["b"],
            "grid_log": grid_log,
            "n_train_sub": len(train_sub_df), "n_calib": len(calib_df),
        }, f, indent=2, ensure_ascii=False)

    # ---------- feature_meta.json(v26 것을 베이스로 blend/calibration만 갱신) ----------
    with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["blend_weight_catboost"] = W_CATBOOST_RETRIEVAL
    meta["blend_weight_ebglmm"] = best["w_ebglmm"]
    meta["calibration"] = {"method": "platt_sigmoid", "a": best["a"], "b": best["b"]}
    with open(os.path.join(V38_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {V38_MODEL_DIR}\\catboost_seed42.cbm, ebglmm_state.pkl, feature_meta.json", flush=True)
    print(f" metrics: {V38_OUT_DIR}\\metrics_v38.json", flush=True)


if __name__ == "__main__":
    main()
