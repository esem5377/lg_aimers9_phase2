"""v40: v34(CatBoost:retrieval=0.7:0.3 + EB-GLMM 3-way, 팀 최고 1024점) 레시피에서
CatBoost만 교체 -- pitcher_id/batter_id(RAW_ID_COLS)를 지금까지처럼 label-encoded
숫자(수치 피처)가 아니라 CatBoost native categorical feature로 취급해서 학습
(cat_features에 추가). retrieval encoder는 v26 것을 그대로 재사용(재학습 없음),
EB-GLMM(last1season)도 v34와 동일 방식으로 재학습.

배경(2026-08-29): "완전 새로운 방향" 요청 -> 조사 결과 max_ctr_complexity를
그냥 올리는 건 es_ws가 2026-08-16에 이미 시도해 완전 무효 확인(기존 7개
저카디널리티 범주형 컬럼만으로는 기본값에서 이미 포화). 그런데 그 실험은
RAW_ID_COLS(pitcher_id=792종/batter_id=830종, 고카디널리티)를 건드리지
않았음 -- 이 프로젝트 전체 이력에서 RAW_ID_COLS는 항상 CatBoost에
**numeric**으로만 들어갔고, CatBoost의 leak-safe ordered CTR 조합 탐색
범위에 이 두 컬럼이 들어간 적이 없음. 이번 실험은 그 공백을 메움 -- 과거
실패한 수동 매치업 히스토리 피처(콜드스타트로 -7.61 기각)와 달리 CatBoost
자체의 카운터 기반 prior 스무딩 + ordered boosting(타겟 누출 방지)을 씀.

사용자가 fold0/fold2 walk-forward 검증 없이 바로 실제 제출까지 진행하기로
명시적으로 결정(2026-08-29) -- 이 프로젝트에서 v37/v38b처럼 사전 선례 있음
(성공/실패 혼재), 사용자의 명시적 판단.

script.py는 v34와 동일(수정 없음) -- build_catboost_features가 meta의
cat_cols(=CATBOOST_CAT_COLS, 7개)만 str로 캐스팅하고 raw_id_cols는 계속
int로 유지하므로, "이 컬럼이 실제로 categorical인지"는 저장된 .cbm 모델
내부 정보(학습 시 cat_features 인자)에만 좌우됨 -- 추론 스크립트 변경 불필요.
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
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
V40_DIR = os.path.dirname(__file__)
V40_MODEL_DIR = os.path.join(V40_DIR, "model")
V40_OUT_DIR = os.path.join(V40_DIR, "output")

EBGLMM_SEASON_WINDOW = (2024, 2024)
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "finalize"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)
TARGET_COL, ID_COL, SEED = g["TARGET_COL"], g["ID_COL"], g["SEED"]

# ================= EB-GLMM(last1season) -- v34와 동일, 자체 포함 =================
CAT_FIXED_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RANDOM_EFFECT_ID_COLS = ["pitcher_id", "batter_id"]
GROUPS = ["pitcher_id", "batter_id", "count_base"]
N_BACKFIT_ROUNDS = 4
L2_FIXED = 10.0
IRLS_MAX_ITER = 25
IRLS_TOL = 1e-6
MIN_GROUP_W_FOR_MOM = 3.0
SIGMA_U2_FLOOR = 1e-4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def add_count_base(df):
    df = df.copy()
    df["count_base"] = (
        df["balls_before"].astype(str) + "_"
        + df["strikes_before"].astype(str) + "_"
        + df["base_state"].astype(str)
    )
    return df


class FixedEffectDesign:
    def __init__(self):
        self.numeric_cols = None
        self.num_mean = None
        self.num_std = None
        self.num_median = None
        self.cat_categories = {}
        self.columns_ = None

    def fit(self, df):
        exclude = set([ID_COL, TARGET_COL] + CAT_FIXED_COLS + RANDOM_EFFECT_ID_COLS + ["count_base"])
        self.numeric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.num_median = df[self.numeric_cols].median()
        filled = df[self.numeric_cols].fillna(self.num_median)
        self.num_mean = filled.mean()
        self.num_std = filled.std().replace(0.0, 1.0).fillna(1.0)
        for c in CAT_FIXED_COLS:
            cats = sorted(df[c].astype(str).fillna("missing").unique().tolist())
            self.cat_categories[c] = cats
        X = self.transform(df, _fitting_columns=True)
        self.columns_ = list(X.columns)
        return self

    def transform(self, df, _fitting_columns=False):
        parts = []
        num = df[self.numeric_cols].fillna(self.num_median)
        num = (num - self.num_mean) / self.num_std
        parts.append(num.reset_index(drop=True))
        for c in CAT_FIXED_COLS:
            s = df[c].astype(str).fillna("missing")
            cats = self.cat_categories[c]
            s = pd.Categorical(s, categories=cats)
            dummies = pd.get_dummies(s, prefix=c, drop_first=True)
            parts.append(dummies.reset_index(drop=True))
        X = pd.concat(parts, axis=1)
        if not _fitting_columns and self.columns_ is not None:
            X = X.reindex(columns=self.columns_, fill_value=0.0)
        return X.astype(float)


def fit_logistic_irls_with_offset(X, y, offset, l2=L2_FIXED, max_iter=IRLS_MAX_ITER, tol=IRLS_TOL):
    n, p = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(Xd.shape[1])
    reg = np.eye(Xd.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(max_iter):
        eta = offset + Xd @ beta
        p_hat = sigmoid(eta)
        w = np.clip(p_hat * (1 - p_hat), 1e-6, None)
        z = (eta - offset) + (y - p_hat) / w
        WX = Xd * w[:, None]
        A = Xd.T @ WX + reg
        bvec = Xd.T @ (w * z)
        beta_new = np.linalg.solve(A, bvec)
        delta = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if delta < tol:
            break
    return beta


def estimate_sigma_u2(u_raw, sw):
    mask = sw >= MIN_GROUP_W_FOR_MOM
    if mask.sum() < 5:
        return SIGMA_U2_FLOOR
    u_use = u_raw[mask]
    sw_use = sw[mask]
    var_raw = np.var(u_use)
    mean_recip = np.mean(1.0 / sw_use)
    sigma_u2 = var_raw - mean_recip
    return max(SIGMA_U2_FLOOR, sigma_u2)


def update_group_effect(group_codes_train, eta_excl_g, z, w, n_levels):
    local_target = z - eta_excl_g
    sw = np.bincount(group_codes_train, weights=w, minlength=n_levels)
    swt = np.bincount(group_codes_train, weights=w * local_target, minlength=n_levels)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_raw = np.where(sw > 0, swt / np.maximum(sw, 1e-12), 0.0)
    sigma_u2 = estimate_sigma_u2(u_raw, np.maximum(sw, 1e-12))
    shrink = sw / (sw + 1.0 / sigma_u2)
    u_final = shrink * u_raw
    u_final[sw == 0] = 0.0
    return u_final, sigma_u2, sw


def train_ebglmm_production(train_sub_df_full, season_window):
    lo, hi = season_window
    train_df = train_sub_df_full[
        (train_sub_df_full["season"] >= lo) & (train_sub_df_full["season"] <= hi)
    ].copy()
    train_df = add_count_base(train_df)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    print(f"  [EB-GLMM] season_window={season_window} n_train={len(train_df)}", flush=True)

    fe = FixedEffectDesign().fit(train_df)
    X_train = fe.transform(train_df).to_numpy(dtype=float)

    group_maps, group_codes_train, n_levels = {}, {}, {}
    for gcol in GROUPS:
        uniq = pd.unique(train_df[gcol])
        gmap = {v: i for i, v in enumerate(uniq)}
        group_maps[gcol] = gmap
        group_codes_train[gcol] = train_df[gcol].map(gmap).to_numpy()
        n_levels[gcol] = len(uniq)

    u_effects = {gcol: np.zeros(n_levels[gcol]) for gcol in GROUPS}
    beta = None

    def group_offset_train():
        total = np.zeros(len(train_df))
        for gcol in GROUPS:
            total += u_effects[gcol][group_codes_train[gcol]]
        return total

    for rnd in range(N_BACKFIT_ROUNDS):
        offset_fixed = group_offset_train()
        beta = fit_logistic_irls_with_offset(X_train, y_train, offset_fixed, l2=L2_FIXED)
        Xd_train = np.hstack([np.ones((len(train_df), 1)), X_train])
        fixed_eta_train = Xd_train @ beta
        for gcol in GROUPS:
            eta_excl = fixed_eta_train.copy()
            for g2 in GROUPS:
                if g2 == gcol:
                    continue
                eta_excl += u_effects[g2][group_codes_train[g2]]
            eta_now = eta_excl + u_effects[gcol][group_codes_train[gcol]]
            p_hat = sigmoid(eta_now)
            w = np.clip(p_hat * (1 - p_hat), 1e-6, None)
            z = eta_now + (y_train - p_hat) / w
            u_new, sigma_u2, sw = update_group_effect(group_codes_train[gcol], eta_excl, z, w, n_levels[gcol])
            u_effects[gcol] = u_new
        print(f"    round {rnd+1}/{N_BACKFIT_ROUNDS} done", flush=True)

    def predict_raw(df_with_count_base):
        Xd = np.hstack([np.ones((len(df_with_count_base), 1)), fe.transform(df_with_count_base).to_numpy(dtype=float)])
        eta = Xd @ beta
        for gcol in GROUPS:
            codes = df_with_count_base[gcol].map(group_maps[gcol]).to_numpy()
            valid = ~pd.isna(codes)
            add = np.zeros(len(df_with_count_base))
            idx = codes[valid].astype(int)
            add[valid] = u_effects[gcol][idx]
            eta += add
        return sigmoid(eta)

    state = {
        "numeric_cols": fe.numeric_cols,
        "num_mean": fe.num_mean.to_dict(),
        "num_std": fe.num_std.to_dict(),
        "num_median": fe.num_median.to_dict(),
        "cat_categories": fe.cat_categories,
        "columns_": fe.columns_,
        "group_maps": group_maps,
        "u_effects": u_effects,
        "beta": beta,
        "cat_fixed_cols": CAT_FIXED_COLS,
        "groups": GROUPS,
        "season_window": season_window,
    }
    return predict_raw, state


def main():
    print("=== v40: CatBoost(pitcher_id/batter_id as categorical) + retrieval(v26 재사용,0.7:0.3) + EB-GLMM(last1season) ===", flush=True)
    ckpt_path = os.path.join(V40_OUT_DIR, "checkpoint_stage1.pkl")
    os.makedirs(V40_OUT_DIR, exist_ok=True)
    os.makedirs(V40_MODEL_DIR, exist_ok=True)

    print("Load data(전체 이력 2019~2024) + split(재현, v26/v34와 동일 split)...", flush=True)
    df = g["load_data"]()
    df = g["add_risk_score_drop_ingredients"](df)
    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[TARGET_COL].to_numpy(dtype=float)
    print(f" train_sub={len(train_sub_df)} calib={len(calib_df)}", flush=True)

    cb_model_path = os.path.join(V40_MODEL_DIR, "catboost_seed42.cbm")
    cb_calib_raw_path = os.path.join(V40_OUT_DIR, "cb_calib_raw.npy")

    cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
    X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)

    if os.path.exists(cb_model_path) and os.path.exists(cb_calib_raw_path):
        print("\nCatBoost(id_as_cat) 체크포인트 발견, 재학습 스킵...", flush=True)
        cb_model = CatBoostClassifier()
        cb_model.load_model(cb_model_path)
        cb_calib_raw = np.load(cb_calib_raw_path)
        X_train_cb_columns = list(X_calib_cb.columns)
    else:
        print("\nCatBoost 학습(pitcher_id/batter_id를 cat_features에 추가)...", flush=True)
        X_train_cb = g["build_catboost_features"](train_sub_df, cb_id_mappings)
        cat_cols_treatment = g["CATBOOST_CAT_COLS"] + g["RAW_ID_COLS"]
        cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols_treatment]
        t0 = time.time()
        cb_model = CatBoostClassifier(
            iterations=g["ITERATIONS"], loss_function="Logloss", random_seed=SEED,
            cat_features=cat_idx, verbose=200, **g["BEST_PARAMS"],
        )
        cb_model.fit(X_train_cb, train_sub_df[TARGET_COL])
        print(f" CatBoost 학습 완료 ({time.time()-t0:.1f}s)", flush=True)
        cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]
        cb_model.save_model(cb_model_path)
        np.save(cb_calib_raw_path, cb_calib_raw)
        X_train_cb_columns = list(X_train_cb.columns)
        print(" CatBoost 체크포인트 저장 완료", flush=True)

    print("\nRetrieval raw 예측(calib, v26 재사용, 재학습 없음, 참조=1,401,337행)...", flush=True)
    t0 = time.time()
    nca_calib_raw_path = os.path.join(V40_OUT_DIR, "nca_calib_raw.npy")
    if os.path.exists(nca_calib_raw_path):
        print(" retrieval 체크포인트 발견, 재계산 스킵", flush=True)
        nca_calib_raw = np.load(nca_calib_raw_path)
    else:
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
        np.save(nca_calib_raw_path, nca_calib_raw)
    print(f" retrieval calib 예측 완료({time.time()-t0:.1f}s)", flush=True)

    print("\nEB-GLMM(last1season=2024) 학습...", flush=True)
    eb_calib_raw_path = os.path.join(V40_OUT_DIR, "eb_calib_raw.npy")
    ebglmm_state_path = os.path.join(V40_MODEL_DIR, "ebglmm_state.pkl")
    if os.path.exists(eb_calib_raw_path) and os.path.exists(ebglmm_state_path):
        print(" EB-GLMM 체크포인트 발견, 재학습 스킵", flush=True)
        eb_calib_raw = np.load(eb_calib_raw_path)
        ebglmm_state = joblib.load(ebglmm_state_path)
    else:
        t0 = time.time()
        predict_raw_fn, ebglmm_state = train_ebglmm_production(train_sub_df, EBGLMM_SEASON_WINDOW)
        eb_calib_raw = predict_raw_fn(add_count_base(calib_df))
        np.save(eb_calib_raw_path, eb_calib_raw)
        joblib.dump(ebglmm_state, ebglmm_state_path)
        print(f" EB-GLMM 학습+calib예측 완료({time.time()-t0:.1f}s)", flush=True)

    np.save(os.path.join(V40_OUT_DIR, "y_calib.npy"), y_calib)

    print("\n저자유도 그리드서치(CatBoost:retrieval=0.7:0.3 고정, w_ebglmm만 탐색)...", flush=True)
    W_CATBOOST_RETRIEVAL = 0.7
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
    print(f" best_w_ebglmm={best['w_ebglmm']:.2f}  calib_bss={best['calib_bss']:.2f}  "
          f"(참고: w_ebglmm=0.00일 때 calib_bss={grid_log[0]['calib_bss']:.2f} == 순수 CatBoost(id_as_cat)+retrieval(0.7:0.3), "
          f"v34 기준값=2082.78)",
          flush=True)

    with open(os.path.join(V40_OUT_DIR, "metrics_v40.json"), "w", encoding="utf-8") as f:
        json.dump({
            "w_catboost_retrieval": W_CATBOOST_RETRIEVAL,
            "best_w_ebglmm": best["w_ebglmm"],
            "carveout_calib_bss": best["calib_bss"],
            "calib_bss_no_ebglmm_reference": grid_log[0]["calib_bss"],
            "v34_reference_calib_bss": 2082.78,
            "calibration_a": best["a"], "calibration_b": best["b"],
            "grid_log": grid_log,
            "ebglmm_season_window": EBGLMM_SEASON_WINDOW,
            "n_train_sub": len(train_sub_df), "n_calib": len(calib_df),
        }, f, indent=2, ensure_ascii=False)

    # ---- feature_meta.json: retrieval 블록은 v26 것 재사용, catboost 블록만 이 실행 기준으로 갱신 ----
    with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["catboost"] = {
        "columns": X_train_cb_columns,
        "cat_cols": g["CATBOOST_CAT_COLS"],
        "raw_id_cols": g["RAW_ID_COLS"],
        "id_mappings": cb_id_mappings,
    }
    meta["blend_weight_catboost"] = W_CATBOOST_RETRIEVAL
    meta["blend_weight_ebglmm"] = best["w_ebglmm"]
    meta["calibration"] = {"method": "platt_sigmoid", "a": best["a"], "b": best["b"]}
    with open(os.path.join(V40_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- retrieval/trackman 아티팩트를 v26에서 그대로 복사 ----
    import shutil
    for fname in ["retrieval_encoder.pt", "reference_embeddings.npy", "reference_labels.npy", "numeric_prep.pkl", "trackman_context.pkl"]:
        shutil.copy(os.path.join(V26_MODEL_DIR, fname), os.path.join(V40_MODEL_DIR, fname))

    print(f"\nSaved: {V40_MODEL_DIR} (catboost_seed42.cbm, ebglmm_state.pkl, feature_meta.json + v26 retrieval 아티팩트 복사) / {V40_OUT_DIR}\\metrics_v40.json", flush=True)


if __name__ == "__main__":
    main()
