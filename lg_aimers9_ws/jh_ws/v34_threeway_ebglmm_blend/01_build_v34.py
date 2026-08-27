"""
v26(CatBoost:retrieval=0.7:0.3, 팀 최고 1023점)의 이미 학습된 CatBoost/encoder/
참조임베딩을 그대로 재사용(재학습 없음)하고, 세 번째 재료로 EB-GLMM(계층적
경험적 베이즈 로지스틱 GLMM, 최근1시즌만 재학습하는 "레짐분리" 버전)을 추가.

배경(hackathon_fresh 세션 로컬 스크리닝 전부 완료):
- fold0/fold2 walk-forward에서 EB-GLMM(last1season)은 CatBoost와 오류상관이
  높아(0.998+) 2-way/3-way(저자유도)/3-way(조인트) 전부 실익이 거의 없거나
  (조인트는 오히려 calib 과적합으로 손해)였음 -- 로컬 근거만 보면 매우 약함.
- 그럼에도 사용자가 실제 제출까지 진행하기로 결정(이 프로젝트에서 v30/v31처럼
  로컬 신호가 약해도 실측 확인 자체에 가치를 두는 선례 있음).
- 블렌드 가중치(w_ebglmm)는 fold0/fold2 결과를 쓰지 않고, v26의 0.7:0.3을
  정했던 방식과 동일하게 "프로덕션 전체 데이터(2019~2024) 5% calib
  carve-out"에서 새로 그리드서치(저자유도 1개, CatBoost:retrieval=0.7:0.3는
  고정 -- 8/27 조인트 탐색이 calib 과적합을 일으켰던 전례 때문에 의도적으로
  자유도를 1개로 제한).

블렌드 공식(production 표준, v26 script.py에서 확인됨): raw 확률을 가중합 ->
그 합 하나에만 Platt calibration.
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
V34_DIR = os.path.dirname(__file__)
V34_MODEL_DIR = os.path.join(V34_DIR, "model")
V34_OUT_DIR = os.path.join(V34_DIR, "output")

EBGLMM_SEASON_WINDOW = (2024, 2024)  # production의 "최근 1시즌"(test=2025 직전)
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# ---- v25/train_final.py 함수 재사용(finalize_w07.py와 동일 방식) ----
g = {"__file__": os.path.join(V25_DIR, "train_final.py"), "__name__": "finalize"}
exec(open(os.path.join(V25_DIR, "train_final.py"), encoding="utf-8").read().split("def main()")[0], g)
TARGET_COL, ID_COL, SEED = g["TARGET_COL"], g["ID_COL"], g["SEED"]

# ================= EB-GLMM(last1season) -- 자체 포함(다른 스크립트 의존 없음) =================
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
    print("=== v34: CatBoost+retrieval(0.7:0.3, 고정) + EB-GLMM(last1season) 3-way ===", flush=True)
    print("Load data(전체 이력 2019~2024) + split(재현, v26과 동일 split)...", flush=True)
    df = g["load_data"]()
    df = g["add_risk_score_drop_ingredients"](df)
    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    y_calib = calib_df[TARGET_COL].to_numpy(dtype=float)
    print(f" train_sub={len(train_sub_df)} calib={len(calib_df)}", flush=True)

    print("\nCatBoost raw 예측(calib, 재학습 없음)...", flush=True)
    cb_id_mappings = g["build_catboost_id_mappings"](train_sub_df)
    X_calib_cb = g["build_catboost_features"](calib_df, cb_id_mappings)
    cb_model = CatBoostClassifier()
    cb_model.load_model(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"))
    cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

    print("\nRetrieval raw 예측(calib, 재학습 없음, 참조=1,401,337행)...", flush=True)
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

    print("\nEB-GLMM(last1season=2024) 학습...", flush=True)
    t0 = time.time()
    predict_raw_fn, ebglmm_state = train_ebglmm_production(train_sub_df, EBGLMM_SEASON_WINDOW)
    eb_calib_raw = predict_raw_fn(add_count_base(calib_df))
    print(f" EB-GLMM 학습+calib예측 완료({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(V34_OUT_DIR, exist_ok=True)
    np.save(os.path.join(V34_OUT_DIR, "cb_calib_raw.npy"), cb_calib_raw)
    np.save(os.path.join(V34_OUT_DIR, "nca_calib_raw.npy"), nca_calib_raw)
    np.save(os.path.join(V34_OUT_DIR, "eb_calib_raw.npy"), eb_calib_raw)
    np.save(os.path.join(V34_OUT_DIR, "y_calib.npy"), y_calib)
    print(" raw 예측 캐시 저장 완료(향후 가중치 재선택 시 재계산 불필요)", flush=True)

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
          f"(참고: w_ebglmm=0.00일 때 calib_bss={grid_log[0]['calib_bss']:.2f} == 순수 CatBoost+retrieval(0.7:0.3))",
          flush=True)

    os.makedirs(V34_MODEL_DIR, exist_ok=True)
    os.makedirs(V34_OUT_DIR, exist_ok=True)
    joblib.dump(ebglmm_state, os.path.join(V34_MODEL_DIR, "ebglmm_state.pkl"))

    with open(os.path.join(V34_OUT_DIR, "metrics_v34.json"), "w", encoding="utf-8") as f:
        json.dump({
            "w_catboost_retrieval": W_CATBOOST_RETRIEVAL,
            "best_w_ebglmm": best["w_ebglmm"],
            "carveout_calib_bss": best["calib_bss"],
            "calib_bss_no_ebglmm_reference": grid_log[0]["calib_bss"],
            "calibration_a": best["a"], "calibration_b": best["b"],
            "grid_log": grid_log,
            "ebglmm_season_window": EBGLMM_SEASON_WINDOW,
            "n_train_sub": len(train_sub_df), "n_calib": len(calib_df),
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(V26_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["blend_weight_ebglmm"] = best["w_ebglmm"]
    meta["calibration"] = {"method": "platt_sigmoid", "a": best["a"], "b": best["b"]}
    with open(os.path.join(V34_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {V34_MODEL_DIR}\\ebglmm_state.pkl, feature_meta.json / {V34_OUT_DIR}\\metrics_v34.json", flush=True)


if __name__ == "__main__":
    main()
