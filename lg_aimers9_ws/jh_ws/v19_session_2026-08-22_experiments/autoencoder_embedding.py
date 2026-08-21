"""투수(및 타자) 오토인코더 임베딩 -- 완전 비지도(정답 라벨 전혀 안 씀,
순수 재구성 손실만으로 학습) 방식으로 투수/타자 "타입"을 저차원으로
표현해 974 레시피에 피처로 추가. 8/21 시도했던 DeepFM 임베딩(지도 신호가
섞인 팩터라이제이션 머신)과 달리 이번엔 target을 전혀 보지 않는다는 점이
다름 -- 사용자가 명시적으로 이 차이를 확인해보고 싶어해서 시도.

설계:
  - 투수별 정적 프로파일 = train 파티션 내에서 그 투수가 등장하는 행들의
    asof_pitcher_* 11개 컬럼(success/reverse/middle/ball/strike_rate +
    prev1/3/5_success/middle_rate) 평균. target(control_success)은
    입력에 전혀 포함 안 함.
  - 오토인코더(11 -> 8 -> BOTTLENECK(4) -> 8 -> 11, MSE 재구성 손실)를
    train 파티션의 고유 투수 프로파일 행렬에만 fit(정답 라벨 미사용,
    순수 비지도). 학습 후 bottleneck(4차원)을 투수별 임베딩으로 추출.
  - eval 파티션에만 있는 신규 투수(OOV)는 train 파티션 전체 평균
    프로파일을 인코더에 통과시킨 임베딩으로 대체(정직성 원칙: eval 정답
    미사용, eval 자체 통계도 미사용).
  - 타자는 asof_batter_* 컬럼이 2개뿐이라 오토인코더 의미가 약해 스킵.
  - 이 4개 임베딩 컬럼을 974 레시피의 71개 피처에 추가(72 -> 75, CAT_COLS
    네이티브/RAW_ID_COLS/trackman context 등 나머지 전부 동일)해 CatBoost
    재학습, fold0/fold2 season split(iterations=1000)으로 baseline(단일
    모델, fold0=2386.49/fold2=832.41 -- 8/21 세션 초반 정규화 실험의
    baseline과 동일 스케일)과 비교.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "autoencoder_embedding_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
ITERATIONS = 1000
SEED = 42
BOTTLENECK_DIM = 4

PITCHER_PROFILE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

KNOWN_BASELINE = {"fold0_2022": 2386.4930909141212, "fold2_2024": 832.4060438655523}


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8), nn.ReLU(), nn.Linear(8, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


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


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_id_mappings(train_df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(train_df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_pitcher_embeddings(train_df):
    """train 파티션만으로 투수별 프로파일 -> 표준화 -> 오토인코더(비지도,
    라벨 미사용) -> bottleneck 임베딩. eval에만 있는 신규 투수는 train
    전체 평균 프로파일의 임베딩으로 대체."""
    profile = train_df.groupby(train_df["pitcher_id"].astype(str))[PITCHER_PROFILE_COLS].mean()
    global_mean = train_df[PITCHER_PROFILE_COLS].mean()
    profile = profile.fillna(global_mean)  # cold-start NaN -> 전역 평균 (train 내부 기준)

    mu = profile.mean()
    sigma = profile.std().replace(0, 1.0)
    profile_std = (profile - mu) / sigma
    global_mean_std = ((global_mean - mu) / sigma).fillna(0.0)

    X_profile = torch.tensor(profile_std.values, dtype=torch.float32)
    model = AutoEncoder(len(PITCHER_PROFILE_COLS), BOTTLENECK_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    torch.manual_seed(SEED)
    model.train()
    for epoch in range(300):
        opt.zero_grad()
        recon, _ = model(X_profile)
        loss = loss_fn(recon, X_profile)
        loss.backward()
        opt.step()
    final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        _, emb = model(X_profile)
        unknown_vec = torch.tensor(global_mean_std.values, dtype=torch.float32).unsqueeze(0)
        _, unknown_emb = model(unknown_vec)

    emb_df = pd.DataFrame(
        emb.numpy(), index=profile.index,
        columns=[f"pitcher_emb_{i}" for i in range(BOTTLENECK_DIM)],
    )
    unknown_emb_vec = unknown_emb.numpy()[0]
    return emb_df, unknown_emb_vec, final_loss


def attach_pitcher_embeddings(df, emb_df, unknown_emb_vec):
    df = df.copy()
    pid_str = df["pitcher_id"].astype(str)
    joined = emb_df.reindex(pid_str.values)
    for i, col in enumerate(emb_df.columns):
        vals = joined[col].values
        vals = np.where(pd.isna(vals), unknown_emb_vec[i], vals)
        df[col] = vals
    return df


def build_features(df, id_mappings, emb_cols):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def run_variant(tag, train_df, eval_df, use_embedding):
    id_mappings = build_id_mappings(train_df)
    y_train_full = train_df[TARGET_COL]

    train_sub_df, calib_df = train_test_split(
        train_df, test_size=0.05, stratify=y_train_full, random_state=SEED,
    )

    emb_cols = []
    if use_embedding:
        emb_df, unknown_emb_vec, ae_loss = build_pitcher_embeddings(train_sub_df)
        train_sub_df = attach_pitcher_embeddings(train_sub_df, emb_df, unknown_emb_vec)
        calib_df = attach_pitcher_embeddings(calib_df, emb_df, unknown_emb_vec)
        eval_df = attach_pitcher_embeddings(eval_df, emb_df, unknown_emb_vec)
        emb_cols = list(emb_df.columns)
        print(f"  [{tag}] autoencoder 최종 재구성 MSE(표준화 공간)={ae_loss:.4f}  "
              f"n_unique_pitcher={len(emb_df)}", flush=True)
    else:
        ae_loss = None

    X_train_sub = build_features(train_sub_df, id_mappings, emb_cols)
    y_train_sub = train_sub_df[TARGET_COL]
    X_calib = build_features(calib_df, id_mappings, emb_cols)
    y_calib = calib_df[TARGET_COL]
    X_eval = build_features(eval_df, id_mappings, emb_cols)
    y_eval = eval_df[TARGET_COL]

    cat_idx = [X_train_sub.columns.get_loc(c) for c in CAT_COLS]

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=100,
        verbose=False, thread_count=-1, **BEST_PARAMS,
    )
    model.fit(X_train_sub, y_train_sub, eval_set=(X_calib, y_calib))
    elapsed = time.time() - t0

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = model.predict_proba(X_eval)[:, 1]
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag, "n_features": X_train_sub.shape[1],
        "best_iteration": model.get_best_iteration(),
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
        "autoencoder_final_recon_mse": ae_loss,
    }
    print(
        f"  [{tag}] n_features={result['n_features']} auc={result['auc']:.4f} "
        f"bss_calib={result['bss_calibrated']:.2f} ({elapsed:.1f}s)",
        flush=True,
    )
    return result


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    print(f" shape={df.shape}", flush=True)

    all_results = {}
    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    for fold_name, (train_df, eval_df) in fold_specs.items():
        print(f"\n=== {fold_name} ===", flush=True)
        baseline_r = run_variant(f"{fold_name}/baseline", train_df, eval_df, use_embedding=False)
        emb_r = run_variant(f"{fold_name}/autoencoder_emb", train_df, eval_df, use_embedding=True)
        all_results[fold_name] = {
            "baseline": baseline_r,
            "with_embedding": emb_r,
            "delta_calibrated": emb_r["bss_calibrated"] - baseline_r["bss_calibrated"],
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    all_positive = all(v["delta_calibrated"] > 0 for v in all_results.values())
    summary = {k: v["delta_calibrated"] for k, v in all_results.items()}
    summary["all_axes_positive"] = all_positive
    all_results["summary"] = summary

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (baseline -> +autoencoder embedding, calibrated BSS) ===", flush=True)
    for k, v in all_results.items():
        if k == "summary":
            continue
        print(f"  {k}: {v['baseline']['bss_calibrated']:.2f} -> {v['with_embedding']['bss_calibrated']:.2f}  (delta {v['delta_calibrated']:+.2f})", flush=True)
    print(f"\n  ALL AXES POSITIVE: {all_positive}", flush=True)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
