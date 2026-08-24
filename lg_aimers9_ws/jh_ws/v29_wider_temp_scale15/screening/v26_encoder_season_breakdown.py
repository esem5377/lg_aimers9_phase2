"""가설(a) 검증: v29(1013점, -10)의 실패가 "fold 전용으로 작게 학습한
encoder"와 "프로덕션의 v26 encoder(2019~2024 거의 전체로 학습)" 사이의
차이 때문이었는지 확인하기 위해, v26의 실제 프로덕션 encoder/CatBoost/
참조임베딩(140만행)을 그대로 쓰되 calib_df(랜덤 5% 홀드아웃, production과
동일 split)를 season별로 쪼개서 scale=1.0 vs scale=15.0 델타를 비교한다.

주의: 이건 fold0/fold2 같은 진짜 미래-시즌 walk-forward가 아니다 -- v26의
참조집합(train_sub_df)에는 2024 데이터의 대부분이 이미 포함돼 있고, calib_df는
그 중 무작위 5%일 뿐이다(같은 분포 내 랜덤 홀드아웃). 그래도 "같은 encoder를
쓰되 season만 나눠서 scale의 효과가 season마다 다른지"는 확인 가능하다.
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

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
V26_MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v26_retrieval_blend_w07\model"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "v26_encoder_season_breakdown_results.json")

SEED = 42
ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
ALL_CAT_FOR_NN = ["pitcher_id", "batter_id"] + TEAM_COLS + CAT_COLS
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS

EMBED_DIMS = {
    "pitcher_id": 24, "batter_id": 24, "pitcher_team_id": 6, "batter_team_id": 6,
    "top_bottom": 2, "game_type": 2, "pitcher_hand": 2, "batter_hand": 2, "base_state": 4,
}
ENCODER_HIDDEN = [256, 128]
EMBED_OUT_DIM = 32
REFERENCE_CHUNK = 20000
QUERY_CHUNK = 2000
TEMP_SCALES = [1.0, 15.0]
W_CATBOOST = 0.5


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
    context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df.drop(columns=INGREDIENT_COLS)


def build_catboost_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_catboost_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str)
    return X


class RowEncoder(nn.Module):
    def __init__(self, cat_cardinalities, n_numeric):
        super().__init__()
        self.embeds = nn.ModuleDict()
        embed_out = 0
        for col, card in cat_cardinalities.items():
            dim = EMBED_DIMS[col]
            self.embeds[col] = nn.Embedding(card + 1, dim)
            embed_out += dim
        in_dim = embed_out + n_numeric
        layers = []
        prev = in_dim
        for h in ENCODER_HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.0)]
            prev = h
        layers += [nn.Linear(prev, EMBED_OUT_DIM)]
        self.mlp = nn.Sequential(*layers)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def encode(self, cat_tensors, x_num):
        parts = [self.embeds[col](cat_tensors[col]) for col in self.embeds]
        parts.append(x_num)
        x = torch.cat(parts, dim=1)
        z = self.mlp(x)
        return nn.functional.normalize(z, dim=1)


def build_nn_cat_mappings(df):
    mappings, cardinalities = {}, {}
    for col in ALL_CAT_FOR_NN:
        uniq = sorted(df[col].astype(str).unique())
        mappings[col] = {v: i for i, v in enumerate(uniq)}
        cardinalities[col] = len(uniq)
    return mappings, cardinalities


def encode_cats(df, mappings):
    out = {}
    for col, m in mappings.items():
        oov = len(m)
        out[col] = df[col].astype(str).map(m).fillna(oov).astype(int).values
    return out


def prep_numeric_transform(df, numeric_cols, medians, scaler):
    X = df[numeric_cols].copy()
    isna_flags = X.isna().astype(np.float32)
    isna_flags.columns = [f"{c}__isna" for c in numeric_cols]
    X = X.fillna(medians)
    X_all = pd.concat([X, isna_flags], axis=1)
    return scaler.transform(X_all.values.astype(np.float32)).astype(np.float32)


@torch.no_grad()
def retrieve_predict_online(model, cat_query, x_num_query, z_ref, y_ref, scale):
    """온라인(스트리밍) softmax 누적 -- production script.py와 동일 원리.
    참조가 140만행(70 ref_chunk)이라 모든 ref_chunk의 유사도 행렬을 리스트에
    쌓아두면 쿼리 청크당 최대 11GB(2000x20000x70x4bytes)를 점유해 극심한
    스와핑을 유발함(8/25 세션에서 실제로 겪은 버그와 동일 패턴) -- 절대
    재사용하지 말 것. ref_chunk 하나 처리 후 즉시 버리고 누적값만 유지."""
    model.eval()
    base_temp = torch.exp(model.log_temp).clamp(min=1e-3, max=10.0)
    temp = base_temp * scale
    cat_query_t = {c: torch.tensor(v, dtype=torch.long) for c, v in cat_query.items()}
    x_num_query_t = torch.tensor(x_num_query, dtype=torch.float32)
    n_q = x_num_query_t.shape[0]
    preds = np.zeros(n_q, dtype=np.float64)

    for qi in range(0, n_q, QUERY_CHUNK):
        cat_q_chunk = {c: v[qi:qi + QUERY_CHUNK] for c, v in cat_query_t.items()}
        x_q_chunk = x_num_query_t[qi:qi + QUERY_CHUNK]
        z_q = model.encode(cat_q_chunk, x_q_chunk)

        running_max = torch.full((z_q.shape[0],), float("-inf"))
        running_numer = torch.zeros(z_q.shape[0])
        running_denom = torch.zeros(z_q.shape[0])
        for i in range(0, z_ref.shape[0], REFERENCE_CHUNK):
            z_ref_c = z_ref[i:i + REFERENCE_CHUNK]
            y_ref_c = y_ref[i:i + REFERENCE_CHUNK]
            sim = (z_q @ z_ref_c.T) / temp
            chunk_max = sim.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)
            scale_old = torch.exp(running_max - new_max)
            scale_old = torch.where(torch.isfinite(scale_old), scale_old, torch.zeros_like(scale_old))
            w_chunk = torch.exp(sim - new_max.unsqueeze(1))
            running_numer = running_numer * scale_old + (w_chunk * y_ref_c.unsqueeze(0)).sum(dim=1)
            running_denom = running_denom * scale_old + w_chunk.sum(dim=1)
            running_max = new_max
        pred_chunk = (running_numer / running_denom).clamp(1e-6, 1 - 1e-6)
        preds[qi:qi + QUERY_CHUNK] = pred_chunk.numpy()
        if qi % (QUERY_CHUNK * 10) == 0:
            print(f"    query chunk {qi}/{n_q}", flush=True)

    return preds


def main():
    print("Load data + reproduce v26 split...", flush=True)
    df = load_data()
    df = add_risk_score_drop_ingredients(df)
    y_all = df[TARGET_COL]
    train_sub_df, calib_df = train_test_split(df, test_size=0.05, stratify=y_all, random_state=SEED)
    train_sub_df = train_sub_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    print(f" train_sub={train_sub_df.shape}  calib={calib_df.shape}", flush=True)
    print(f" calib season counts:\n{calib_df['season'].value_counts().sort_index()}", flush=True)

    print("CatBoost 예측 계산 (v26 재사용)...", flush=True)
    cb_id_mappings = build_catboost_id_mappings(train_sub_df)
    X_calib_cb = build_catboost_features(calib_df, cb_id_mappings)
    from catboost import CatBoostClassifier
    cb_model = CatBoostClassifier()
    cb_model.load_model(os.path.join(V26_MODEL_DIR, "catboost_seed42.cbm"))
    cb_calib_raw = cb_model.predict_proba(X_calib_cb)[:, 1]

    print("Retrieval 준비 (v26 encoder/참조임베딩 재사용)...", flush=True)
    nn_cat_mappings, cardinalities = build_nn_cat_mappings(train_sub_df)
    numeric_cols = [c for c in train_sub_df.columns if c not in [ID_COL, TARGET_COL] + ALL_CAT_FOR_NN]
    cat_calib_nn = encode_cats(calib_df, nn_cat_mappings)
    numeric_prep = joblib.load(os.path.join(V26_MODEL_DIR, "numeric_prep.pkl"))
    medians, scaler = numeric_prep["medians"], numeric_prep["scaler"]
    x_num_calib = prep_numeric_transform(calib_df, numeric_cols, medians, scaler)

    encoder = RowEncoder(cardinalities, len(numeric_cols) * 2)
    encoder.load_state_dict(torch.load(os.path.join(V26_MODEL_DIR, "retrieval_encoder.pt"), map_location="cpu"))
    encoder.eval()
    z_ref = torch.tensor(np.load(os.path.join(V26_MODEL_DIR, "reference_embeddings.npy")), dtype=torch.float32)
    y_ref = torch.tensor(np.load(os.path.join(V26_MODEL_DIR, "reference_labels.npy")), dtype=torch.float32)
    print(f" reference_size={len(y_ref)}", flush=True)

    calib_preds = {}
    for s in TEMP_SCALES:
        print(f"\n[scale={s}] calib_df 전체({len(calib_df)}행) 온라인 스트리밍 계산...", flush=True)
        t0 = time.time()
        calib_preds[s] = retrieve_predict_online(encoder, cat_calib_nn, x_num_calib, z_ref, y_ref, s)
        print(f" 완료 ({time.time()-t0:.1f}s)", flush=True)

    y_calib_arr = calib_df[TARGET_COL].values
    a_cb, b_cb = fit_platt(cb_calib_raw, calib_df[TARGET_COL])
    cb_calib_calibrated = apply_platt(cb_calib_raw, a_cb, b_cb)

    results = {}
    for s in TEMP_SCALES:
        blend_raw_all = W_CATBOOST * cb_calib_raw + (1 - W_CATBOOST) * calib_preds[s]
        a, b = fit_platt(blend_raw_all, calib_df[TARGET_COL])
        blend_calib_all = apply_platt(blend_raw_all, a, b)
        overall_bss = bss_score(blend_calib_all, y_calib_arr)
        cb_only_bss = bss_score(cb_calib_calibrated, y_calib_arr)

        season_bss = {}
        for season in sorted(calib_df["season"].unique()):
            mask = (calib_df["season"] == season).values
            n = int(mask.sum())
            bss_season = bss_score(blend_calib_all[mask], y_calib_arr[mask])
            cb_only_season_bss = bss_score(cb_calib_calibrated[mask], y_calib_arr[mask])
            season_bss[int(season)] = {
                "n": n, "blend_bss": bss_season, "cb_only_bss": cb_only_season_bss,
                "delta_vs_cb": bss_season - cb_only_season_bss,
            }
            print(f"  scale={s} season={season} n={n} blend_bss={bss_season:.2f} "
                  f"cb_only_bss={cb_only_season_bss:.2f} delta={bss_season-cb_only_season_bss:+.2f}", flush=True)

        results[str(s)] = {
            "overall_blend_bss": overall_bss, "overall_cb_only_bss": cb_only_bss,
            "overall_delta": overall_bss - cb_only_bss, "by_season": season_bss,
        }
        print(f" scale={s} OVERALL blend_bss={overall_bss:.2f} cb_only_bss={cb_only_bss:.2f} "
              f"delta={overall_bss-cb_only_bss:+.2f}", flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
