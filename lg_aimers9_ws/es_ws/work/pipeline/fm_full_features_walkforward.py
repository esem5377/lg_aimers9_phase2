"""FM이 fold2_2024에서 무너진 게 "매치업 신호 자체의 한계"가 아니라 "쓸 수
있는 피처가 너무 좁아서(범주형 소수만)"였는지 확인 -- CatBoost가 쓰는 수치형
피처(as-of 누적률, 트랙맨 구속/회전수 등)까지 FM에 다 넣어서 재검증.

수치형 필드는 표준 FM 확장 방식으로 처리: 필드별 임베딩 벡터에 스케일된
수치값을 곱해서(x_i * v_i) 다른 필드들과 동일하게 상호작용 항에 참여시킴.
범주형은 기존과 동일(one-hot 임베딩 lookup).
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
CONTEXT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model/trackman_context.pkl"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/fm_full_features_walkforward.json"
TARGET_COL = "control_success"
SEED = 42
EMBED_DIM = 16
EPOCHS = 15
BATCH_SIZE = 4096
LR = 5e-3
WEIGHT_DECAY = 1e-5

CAT_FIELDS = [
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "base_state", "top_bottom", "game_type",
    "pitcher_team_id", "batter_team_id",
]
NUM_FIELDS = [
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "tk_cnt_rel_speed_mean", "tk_cnt_spin_rate_mean", "tk_cnt_induced_vert_break_mean",
    "tk_cnt_horz_break_mean", "tk_cnt_fastball_rate", "tk_cnt_breaking_rate",
    "tk_cnt_offspeed_rate", "tk_cnt_n",
    "tk_hand_rel_speed_mean", "tk_hand_spin_rate_mean", "tk_hand_induced_vert_break_mean",
    "tk_hand_horz_break_mean", "tk_hand_fastball_rate", "tk_hand_breaking_rate",
    "tk_hand_offspeed_rate", "tk_hand_n",
    "tk_inn_rel_speed_mean", "tk_inn_spin_rate_mean", "tk_inn_induced_vert_break_mean",
    "tk_inn_horz_break_mean", "tk_inn_fastball_rate", "tk_inn_breaking_rate",
    "tk_inn_offspeed_rate", "tk_inn_n",
]


def log(msg):
    print(msg, flush=True)


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


class FullFM(nn.Module):
    def __init__(self, cat_cardinalities, num_fields, embed_dim):
        super().__init__()
        self.cat_fields = list(cat_cardinalities.keys())
        self.num_fields = num_fields
        self.w0 = nn.Parameter(torch.zeros(1))
        self.cat_biases = nn.ModuleDict()
        self.cat_embeds = nn.ModuleDict()
        for f, card in cat_cardinalities.items():
            self.cat_biases[f] = nn.Embedding(card + 1, 1, padding_idx=card)
            self.cat_embeds[f] = nn.Embedding(card + 1, embed_dim, padding_idx=card)
            nn.init.zeros_(self.cat_biases[f].weight)
            nn.init.normal_(self.cat_embeds[f].weight, std=0.01)
        self.num_bias = nn.Parameter(torch.zeros(len(num_fields)))
        self.num_embed = nn.Parameter(torch.randn(len(num_fields), embed_dim) * 0.01)

    def forward(self, cat_tensors, x_num):
        linear = self.w0 + (x_num * self.num_bias).sum(dim=1)
        cat_embs = []
        for f in self.cat_fields:
            linear = linear + self.cat_biases[f](cat_tensors[f]).squeeze(1)
            cat_embs.append(self.cat_embeds[f](cat_tensors[f]))  # (batch, embed_dim)

        num_contrib = x_num.unsqueeze(2) * self.num_embed.unsqueeze(0)  # (batch, n_num, embed_dim)
        if cat_embs:
            cat_stack = torch.stack(cat_embs, dim=1)  # (batch, n_cat, embed_dim)
            all_stack = torch.cat([cat_stack, num_contrib], dim=1)  # (batch, n_cat+n_num, embed_dim)
        else:
            all_stack = num_contrib

        sum_sq = all_stack.sum(dim=1).pow(2).sum(dim=1)
        sq_sum = all_stack.pow(2).sum(dim=(1, 2))
        interaction = 0.5 * (sum_sq - sq_sum)
        return linear + interaction


def build_mappings(train_df):
    mappings, cardinalities = {}, {}
    for f in CAT_FIELDS:
        uniq = sorted(train_df[f].astype(str).unique())
        mappings[f] = {v: i for i, v in enumerate(uniq)}
        cardinalities[f] = len(uniq)
    return mappings, cardinalities


def encode_cat(df, mappings):
    out = {}
    for f, m in mappings.items():
        card = len(m)
        out[f] = df[f].astype(str).map(m).fillna(card).astype(int).values
    return out


def prep_numeric(df, medians, scaler):
    X = df[NUM_FIELDS].fillna(medians)
    return scaler.transform(X.values.astype(np.float32)).astype(np.float32)


def to_cat_tensors(enc, device):
    return {f: torch.tensor(v, dtype=torch.long, device=device) for f, v in enc.items()}


def train_fm(cat_train, x_num_train, y_train, cardinalities, device):
    model = FullFM(cardinalities, NUM_FIELDS, EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    cat_t = to_cat_tensors(cat_train, device)
    x_num_t = torch.tensor(x_num_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(y_train)

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED + epoch)
        idx = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            batch = idx[i:i + BATCH_SIZE]
            batch_cat = {f: v[batch] for f, v in cat_t.items()}
            batch_num = x_num_t[batch]
            opt.zero_grad()
            logit = model(batch_cat, batch_num)
            loss = loss_fn(logit, y_t[batch])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"    epoch {epoch+1}/{EPOCHS} loss={epoch_loss/n_batches:.5f}")
    return model


@torch.no_grad()
def predict_fm(model, cat_enc, x_num, device):
    model.eval()
    cat_t = to_cat_tensors(cat_enc, device)
    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    logit = model(cat_t, x_num_t)
    return torch.sigmoid(logit).cpu().numpy()


def run_fold(tag, train_df, eval_df, device):
    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)

    mappings, cardinalities = build_mappings(train_sub)
    cat_train = encode_cat(train_sub, mappings)
    cat_calib = encode_cat(calib, mappings)
    cat_eval = encode_cat(eval_df, mappings)

    medians = train_sub[NUM_FIELDS].median()
    scaler = StandardScaler().fit(train_sub[NUM_FIELDS].fillna(medians).values.astype(np.float32))
    x_num_train = prep_numeric(train_sub, medians, scaler)
    x_num_calib = prep_numeric(calib, medians, scaler)
    x_num_eval = prep_numeric(eval_df, medians, scaler)

    y_train = train_sub[TARGET_COL].values.astype(np.float32)
    y_calib = calib[TARGET_COL].values
    y_eval = eval_df[TARGET_COL].values

    t0 = time.time()
    model = train_fm(cat_train, x_num_train, y_train, cardinalities, device)
    elapsed = time.time() - t0

    calib_raw = predict_fm(model, cat_calib, x_num_calib, device)
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = predict_fm(model, cat_eval, x_num_eval, device)
    eval_calib = apply_platt(eval_raw, a, b)

    result = {
        "tag": tag,
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    log(f"  [{tag}] auc={result['auc']:.4f} bss_calib={result['bss_calibrated']:.2f} ({elapsed:.0f}s)")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    log(f"shape={df.shape}")

    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }
    baselines = {"fold0_2022": 2386.4930909141212, "fold2_2024": 832.4060438655523}

    results = {}
    for fold_name, (train_df, eval_df) in fold_specs.items():
        log(f"\n=== {fold_name} ===")
        results[fold_name] = run_fold(fold_name, train_df, eval_df, device)
        results[fold_name]["baseline_calibrated"] = baselines[fold_name]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("\n=== SUMMARY (전체 피처 FM vs CatBoost baseline) ===")
    for fn in fold_specs:
        log(f"  {fn}: FM auc={results[fn]['auc']:.4f} bss_calib={results[fn]['bss_calibrated']:.2f}  "
            f"CatBoost_baseline={baselines[fn]:.2f}")
    log(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
