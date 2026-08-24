"""투수x타자 매치업 상호작용을 잡는 Factorization Machine(bilinear) 모델을
CatBoost/retrieval과는 별개 축으로 추가할 수 있는지 확인. CatBoost는 트리
분기 구조상, retrieval은 전체 벡터 유사도 구조상 pitcher_id x batter_id의
"이 조합 특유의" 곱셈적 상호작용을 깔끔하게 못 잡을 수 있다는 가설.

오늘(8/24) 랜덤 carve-out으로 레시피 결정을 내렸다가 리더보드에서 -19 회귀를
겪은 직후라, 이번엔 처음부터 이 프로젝트의 검증된 walk-forward 방식
(fold0: train<=2021 -> eval=2022, fold2: train<=2023 -> eval=2024, 2023
자체는 known regime-shift 이상치라 skip)으로만 검증한다. 우선 FM 단독
신호가 있는지만 빠르게 확인(1단계) -- 있으면 CatBoost 블렌드까지 확장(2단계).

logit(pitcher,batter) = w0 + b_p[pitcher] + b_b[batter] + <e_p[pitcher], e_b[batter]>
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
OUT_PATH = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output/fm_matchup_walkforward.json"
TARGET_COL = "control_success"
SEED = 42
EMBED_DIM = 16
EPOCHS = 15
BATCH_SIZE = 4096
LR = 5e-3
WEIGHT_DECAY = 1e-4


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


class MatchupFM(nn.Module):
    def __init__(self, n_pitchers, n_batters, embed_dim):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b_p = nn.Embedding(n_pitchers + 1, 1, padding_idx=n_pitchers)
        self.b_b = nn.Embedding(n_batters + 1, 1, padding_idx=n_batters)
        self.e_p = nn.Embedding(n_pitchers + 1, embed_dim, padding_idx=n_pitchers)
        self.e_b = nn.Embedding(n_batters + 1, embed_dim, padding_idx=n_batters)
        nn.init.normal_(self.e_p.weight, std=0.01)
        nn.init.normal_(self.e_b.weight, std=0.01)
        nn.init.zeros_(self.b_p.weight)
        nn.init.zeros_(self.b_b.weight)

    def forward(self, pid, bid):
        interaction = (self.e_p(pid) * self.e_b(bid)).sum(dim=1)
        return self.w0 + self.b_p(pid).squeeze(1) + self.b_b(bid).squeeze(1) + interaction


def build_mapping(train_ids):
    uniq = sorted(train_ids.astype(str).unique())
    return {v: i for i, v in enumerate(uniq)}, len(uniq)


def encode(ids, mapping, n):
    return ids.astype(str).map(mapping).fillna(n).astype(int).values


def train_fm(pid_train, bid_train, y_train, n_p, n_b, device):
    model = MatchupFM(n_p, n_b, EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    pid_t = torch.tensor(pid_train, dtype=torch.long, device=device)
    bid_t = torch.tensor(bid_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(y_train)

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED + epoch)
        idx = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            batch = idx[i:i + BATCH_SIZE]
            opt.zero_grad()
            logit = model(pid_t[batch], bid_t[batch])
            loss = loss_fn(logit, y_t[batch])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"    epoch {epoch+1}/{EPOCHS} loss={epoch_loss/n_batches:.5f}")
    return model


@torch.no_grad()
def predict_fm(model, pid, bid, device):
    model.eval()
    pid_t = torch.tensor(pid, dtype=torch.long, device=device)
    bid_t = torch.tensor(bid, dtype=torch.long, device=device)
    logit = model(pid_t, bid_t)
    return torch.sigmoid(logit).cpu().numpy()


def run_fold(tag, train_df, eval_df, device):
    n_p_before = train_df["pitcher_id"].nunique()
    n_b_before = train_df["batter_id"].nunique()

    from sklearn.model_selection import train_test_split
    y_full = train_df[TARGET_COL]
    train_sub, calib = train_test_split(train_df, test_size=0.05, stratify=y_full, random_state=SEED)

    p_map, n_p = build_mapping(train_sub["pitcher_id"])
    b_map, n_b = build_mapping(train_sub["batter_id"])

    pid_train = encode(train_sub["pitcher_id"], p_map, n_p)
    bid_train = encode(train_sub["batter_id"], b_map, n_b)
    y_train = train_sub[TARGET_COL].values.astype(np.float32)

    pid_calib = encode(calib["pitcher_id"], p_map, n_p)
    bid_calib = encode(calib["batter_id"], b_map, n_b)
    y_calib = calib[TARGET_COL].values

    pid_eval = encode(eval_df["pitcher_id"], p_map, n_p)
    bid_eval = encode(eval_df["batter_id"], b_map, n_b)
    y_eval = eval_df[TARGET_COL].values

    t0 = time.time()
    model = train_fm(pid_train, bid_train, y_train, n_p, n_b, device)
    elapsed = time.time() - t0

    calib_raw = predict_fm(model, pid_calib, bid_calib, device)
    a, b = fit_platt(calib_raw, y_calib)
    eval_raw = predict_fm(model, pid_eval, bid_eval, device)
    eval_calib = apply_platt(eval_raw, a, b)

    n_pitcher_oov = int((pid_eval == n_p).sum())
    n_batter_oov = int((bid_eval == n_b).sum())
    n_pair_seen = int(pd.Series(list(zip(eval_df["pitcher_id"], eval_df["batter_id"]))).isin(
        set(zip(train_sub["pitcher_id"], train_sub["batter_id"]))).sum())

    result = {
        "tag": tag,
        "n_pitchers_train": n_p, "n_batters_train": n_b,
        "eval_pitcher_oov_rate": n_pitcher_oov / len(eval_df),
        "eval_batter_oov_rate": n_batter_oov / len(eval_df),
        "eval_exact_pair_seen_rate": n_pair_seen / len(eval_df),
        "auc": roc_auc_score(y_eval, eval_raw),
        "bss_raw": bss_score(eval_raw, y_eval),
        "bss_calibrated": bss_score(eval_calib, y_eval),
        "elapsed_sec": elapsed,
    }
    log(f"  [{tag}] auc={result['auc']:.4f} bss_calib={result['bss_calibrated']:.2f} "
        f"pair_seen_rate={result['eval_exact_pair_seen_rate']:.3f} ({elapsed:.0f}s)")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "season", "pitcher_id", "batter_id", TARGET_COL])
    log(f"shape={df.shape}")

    fold_specs = {
        "fold0_2022": (df[df["season"] <= 2021], df[df["season"] == 2022]),
        "fold2_2024": (df[df["season"] <= 2023], df[df["season"] == 2024]),
    }

    results = {}
    for fold_name, (train_df, eval_df) in fold_specs.items():
        log(f"\n=== {fold_name} ===")
        results[fold_name] = run_fold(fold_name, train_df, eval_df, device)

    # baseline(fold0=2386.49, fold2=832.41)은 control_risk_score_results.json에서 재사용
    baselines = {"fold0_2022": 2386.4930909141212, "fold2_2024": 832.4060438655523}
    for fn in fold_specs:
        results[fn]["baseline_calibrated"] = baselines[fn]
        results[fn]["delta_vs_baseline"] = results[fn]["bss_calibrated"] - baselines[fn]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("\n=== SUMMARY (FM 단독 vs CatBoost baseline, 같은 fold) ===")
    for fn in fold_specs:
        log(f"  {fn}: FM={results[fn]['bss_calibrated']:.2f}  CatBoost_baseline={baselines[fn]:.2f}  "
            f"delta={results[fn]['delta_vs_baseline']:+.2f}")
    log(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
