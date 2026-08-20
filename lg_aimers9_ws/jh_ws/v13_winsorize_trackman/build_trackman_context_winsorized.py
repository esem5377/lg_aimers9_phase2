"""es_ws build_trackman_context.py 동일 로직 + METRIC_COLS 4개(rel_speed/
spin_rate/induced_vert_break/horz_break)에 대해 상황별 평균을 내기 전에
1%/99% percentile로 clip(winsorize)하는 것만 추가.

이상치 실측(2026-08-20 EDA): induced_vert_break/horz_break는 99.9%
percentile과 max 사이 격차가 크고(예: ivb 99.9%=71.7 vs max=153.3),
spin_rate min(434.9rpm)도 비정상적으로 낮아 극단치 몇 개가 상황별 평균을
흔들 가능성이 있음. extension/rel_height도 물리적으로 불가능한 값(음수
extension, 0에 가까운 rel_height)이 있었지만 애초에 METRIC_COLS에
포함 안 돼 있어(build_trackman_context.py 원본 확인) 영향 없음.
"""
import os

import joblib
import pandas as pd

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v13_winsorize_trackman\model"

HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}

METRIC_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99

GROUPINGS = {
    "count_state": ["balls_before", "strikes_before", "outs_before"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "inning_state": ["inning", "top_bottom"],
}


def load_trackman():
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    df["top_bottom"] = df["top_bottom"].map(TOP_BOTTOM_MAP)
    return df


def winsorize(df):
    df = df.copy()
    for c in METRIC_COLS:
        lo, hi = df[c].quantile(WINSOR_LOW), df[c].quantile(WINSOR_HIGH)
        n_clipped = int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lower=lo, upper=hi)
        print(f" winsorize {c}: [{lo:.2f}, {hi:.2f}]  clipped_rows={n_clipped}")
    return df


def build_group_table(df, keys, prefix):
    g = df.groupby(keys, dropna=False)
    out = g[METRIC_COLS].mean()
    out.columns = [f"{prefix}_{c}_mean" for c in out.columns]

    pitch_rate = (
        pd.crosstab(
            [df[k] for k in keys], df["pitch_type_group"], normalize="index"
        )
    )
    for pg in PITCH_GROUPS:
        out[f"{prefix}_{pg}_rate"] = pitch_rate.get(pg, 0.0)

    out[f"{prefix}_n"] = g.size()
    return out.reset_index()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Load trackman_history...")
    th = load_trackman()
    print(f" shape={th.shape}")

    print("Winsorize METRIC_COLS (1%/99%)...")
    th = winsorize(th)

    context = {}
    for name, keys in GROUPINGS.items():
        table = build_group_table(th, keys, {"count_state": "tk_cnt", "hand_matchup": "tk_hand", "inning_state": "tk_inn"}[name])
        print(f"\n[{name}] keys={keys} rows={len(table)}")
        context[name] = {"keys": keys, "table": table}

    out_path = os.path.join(MODEL_DIR, "trackman_context.pkl")
    joblib.dump(context, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
