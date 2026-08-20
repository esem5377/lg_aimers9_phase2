"""es_ws build_trackman_context.py와 동일한 그룹핑 구조(count_state/
hand_matchup/inning_state, +40점 기여 검증됨)를 그대로 쓰되, METRIC_COLS를
4개(rel_speed/spin_rate/induced_vert_break/horz_break) -> 8개로 확장.

추가된 4개(extension/rel_height/rel_side/zone_speed)는 지금까지 이
프로젝트에서 한 번도 안 쓰인 컬럼들 -- 릴리스 익스텐션/높이/좌우위치/
플레이트 통과속도는 투구 디셉션과 직결되는 실전 지표라 새로운 신호일
가능성이 있음. 8/20 EDA에서 extension(음수 불가)/rel_height(0에 가까운
값 불가)에 물리적으로 불가능한 값이 확인돼 이 4개 신규 컬럼만 1%/99%
winsorize 적용(기존 4개는 winsorize가 이미 실패했던 컬럼이라 건드리지
않음 -- 8/20 v13 실험, -6점).
"""
import os

import joblib
import pandas as pd

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v15_extra_trackman_metrics\model"

HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}

METRIC_COLS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]
NEW_METRIC_COLS = ["extension", "rel_height", "rel_side", "zone_speed"]
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


def winsorize_new_cols(df):
    df = df.copy()
    for c in NEW_METRIC_COLS:
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

    print("Winsorize 신규 METRIC_COLS (1%/99%)...")
    th = winsorize_new_cols(th)

    context = {}
    for name, keys in GROUPINGS.items():
        table = build_group_table(th, keys, {"count_state": "tk_cnt", "hand_matchup": "tk_hand", "inning_state": "tk_inn"}[name])
        print(f"\n[{name}] keys={keys} rows={len(table)} cols={list(table.columns)}")
        context[name] = {"keys": keys, "table": table}

    out_path = os.path.join(MODEL_DIR, "trackman_context.pkl")
    joblib.dump(context, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
