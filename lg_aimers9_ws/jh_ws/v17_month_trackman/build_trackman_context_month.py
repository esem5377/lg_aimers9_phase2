"""es_ws build_trackman_context.py의 검증된 구조(count_state/hand_matchup/
inning_state, +40점 기여 확인됨)를 그대로 두고, 4번째 독립 축으로
`game_month`를 추가. METRIC_COLS는 원래 4개(rel_speed/spin_rate/
induced_vert_break/horz_break) 그대로 -- 8개 확장판(v15)은 이미 실제
제출로 -14.9 기각됐으므로 재사용 안 함.

도메인 근거: 초반 시즌(3~4월) 저온 환경에서 그립감 저하로 인한 제구
불안정은 야구에서 잘 알려진 현상. game_month는 train.csv와
trackman_history.csv 둘 다에 존재해 join 가능한, 지금까지 안 쓰인 축.

기존 3축과 마찬가지로 독립적으로 유지(다른 축과 결합 안 함) --
jh_ws sit_*(5축 combined groupby, 실패)가 아니라 es_ws 방식(단순 축 각각,
CatBoost가 트리 분기로 알아서 조합)을 따름.
"""
import os

import joblib
import pandas as pd

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
MODEL_DIR = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\jh_ws\v17_month_trackman\model"

HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}

METRIC_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]

GROUPINGS = {
    "count_state": ["balls_before", "strikes_before", "outs_before"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "inning_state": ["inning", "top_bottom"],
    "month_state": ["game_month"],
}

PREFIX_MAP = {
    "count_state": "tk_cnt", "hand_matchup": "tk_hand",
    "inning_state": "tk_inn", "month_state": "tk_month",
}


def load_trackman():
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    df["top_bottom"] = df["top_bottom"].map(TOP_BOTTOM_MAP)
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

    context = {}
    for name, keys in GROUPINGS.items():
        table = build_group_table(th, keys, PREFIX_MAP[name])
        print(f"\n[{name}] keys={keys} rows={len(table)}")
        if name == "month_state":
            print(table)
        context[name] = {"keys": keys, "table": table}

    out_path = os.path.join(MODEL_DIR, "trackman_context.pkl")
    joblib.dump(context, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
