"""trackman_history.csv로 '상황(context) 단위' 투구 특성 통계 피처를 만든다.

trackman_history의 pitcher_trackman_id/batter_trackman_id/pitcher_team은
train.csv의 pitcher_id/batter_id/pitcher_team_id와 겹치는 값이 하나도 없는
별개의 익명화 체계다 (선수 단위 join 불가, data_description.md에서 경고한
"1:1 결합 테이블 아님"이 이 뜻). 대신 두 데이터에 공통으로 존재하는 상황
키(카운트/아웃, 투타 좌우, 이닝/초말)로 묶어 "이런 상황에서 투수들이 보통
어떤 구속/무브먼트/구종 비율로 던지는가"를 집계해 train/test에 붙인다.

trackman_history는 2019~2024년치뿐이라(2025 없음) 평가 시점(2025)보다 항상
과거 데이터이므로 test에는 안전하다. train 쪽은 선수 단위가 아닌 전체 기간
집계 통계라 개별 행의 결과 정보를 담고 있지 않다.
"""
import os

import joblib
import pandas as pd

DATA_DIR = "/home/esem5377/lg_aimers7_ws/lg_aimers7_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers7_ws/lg_aimers7_ws/work/model"

# trackman_history 값 -> train.csv 인코딩으로 맞추는 매핑
# (빈도 비율 대조로 확인: pitcher_hand 2=Right/1=Left, batter_hand 2=Right/1=Left)
HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}

METRIC_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]

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
        table = build_group_table(th, keys, {"count_state": "tk_cnt", "hand_matchup": "tk_hand", "inning_state": "tk_inn"}[name])
        print(f"\n[{name}] keys={keys} rows={len(table)}")
        print(table.head())
        context[name] = {"keys": keys, "table": table}

    out_path = os.path.join(MODEL_DIR, "trackman_context.pkl")
    joblib.dump(context, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
