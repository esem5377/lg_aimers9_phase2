"""build_trackman_context.py 확장판 — 지금까지 안 쓴 트랙맨 물리량과 분산 통계 추가.

기존 trackman_context.pkl(pipeline/build_trackman_context.py)은 상황(count_state/
hand_matchup/inning_state) 단위로 rel_speed/spin_rate/induced_vert_break/horz_break
'평균'과 구종 비율만 집계했다. 이 스크립트는 같은 상황 키·같은 조인 방식(개인
단위 join은 여전히 불가능하므로 인구 평균 통계라는 성격은 동일)을 유지한 채:
  1. 지금까지 미사용이던 extension/rel_height/rel_side/zone_speed 추가
  2. 8개 물리량 전부 mean뿐 아니라 std(퍼짐 정도)도 추가
로 확장한 trackman_context_v2.pkl을 만든다. 프로덕션 trackman_context.pkl은
건드리지 않고 별도 파일로 저장 — tune_trackman_v2.py에서 baseline과 비교 검증용.
"""
import os

import joblib
import pandas as pd

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"

HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}

METRIC_COLS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]

GROUPINGS = {
    "count_state": ["balls_before", "strikes_before", "outs_before"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "inning_state": ["inning", "top_bottom"],
}
PREFIX_MAP = {"count_state": "tk2_cnt", "hand_matchup": "tk2_hand", "inning_state": "tk2_inn"}


def load_trackman():
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    df["top_bottom"] = df["top_bottom"].map(TOP_BOTTOM_MAP)
    return df


def build_group_table(df, keys, prefix):
    g = df.groupby(keys, dropna=False)
    stats = g[METRIC_COLS].agg(["mean", "std"])
    stats.columns = [f"{prefix}_{c}_{stat}" for c, stat in stats.columns]

    pitch_rate = pd.crosstab([df[k] for k in keys], df["pitch_type_group"], normalize="index")
    for pg in PITCH_GROUPS:
        stats[f"{prefix}_{pg}_rate"] = pitch_rate.get(pg, 0.0)

    stats[f"{prefix}_n"] = g.size()
    return stats.reset_index()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Load trackman_history...")
    th = load_trackman()
    print(f" shape={th.shape}")

    context = {}
    for name, keys in GROUPINGS.items():
        table = build_group_table(th, keys, PREFIX_MAP[name])
        print(f"\n[{name}] keys={keys} rows={len(table)} cols={len(table.columns)}")
        print(table.head())
        context[name] = {"keys": keys, "table": table}

    out_path = os.path.join(MODEL_DIR, "trackman_context_v2.pkl")
    joblib.dump(context, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
