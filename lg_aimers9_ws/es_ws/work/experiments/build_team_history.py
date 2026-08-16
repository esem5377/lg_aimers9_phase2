"""train.csv 자체로 '팀 단위 as-of 이력' 피처를 만든다.

train.csv에는 정확한 경기 날짜/순번이 없고 season/game_month/game_dayofweek
까지만 있어서, 개별 투구 단위 as-of는 재현할 수 없다(그래서 asof_* 컬럼은
주최측이 원본 정밀 시각으로 미리 계산해 제공한 것). 하지만 시즌 단위라면
안전하게 직접 만들 수 있다: 각 팀의 "이전 시즌까지 누적 성공률"은 해당 행이
속한 시즌 내부의 다른 행을 전혀 보지 않으므로(build_trackman_context.py의
과거-시즌만 사용하는 방식과 동일한 논리) 리크가 없다.

pitcher_team_id/batter_team_id 두 컬럼 모두 같은 13개 팀 코드 체계를
공유한다(값 교집합 동일 확인됨). 각각에 대해:
  - team_cum_success_rate : 그 팀이 이전 시즌들(현재 시즌 미포함)에 걸쳐
    누적된 control_success 평균
  - team_cum_n            : 위 누적에 쓰인 표본 수 (콜드스타트 신뢰도용)
  - team_prev1season_success_rate : 바로 직전 시즌 하나만의 평균

test.csv(2025시즌)에는 해당 시즌 데이터가 없으므로, 2019~2024 전체를
"이전 시즌"으로 취급하는 season=2025 행을 테이블에 추가해 넣는다.
"""
import os

import joblib
import pandas as pd

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"

TEAM_SPECS = {
    "pitcher_team": {"team_col": "pitcher_team_id", "prefix": "pt_team"},
    "batter_team": {"team_col": "batter_team_id", "prefix": "bt_team"},
}


def build_asof_season_table(df, team_col, prefix, eval_season):
    """eval_season: 확장 행을 붙일 평가 시즌(예: 2025). 팀마다 마지막 관측 시즌이

    달라(팀 코드 통폐합/변경 등으로 일부 팀은 2019~2020만 존재) "마지막 시즌+1"로
    확장하면 팀별로 제각각인 시즌에 확장 행이 생겨 실제 평가 시즌과 어긋날 수
    있다. 그래서 모든 팀에 대해 동일하게 eval_season 하나로 고정해서 확장한다.
    """
    season_stats = (
        df.groupby([team_col, "season"])["control_success"]
        .agg(n="count", rate="mean")
        .reset_index()
        .sort_values([team_col, "season"])
    )
    season_stats["success_n"] = season_stats["n"] * season_stats["rate"]

    rows = []
    for team, g in season_stats.groupby(team_col):
        g = g.sort_values("season").reset_index(drop=True)
        cum_n_prior = g["n"].cumsum().shift(fill_value=0)
        cum_success_prior = g["success_n"].cumsum().shift(fill_value=0)
        cum_rate_prior = cum_success_prior / cum_n_prior
        prev1_rate = g["rate"].shift(1)

        out = pd.DataFrame({
            team_col: team,
            "season": g["season"],
            f"{prefix}_cum_success_rate": cum_rate_prior,
            f"{prefix}_cum_n": cum_n_prior,
            f"{prefix}_prev1season_success_rate": prev1_rate,
        })

        # 평가 시즌(eval_season)을 위한 확장 행: 그 팀이 관측된 전체 시즌을 누적으로 사용
        extra = pd.DataFrame({
            team_col: [team],
            "season": [eval_season],
            f"{prefix}_cum_success_rate": [g["success_n"].sum() / g["n"].sum()],
            f"{prefix}_cum_n": [g["n"].sum()],
            f"{prefix}_prev1season_success_rate": [g["rate"].iloc[-1]],
        })
        rows.append(pd.concat([out, extra], ignore_index=True))

    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Load train data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                      usecols=["pitcher_team_id", "batter_team_id", "season", "control_success"])
    print(f" shape={df.shape}")

    eval_season = int(df["season"].max()) + 1

    team_history = {}
    for name, spec in TEAM_SPECS.items():
        table = build_asof_season_table(df, spec["team_col"], spec["prefix"], eval_season)
        print(f"\n[{name}] rows={len(table)}")
        print(table.head(8))
        team_history[name] = {"keys": [spec["team_col"], "season"], "table": table}

    out_path = os.path.join(MODEL_DIR, "team_history.pkl")
    joblib.dump(team_history, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
