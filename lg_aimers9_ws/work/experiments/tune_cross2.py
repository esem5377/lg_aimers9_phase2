"""실험: 저카디널리티 2키 trackman 교차 피처 몇 종을 개별/전체 추가해 비교.

tune_v2.py에서 시도했던 count_state x hand_matchup 5키 풀교차(181개 조합)는
표본이 너무 쪼개져 실패했다(AUC 0.54808 < baseline 0.54887). 이번엔 조합
수를 훨씬 작게 줄인 2~3키 교차 후보 몇 개를 하나씩 추가해보고, 도움되는
것만 모아 전부 합친 버전까지 비교한다. 각 후보는 기존 3종 컨텍스트
(count_state/hand_matchup/inning_state)에 추가로 붙이는 형태이며, CatBoost
현재 채택 하이퍼파라미터(BEST_PARAMS)로 고정해 비교한다.

후보 (trackman_history.csv에 실제 존재하는 컬럼만 사용 — base_state/주자 정보는
trackman_history에 없어서 제외):
  - hand_outs:  pitcher_hand x batter_hand x outs_before   (2*2*3=12 조합)
  - count_light: balls_before x strikes_before (outs 제외)  (4*3=12 조합)
  - hand_inning_half: pitcher_hand x batter_hand x top_bottom (2*2*2=8 조합)
"""
import json
import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}
METRIC_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]

BASELINE_AUC = 0.55056  # 현재 채택 버전 (교차 피처 없음)

CB_BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

CANDIDATES = {
    "hand_outs": ["pitcher_hand", "batter_hand", "outs_before"],
    "count_light": ["balls_before", "strikes_before"],
    "hand_inning_half": ["pitcher_hand", "batter_hand", "top_bottom"],
}


def build_group_table(df, keys, prefix):
    g = df.groupby(keys, dropna=False)
    out = g[METRIC_COLS].mean()
    out.columns = [f"{prefix}_{c}_mean" for c in out.columns]
    pitch_rate = pd.crosstab([df[k] for k in keys], df["pitch_type_group"], normalize="index")
    for pg in PITCH_GROUPS:
        out[f"{prefix}_{pg}_rate"] = pitch_rate.get(pg, 0.0)
    out[f"{prefix}_n"] = g.size()
    return out.reset_index()


def load_trackman():
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    df["top_bottom"] = df["top_bottom"].map(TOP_BOTTOM_MAP)
    return df


def build_candidate_specs(th):
    specs = {}
    for name, keys in CANDIDATES.items():
        table = build_group_table(th, keys, f"tk_{name}")
        print(f"[{name}] keys={keys} rows={len(table)}")
        specs[name] = {"keys": keys, "table": table}
    return specs


def load_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def fit_eval(X_tr, y_tr, X_va, y_va, label):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **CB_BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    print(f"[{label}] auc={auc:.5f}  best_iter={model.get_best_iteration()}")
    return auc


def main():
    print("Load trackman_history & build candidate cross tables...")
    th = load_trackman()
    specs = build_candidate_specs(th)

    print("\nLoad base train data...")
    df_base = load_base_df()
    train_mask = df_base["season"] <= 2023
    valid_mask = df_base["season"] == 2024
    y_all = df_base[TARGET_COL]

    results = {}

    print("\n=== baseline (no extra cross) ===")
    X_base = build_features(df_base)
    results["baseline"] = fit_eval(
        X_base[train_mask], y_all[train_mask], X_base[valid_mask], y_all[valid_mask], "baseline")

    helpful = []
    for name, spec in specs.items():
        print(f"\n=== + {name} ===")
        df_c = df_base.merge(spec["table"], on=spec["keys"], how="left")
        X_c = build_features(df_c)
        auc = fit_eval(
            X_c[train_mask], y_all[train_mask], X_c[valid_mask], y_all[valid_mask], name)
        results[name] = auc
        if auc > results["baseline"]:
            helpful.append(name)

    if helpful:
        print(f"\n=== + all helpful combined ({helpful}) ===")
        df_all = df_base
        for name in helpful:
            df_all = df_all.merge(specs[name]["table"], on=specs[name]["keys"], how="left")
        X_all = build_features(df_all)
        results["all_helpful_combined"] = fit_eval(
            X_all[train_mask], y_all[train_mask], X_all[valid_mask], y_all[valid_mask],
            "all_helpful_combined")
    else:
        print("\n어떤 후보도 baseline보다 안 나아서 combined 버전은 생략.")

    print("\n=== summary ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        marker = " <- BEST" if v == max(results.values()) else ""
        print(f"{k:25s} auc={v:.5f}  delta={v - results['baseline']:+.5f}{marker}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_cross2_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
