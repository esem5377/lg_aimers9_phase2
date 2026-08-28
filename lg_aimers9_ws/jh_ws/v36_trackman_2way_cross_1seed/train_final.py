"""트랙맨 2-way 교차 피처(count_state x hand_matchup, count_state x inning_state,
hand_matchup x inning_state, 24개 신규 컬럼)를 987점 베이스(v22: control_risk_score
+ 원재료 3개 제거, 70피처) 위에 추가한 1시드 실제 제출 준비.

주의(중요, 사용자에게 명시적으로 고지 완료 후 진행, 2026-08-28):
  - 이 피처는 fold0/fold2 walk-forward 스크리닝(session_2026-08-28_trackman_2way_cross/
    trackman_2way_walkforward.py)에서 fold2 +4.56 / fold0 -2.16으로 부호가 갈려
    이 프로젝트 기준("두 폴드 다 양성이어야 통과")을 충족하지 못해 기각 판정을 받았다.
  - v30(pruning)/v31(beta calib)처럼 로컬 신호가 약해도(또는 엇갈려도) 실측 확인
    자체에 가치를 두는 선례를 따라, 사용자가 명시적으로 실제 제출 진행을 선택함.

나머지(control_risk_score 레시피/CAT_COLS/RAW_ID_COLS/BEST_PARAMS/95:5 calib
carve-out/Platt calibration)는 v22/v30과 동일. pruning은 적용하지 않음(v30의
pruning 자체가 별도로 기각된 실험이라 이 실험과는 독립).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
BASE_TRACKMAN_CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
V36_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(V36_DIR, "model")
OUT_DIR = os.path.join(V36_DIR, "output")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)
ITERATIONS = 2000
SEED = 42
DATA_SPLIT_SEED = 42

# ---- trackman_history 2-way 교차 (trackman_2way_walkforward.py와 동일 로직) ----
HAND_MAP = {"Right": 2, "Left": 1}
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B"}
METRIC_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]

CROSS_GROUPINGS = {
    "cnt_hand": (["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"], "tk_cnthand"),
    "cnt_inn": (["balls_before", "strikes_before", "outs_before", "inning", "top_bottom"], "tk_cntinn"),
    "hand_inn": (["pitcher_hand", "batter_hand", "inning", "top_bottom"], "tk_handinn"),
}


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt_scaling(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt_scaling(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def load_trackman_for_cross():
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    df["top_bottom"] = df["top_bottom"].map(TOP_BOTTOM_MAP)
    return df


def build_group_table(df, keys, prefix):
    g = df.groupby(keys, dropna=False)
    out = g[METRIC_COLS].mean()
    out.columns = [f"{prefix}_{c}_mean" for c in out.columns]
    pitch_rate = pd.crosstab([df[k] for k in keys], df["pitch_type_group"], normalize="index")
    for pg in PITCH_GROUPS:
        out[f"{prefix}_{pg}_rate"] = pitch_rate.get(pg, 0.0)
    out[f"{prefix}_n"] = g.size()
    return out.reset_index()


def build_cross_context():
    print("Load trackman_history for 2-way cross...", flush=True)
    th = load_trackman_for_cross()
    print(f" shape={th.shape}", flush=True)
    context = {}
    for name, (keys, prefix) in CROSS_GROUPINGS.items():
        table = build_group_table(th, keys, prefix)
        n_new = len([c for c in table.columns if c not in keys])
        print(f" [{name}] keys={keys} rows={len(table)} new_cols={n_new}", flush=True)
        context[name] = {"keys": keys, "table": table}
    return context


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    base_context = joblib.load(BASE_TRACKMAN_CONTEXT_PATH)
    for spec in base_context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    cross_context = build_cross_context()
    for spec in cross_context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    combined_context = {**base_context, **cross_context}
    return df, combined_context


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    df = df.drop(columns=INGREDIENT_COLS)
    return df


def build_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Load train data (전체) + base trackman context + 2-way 교차 context + "
          "control_risk_score(*2) 추가, 원재료 3개 제거...", flush=True)
    df, combined_context = load_data()
    df = add_risk_score_drop_ingredients(df)
    print(f" shape={df.shape}", flush=True)

    context_path = os.path.join(MODEL_DIR, "trackman_context.pkl")
    joblib.dump(combined_context, context_path)
    print(f" saved combined trackman context (base 3축 + cross 3축): {context_path}", flush=True)

    id_mappings = build_id_mappings(df)
    print(f" id_mappings: pitcher_id n={len(id_mappings['pitcher_id'])}  "
          f"batter_id n={len(id_mappings['batter_id'])}", flush=True)

    X_all = build_features(df, id_mappings)
    y_all = df[TARGET_COL]
    print(f" n_features={X_all.shape[1]} (987베이스 70 + 교차 24)", flush=True)

    print("\nCalibration carve-out(5%) 분리...", flush=True)
    X_train_final, X_calib, y_train_final, y_calib = train_test_split(
        X_all, y_all, test_size=0.05, stratify=y_all, random_state=DATA_SPLIT_SEED,
    )
    print(f" train={X_train_final.shape}  calibration carve-out={X_calib.shape}", flush=True)

    cat_idx = [X_train_final.columns.get_loc(c) for c in CAT_COLS]

    print(f"\n=== seed={SEED} 학습 (iterations={ITERATIONS}, n_features={X_train_final.shape[1]}) ===", flush=True)
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED,
        cat_features=cat_idx, verbose=200,
        **BEST_PARAMS,
    )
    model.fit(X_train_final, y_train_final)
    elapsed = time.time() - t0
    print(f" 학습 완료 ({elapsed:.1f}s)", flush=True)

    calib_raw = model.predict_proba(X_calib)[:, 1]
    a_final, b_final = fit_platt_scaling(calib_raw, y_calib)
    calib_pred = apply_platt_scaling(calib_raw, a_final, b_final)

    metrics = {
        "seed": SEED,
        "elapsed_sec": elapsed,
        "n_features": int(X_all.shape[1]),
        "carveout_bss_raw": bss_score(calib_raw, y_calib),
        "carveout_bss_calibrated": bss_score(calib_pred, y_calib),
    }
    print(f" carve-out BSS: raw={metrics['carveout_bss_raw']:.2f}  "
          f"calibrated={metrics['carveout_bss_calibrated']:.2f}", flush=True)
    with open(os.path.join(OUT_DIR, "metrics_v36.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model_path = os.path.join(MODEL_DIR, f"catboost_seed{SEED}.cbm")
    model.save_model(model_path)
    print(f" saved: {model_path}", flush=True)

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "columns": list(X_train_final.columns),
            "cat_cols": CAT_COLS,
            "raw_id_cols": RAW_ID_COLS,
            "id_mappings": id_mappings,
            "seeds": [SEED],
            "calibration": {"method": "platt_sigmoid", "a": a_final, "b": b_final},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved 1 model + feature_meta.json + trackman_context.pkl to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
