"""1160점 목표 세션(2026-08-28): 기존 검증된 트랙맨 상황 축(count_state/
hand_matchup/inning_state, es_ws에서 +40점 기여 확인됨) 3개를 2개씩 교차한
새 축 3종(cnt_hand/cnt_inn/hand_inn)이 fold0/fold2 walk-forward에서
CatBoost baseline 대비 이득이 있는지 저비용(encoder 없이 CatBoost만) 스크리닝.

v17에서 4번째 "독립" 축(month) 추가는 실패(-18.9)했지만, 기존 3축끼리의
2-way 교차는 아직 미시도. 방금 v32(retrieval_score를 CatBoost 피처로)가
랜덤 carve-out 1위 신호였는데 fold2 walk-forward에서 -89.76으로 크게
반증된 사례가 있어, 이번에도 반드시 fold0/fold2부터 먼저 확인한다.

baseline = 기존 3축 context + control_risk_score(원재료 제거, 987/992 레시피).
treatment = baseline + cnt_hand/cnt_inn/hand_inn 3종 교차 피처.
동일 feature_pool/calib split(v32_walkforward.py와 동일 로직, seed=42)이라
baseline 수치는 v32의 [baseline] eval_bss와 fold2에서 거의 재현되어야 함
(sanity check).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
TRACKMAN_CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "trackman_2way_results.json")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
CATBOOST_CAT_COLS = CAT_COLS + TEAM_COLS
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BEST_PARAMS = dict(learning_rate=0.01, depth=8, l2_leaf_reg=20.0, bagging_temperature=1.0, random_strength=1.0, border_count=32)
ITERATIONS = 2000

# ---- trackman_history 2-way 교차 (es_ws build_trackman_context.py와 동일한 매핑/집계 방식) ----
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


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, a, b):
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
    new_cols = []
    for name, (keys, prefix) in CROSS_GROUPINGS.items():
        table = build_group_table(th, keys, prefix)
        n_new = len([c for c in table.columns if c not in keys])
        print(f" [{name}] keys={keys} rows={len(table)} new_cols={n_new}", flush=True)
        context[name] = {"keys": keys, "table": table}
        new_cols += [c for c in table.columns if c not in keys]
    return context, new_cols


def load_data_with_cross():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    base_context = joblib.load(TRACKMAN_CONTEXT_PATH)
    for spec in base_context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    cross_context, new_cols = build_cross_context()
    for spec in cross_context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df, new_cols


def add_risk_score_drop_ingredients(df):
    df = df.copy()
    df["control_risk_score"] = df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    df["control_risk_score_weighted"] = 0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    return df.drop(columns=INGREDIENT_COLS)


def build_catboost_id_mappings(df):
    return {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in RAW_ID_COLS}


def build_catboost_features(df, id_mappings):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CATBOOST_CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def fit_and_eval(X_fp, y_fp, X_calib, y_calib, X_eval, y_eval, cat_cols, tag):
    cat_idx = [X_fp.columns.get_loc(c) for c in cat_cols]
    t0 = time.time()
    cb = CatBoostClassifier(iterations=ITERATIONS, loss_function="Logloss", random_seed=SEED, cat_features=cat_idx, verbose=False, **BEST_PARAMS)
    cb.fit(X_fp, y_fp)
    elapsed = time.time() - t0
    print(f" [{tag}] CatBoost 학습 완료 ({elapsed:.1f}s)", flush=True)
    calib_raw = cb.predict_proba(X_calib)[:, 1]
    eval_raw = cb.predict_proba(X_eval)[:, 1]
    a, b = fit_platt(calib_raw, y_calib)
    eval_calib = apply_platt(eval_raw, a, b)
    bss = bss_score(eval_calib, y_eval)
    auc = roc_auc_score(y_eval, eval_raw)
    print(f" [{tag}] eval_auc={auc:.4f} eval_bss={bss:.2f}", flush=True)
    return {"eval_auc": auc, "eval_bss": bss, "train_seconds": elapsed}, cb


def run_fold(fold_name, train_full_df, eval_df, new_cross_cols, all_results):
    print(f"\n=== {fold_name} ===", flush=True)
    train_full_df = add_risk_score_drop_ingredients(train_full_df).reset_index(drop=True)
    eval_df = add_risk_score_drop_ingredients(eval_df).reset_index(drop=True)

    y_all = train_full_df[TARGET_COL]
    rest_df, calib_df = train_test_split(train_full_df, test_size=0.05, stratify=y_all, random_state=SEED)
    # v32_walkforward.py와 동일 비율(rest 95% -> encoder 45% / feature_pool 50%).
    # 여기선 encoder가 없으므로 feature_pool(50%)만 사용, 나머지 45%는 버림
    # (v32 baseline과 동일 split이 되도록 순서/비율을 그대로 재현 -- sanity check용).
    _encoder_train_unused, feature_pool_df = train_test_split(
        rest_df, test_size=0.5263, stratify=rest_df[TARGET_COL], random_state=SEED,
    )
    feature_pool_df = feature_pool_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    print(f" feature_pool={len(feature_pool_df)} calib={len(calib_df)} eval={len(eval_df)}", flush=True)

    cb_id_mappings = build_catboost_id_mappings(feature_pool_df)
    X_fp_full = build_catboost_features(feature_pool_df, cb_id_mappings)
    X_calib_full = build_catboost_features(calib_df, cb_id_mappings)
    X_eval_full = build_catboost_features(eval_df, cb_id_mappings)
    y_fp = feature_pool_df[TARGET_COL]
    y_calib = calib_df[TARGET_COL]
    y_eval = eval_df[TARGET_COL]

    # baseline: 교차 피처 컬럼 제거
    X_fp_base = X_fp_full.drop(columns=new_cross_cols)
    X_calib_base = X_calib_full.drop(columns=new_cross_cols)
    X_eval_base = X_eval_full.drop(columns=new_cross_cols)

    base_metrics, _ = fit_and_eval(X_fp_base, y_fp, X_calib_base, y_calib, X_eval_base, y_eval, CATBOOST_CAT_COLS, "baseline")
    treat_metrics, cb_treat = fit_and_eval(X_fp_full, y_fp, X_calib_full, y_calib, X_eval_full, y_eval, CATBOOST_CAT_COLS, "treatment(2way_cross)")

    delta = treat_metrics["eval_bss"] - base_metrics["eval_bss"]
    importances = pd.Series(cb_treat.get_feature_importance(), index=X_fp_full.columns).sort_values(ascending=False)
    cross_importance_ranks = {c: int(list(importances.index).index(c)) + 1 for c in new_cross_cols if c in importances.index}
    top_cross = importances[[c for c in new_cross_cols if c in importances.index]].sort_values(ascending=False).head(5)

    print(f" [delta] treatment - baseline = {delta:+.2f}", flush=True)
    print(f" [top cross features by importance] {top_cross.to_dict()}", flush=True)

    all_results[fold_name] = {
        "n_feature_pool": len(feature_pool_df), "n_calib": len(calib_df), "n_eval": len(eval_df),
        "baseline": base_metrics,
        "treatment_2way_cross": treat_metrics,
        "delta": delta,
        "n_features_total": int(len(importances)),
        "cross_feature_importance_ranks": cross_importance_ranks,
        "top5_cross_features": top_cross.to_dict(),
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)


def main():
    print("Load train data + base trackman context + 2-way cross context...", flush=True)
    df, new_cross_cols = load_data_with_cross()
    print(f" shape={df.shape}  new_cross_cols({len(new_cross_cols)})={new_cross_cols}", flush=True)

    all_results = {}
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, encoding="utf-8") as f:
            all_results = json.load(f)

    fold_specs = [
        ("fold2_2024", df[df["season"] <= 2023], df[df["season"] == 2024]),
        ("fold0_2022", df[df["season"] <= 2021], df[df["season"] == 2022]),
    ]
    for fold_name, train_df, eval_df in fold_specs:
        if fold_name in all_results:
            print(f"\n=== {fold_name}: 이미 완료됨, 스킵 ===", flush=True)
            continue
        run_fold(fold_name, train_df, eval_df, new_cross_cols, all_results)

    print("\n=== SUMMARY ===", flush=True)
    for fold_name, r in all_results.items():
        print(f"  {fold_name}: baseline={r['baseline']['eval_bss']:.2f} treatment={r['treatment_2way_cross']['eval_bss']:.2f} delta={r['delta']:+.2f}", flush=True)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
