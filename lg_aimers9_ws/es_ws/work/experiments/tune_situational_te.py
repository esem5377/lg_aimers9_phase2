"""'상황 자체의 난이도' target encoding이 새 신호가 되는지 검증.

지금까지 실패한 방향들(트랙맨 조인, DeepFM 매치업 임베딩, raw pitcher_id)의
공통점: 전부 "이 투수/타자가 누구냐"라는 개체 식별자에 의존하는 피처였다.
실제 검증/평가가 진짜 group k-fold(투수가 train/valid에 안 겹침)라서, 이런
개체 단위 피처는 unseen pitcher에 원천적으로 못 붙는(cold-start) 구조적
한계가 있었다(work worklog 2026-08-16 섹션11, 섹션13).

이번엔 개체가 아니라 "상황 자체가 얼마나 어려운가"를 train.csv 자체의
control_success로 직접 추정한다. pitcher_hand x batter_hand x count,
outs x base_state, inning 흐름, leverage(li) 네 그룹으로 나눠 각각
smoothed target encoding을 만든다. 개체 식별자가 전혀 안 들어가므로
unseen pitcher/batter에도 그대로 일반화된다 — 이 점이 지금까지 시도와
구조적으로 다른 부분.

리키지 방지: 각 CV fold의 train 쪽 데이터로만 TE 테이블을 새로 적합하고
valid 쪽엔 그 테이블을 merge만 한다(섹션13 DeepFM 임베딩 리키지 재발 방지).
검증은 jh_ws에서 신뢰성이 확인된 진짜 StratifiedGroupKFold(5, group=pitcher_id).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
DROP_COLS = ["pitcher_id", "batter_id"]

TUNED_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

N_FOLDS = 5
SMOOTH_M = 50  # 베이지안 스무딩 강도(가상 표본 수)
LI_EDGES = [-0.01, 0.4, 0.8, 1.32, 11.0]  # 전체 데이터 li quartile 근사(고정값, fold별 재계산 안 함 -> 리키지 없음)
INNING_EDGES = [0, 3, 6, 30]
INNING_LABELS = ["early", "mid", "late"]

TE_GROUPS = {
    "te_hand_count": ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"],
    "te_outs_base": ["outs_before", "base_state"],
    "te_inning": ["inning_bucket", "top_bottom"],
    "te_leverage": ["li_bucket"],
}


def add_buckets(df):
    df = df.copy()
    df["li_bucket"] = pd.cut(df["li"], LI_EDGES, labels=False)
    df["inning_bucket"] = pd.cut(df["inning"], INNING_EDGES, labels=INNING_LABELS)
    return df


def fit_te_table(df_tr, keys, global_mean):
    g = df_tr.groupby(keys, dropna=False)[TARGET_COL].agg(["sum", "count"])
    g["te"] = (g["sum"] + SMOOTH_M * global_mean) / (g["count"] + SMOOTH_M)
    return g["te"].reset_index()


def apply_te(df, table, keys, colname, global_mean):
    out = df.merge(table.rename(columns={"te": colname}), on=keys, how="left")
    out[colname] = out[colname].fillna(global_mean)
    return out


def merge_trackman_context(df):
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    out = df.copy()
    for spec in context.values():
        out = out.merge(spec["table"], on=spec["keys"], how="left")
    return out


def build_features(df, extra_cols):
    keep_drop = [c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns]
    X = df.drop(columns=keep_drop)
    # 버킷 원본 컬럼(li_bucket/inning_bucket)은 TE로 이미 흡수했으니 raw는 제외
    for c in ["li_bucket", "inning_bucket"]:
        if c in X.columns and c not in extra_cols:
            X = X.drop(columns=[c])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def fit_eval(X_tr, y_tr, X_va, y_va):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False,
        **TUNED_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    pred = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, pred), model.get_best_iteration(), model


def main():
    t_start = time.time()
    print("Load train.csv + trackman context...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df = merge_trackman_context(df)
    df = add_buckets(df)
    print(f" shape={df.shape}", flush=True)

    groups = df["pitcher_id"].values
    y = df[TARGET_COL].values
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = {"baseline": [], "with_te": []}
    importances_last_fold = None

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(df, y, groups)):
        df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
        n_overlap = len(set(df_tr["pitcher_id"]) & set(df_va["pitcher_id"]))
        print(f"\n[fold{fold_i}] n_train={len(df_tr)} n_valid={len(df_va)} "
              f"pitcher_overlap={n_overlap} (0이어야 진짜 group k-fold)", flush=True)

        global_mean = df_tr[TARGET_COL].mean()

        # ---- baseline (TE 없음) ----
        X_tr_b = build_features(df_tr, extra_cols=[])
        X_va_b = build_features(df_va, extra_cols=[])
        t0 = time.time()
        auc_b, iter_b, _ = fit_eval(X_tr_b, df_tr[TARGET_COL], X_va_b, df_va[TARGET_COL])
        print(f"  baseline        auc={auc_b:.5f} best_iter={iter_b} ({time.time()-t0:.1f}s)", flush=True)
        fold_results["baseline"].append(auc_b)

        # ---- with situational TE (fold train으로만 적합) ----
        df_tr_te, df_va_te = df_tr.copy(), df_va.copy()
        for name, keys in TE_GROUPS.items():
            table = fit_te_table(df_tr, keys, global_mean)
            df_tr_te = apply_te(df_tr_te, table, keys, name, global_mean)
            df_va_te = apply_te(df_va_te, table, keys, name, global_mean)

        extra_cols = list(TE_GROUPS.keys())
        X_tr_t = build_features(df_tr_te, extra_cols=extra_cols)
        X_va_t = build_features(df_va_te, extra_cols=extra_cols)
        t0 = time.time()
        auc_t, iter_t, model_t = fit_eval(X_tr_t, df_tr_te[TARGET_COL], X_va_t, df_va_te[TARGET_COL])
        print(f"  with_situational_te auc={auc_t:.5f} best_iter={iter_t} ({time.time()-t0:.1f}s) "
              f"delta={auc_t-auc_b:+.5f}", flush=True)
        fold_results["with_te"].append(auc_t)

        if fold_i == N_FOLDS - 1:
            fi = sorted(zip(X_tr_t.columns, model_t.get_feature_importance()),
                        key=lambda x: -x[1])
            importances_last_fold = fi[:15]

    print(f"\n{'='*60}\nSUMMARY (elapsed {time.time()-t_start:.0f}s)\n{'='*60}")
    mean_b = float(np.mean(fold_results["baseline"]))
    mean_t = float(np.mean(fold_results["with_te"]))
    deltas = [t - b for t, b in zip(fold_results["with_te"], fold_results["baseline"])]
    consistent = all(d > 0 for d in deltas)
    print(f"baseline   mean_auc={mean_b:.5f}  per-fold={['%.5f'%a for a in fold_results['baseline']]}")
    print(f"with_te    mean_auc={mean_t:.5f}  per-fold={['%.5f'%a for a in fold_results['with_te']]}")
    print(f"delta={mean_t-mean_b:+.5f}  per-fold-delta={['%+.5f'%d for d in deltas]}  "
          f"모든폴드에서_개선={consistent}")
    print("\ntop15 feature importance (last fold, with_te 모델):")
    for name, imp in importances_last_fold:
        print(f"  {name:30s} {imp:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    result = {
        "fold_results": fold_results,
        "mean_baseline": mean_b,
        "mean_with_te": mean_t,
        "delta": mean_t - mean_b,
        "fold_deltas": deltas,
        "consistent_all_folds": consistent,
        "top_importances_last_fold": importances_last_fold,
    }
    with open(os.path.join(OUT_DIR, "tune_situational_te_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_situational_te_results.json")


if __name__ == "__main__":
    main()
