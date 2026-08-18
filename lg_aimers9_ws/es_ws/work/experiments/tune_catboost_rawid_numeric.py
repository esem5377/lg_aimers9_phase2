"""실험: pitcher_id/batter_id를 jh_ws v9(924점) 방식 그대로 재현해서 검증.

tune_catboost_rawid.py(기각됨, auc -0.00344)는 pitcher_id/batter_id를 CatBoost
cat_features(ordered target statistics)로 넣었는데, 이건 jh_ws가 실제로 쓴 방식이
아니다. jh_ws/v9_timesplit/train.py를 다시 보면 label-encoding으로 정수 변환만
하고 cat_features에는 안 넣는다 — 즉 순수 수치형 ID로 취급한다(target statistics
없음, ordering/memoization 신호만 남음). 이 방식은 es_ws에서 아직 테스트 안 됨.

train_catboost.py와 동일 피처(trackman context 포함)·동일 시간 분할
(season<=2023 학습/2024 검증)·동일 하이퍼파라미터(BEST_PARAMS)를 쓰고,
pitcher_id/batter_id만 label-encoded 수치형으로 추가한 버전으로 검증 AUC 비교.
"""
import json
import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"

TARGET_COL = "control_success"
ID_COL = "row_id"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]  # pitcher_id/batter_id는 여기 없음 -- 수치형(label-encoded)으로만 추가

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

BASELINE_AUC = 0.55056  # train_catboost.py 현재 채택 버전 (raw id 제외)
CATSTAT_RAWID_AUC = 0.54712  # tune_catboost_rawid.py (CatBoost cat_features로 넣은 버전, 기각)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")

    X = df.drop(columns=[ID_COL, TARGET_COL])

    # jh_ws v9 방식: pitcher_id/batter_id를 label-encoding으로 정수 변환 (수치형 유지, cat_features 아님)
    for c in ["pitcher_id", "batter_id"]:
        uniq = sorted(X[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        X[c] = X[c].astype(str).map(mapping).astype(int)

    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    y = df[TARGET_COL]

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    return X[train_mask], y[train_mask], X[valid_mask], y[valid_mask]


def main():
    print("Load data (pitcher_id/batter_id를 jh_ws 방식 수치형 label-encoding으로 포함)...")
    X_tr, y_tr, X_va, y_va = load_data()
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]  # pitcher_id/batter_id는 미포함
    print(f" train={X_tr.shape} valid={X_va.shape}")
    print(f" cat_features={CAT_COLS}")
    print(f" numeric raw id 컬럼: pitcher_id (n_unique={X_tr['pitcher_id'].nunique()}), "
          f"batter_id (n_unique={X_tr['batter_id'].nunique()})")

    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC",
        random_seed=42, cat_features=cat_idx,
        early_stopping_rounds=100, verbose=200,
        **BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    best_iter = model.get_best_iteration()

    print(f"\nauc(numeric raw id 포함, jh_ws 방식) = {auc:.5f}  best_iteration={best_iter}")
    print(f"auc(raw id 완전 제외, 현재 채택 버전)     = {BASELINE_AUC:.5f}")
    print(f"auc(raw id를 CatBoost 범주형으로, 기각됨) = {CATSTAT_RAWID_AUC:.5f}")
    print(f"delta vs baseline(제외) = {auc - BASELINE_AUC:+.5f}")
    print(f"delta vs catstat(범주형) = {auc - CATSTAT_RAWID_AUC:+.5f}")

    imp = model.get_feature_importance(prettified=True)
    print("\n=== Top 15 feature importance ===")
    print(imp.head(15).to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_catboost_rawid_numeric_result.json"), "w") as f:
        json.dump({
            "auc_with_numeric_raw_id": auc,
            "auc_baseline_no_raw_id": BASELINE_AUC,
            "auc_catstat_raw_id_rejected": CATSTAT_RAWID_AUC,
            "delta_vs_baseline": auc - BASELINE_AUC,
            "delta_vs_catstat": auc - CATSTAT_RAWID_AUC,
            "best_iteration": int(best_iter),
        }, f, indent=2)


if __name__ == "__main__":
    main()
