"""Adversarial validation으로 "불안정한(연도별로 분포가 다른) 피처" 전수조사.

지금까지(8/17 game_type, 8/23 asof_* 5개)는 "피처-타겟 상관계수가 연도별로
얼마나 흔들리는지"만 몇 개 후보에 대해 확인했음. 이번엔 다른 각도: "피처
값 자체의 분포가 연도별로 다른가"를 전수조사 -- 실제로 시도할 실험이 아니라
진단 도구.

방법: 2019~2023(과거, "train"으로 라벨링) vs 2024(가장 최근, "recent"로
라벨링, 이 프로젝트의 실제 fold2 홀드아웃과 동일 구간이라 2025 일반화
갭과 가장 비슷한 시간 간격) 두 그룹을 구분하는 CatBoost 분류기를 학습.
이 분류기가 어떤 피처로 "연도"를 잘 맞히는지(feature importance)가 곧
"이 피처는 시간에 따라 분포가 달라진다"는 신호. control_success(정답
라벨)는 당연히 피처에서 제외(타겟 누출 방지, 애초에 다른 걸 예측하는
분류기이므로 원래 타겟은 관여 안 함).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
OUT_DIR = os.path.dirname(__file__)
RESULT_PATH = os.path.join(OUT_DIR, "adversarial_validation_results.json")

TARGET_COL = "control_success"
ID_COL = "row_id"
SEED = 42
ITERATIONS = 500

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]
INGREDIENT_COLS = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"]

BEST_PARAMS = dict(
    learning_rate=0.03, depth=6, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def add_risk_score(df):
    df = df.copy()
    df["control_risk_score"] = (
        df["asof_pitcher_reverse_rate"] + df["asof_pitcher_middle_rate"] + df["asof_pitcher_ball_rate"]
    )
    df["control_risk_score_weighted"] = (
        0.4 * df["asof_pitcher_reverse_rate"] + 0.3 * df["asof_pitcher_middle_rate"] + 0.3 * df["asof_pitcher_ball_rate"]
    )
    return df


def main():
    print("Load train data + trackman context...", flush=True)
    df = load_data()
    df = add_risk_score(df).drop(columns=INGREDIENT_COLS)
    print(f" shape={df.shape}", flush=True)

    print("adversarial target 정의: season<=2023(0, 과거) vs season==2024(1, 최근)...", flush=True)
    df["is_recent"] = (df["season"] == 2024).astype(int)
    print(f" is_recent 비율: {df['is_recent'].mean():.4f}  (season==2024 행 수={df['is_recent'].sum()})", flush=True)

    id_mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        id_mappings[c] = {v: i for i, v in enumerate(uniq)}

    X = df.drop(columns=[ID_COL, TARGET_COL, "is_recent", "season"])  # season 자체는 답이 너무 뻔해서 제외, "다른 피처가 얼마나 season을 누설하는지"가 관심사
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    y = df["is_recent"]
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=SEED,
    )

    print(f"\nCatBoost(adversarial) 학습(n_features={X.shape[1]})...", flush=True)
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, cat_features=cat_idx, early_stopping_rounds=50,
        verbose=False, thread_count=-1, **BEST_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    elapsed = time.time() - t0
    val_pred = model.predict_proba(X_val)[:, 1]
    adv_auc = roc_auc_score(y_val, val_pred)
    print(f" best_iteration={model.get_best_iteration()}  adversarial_auc={adv_auc:.4f} ({elapsed:.1f}s)", flush=True)
    print(f" (0.5=완전히 구분 불가=이상적, 1.0=피처만 보고 연도를 완벽히 맞힘=위험 신호)", flush=True)

    importances = model.get_feature_importance(prettified=True)
    print("\n=== 상위 30개 피처 (adversarial importance, 연도 구분에 가장 크게 기여) ===", flush=True)
    for _, row in importances.head(30).iterrows():
        print(f"  {row['Feature Id']:40s} {row['Importances']:.3f}", flush=True)

    result = {
        "adversarial_auc": adv_auc,
        "n_features": int(X.shape[1]),
        "top_features": importances.to_dict(orient="records"),
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
