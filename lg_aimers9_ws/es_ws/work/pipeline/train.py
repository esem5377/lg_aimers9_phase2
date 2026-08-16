"""LG Aimers 9 - control_success 예측 베이스라인 학습 스크립트.

- train.csv(2019~2024)로 학습.
- 실제 평가(test.csv)는 2025시즌이므로, 검증은 최근 시즌(2024)을 홀드아웃하는
  시간 기반 분할을 사용해 미래 시즌 일반화 성능을 가늠한다.
- LightGBM은 결측치와 범주형(category dtype)을 네이티브로 처리하므로
  별도 인코딩/대치 없이 바로 사용한다.
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

DATA_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/open/data"
OUT_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/output"
MODEL_DIR = "/home/esem5377/lg_aimers9_ws/lg_aimers9_ws/es_ws/work/model"

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]

# raw pitcher_id/batter_id(792/830개 범주)를 그대로 범주형으로 넣으면 트리가
# 선수 단위로 과적합해 train gain은 압도적으로 크지만 2024 홀드아웃 AUC는
# 오히려 떨어진다 (0.5377 -> 제거 시 0.5489, 실험 확인 완료). 선수 성향은
# 이미 leak-safe하게 계산된 asof_* 이력 피처로 반영되므로 raw id는 제외한다.
DROP_COLS = ["pitcher_id", "batter_id"]


def load_train():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    return df


def attach_trackman_context(df, context):
    """trackman_history 기반 상황 단위 통계 피처를 좌측 조인으로 붙인다.

    선수 단위 join이 불가능해(build_trackman_context.py 참고) 카운트/아웃,
    투타 좌우, 이닝/초말 같은 공통 상황 키로만 묶은 집계값이라 개별 행의
    결과 정보를 담지 않는다.
    """
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[c for c in [ID_COL, TARGET_COL] + DROP_COLS if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Load train data...")
    df = load_train()
    print(f" shape={df.shape}")

    print("Attach trackman context features...")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    df = attach_trackman_context(df, context)
    print(f" shape after join={df.shape}")

    # 팀 단위 as-of 이력(build_team_history.py, attach_team_history())은 실험 결과
    # 2024 홀드아웃 AUC를 0.5489 -> 0.5466으로 오히려 떨어뜨려 사용하지 않는다.
    # season이 이미 시대별 추세를 잡고 있어 팀별 잔차 신호는 노이즈에 가까웠던 것으로 보임.

    # ---- 시간 기반 분할: 2019~2023 학습 / 2024 검증 ----
    # 실제 평가(test.csv)는 2025시즌이므로 최근 시즌 홀드아웃이 랜덤 분할보다
    # 실전 성능을 더 잘 대변한다.
    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024

    X_all = build_features(df)
    y_all = df[TARGET_COL]

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[valid_mask], y_all[valid_mask]
    print(f" train={X_tr.shape}  valid(2024)={X_va.shape}")

    params = dict(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    print("Fit (time-split validation)...")
    model_cv = lgb.LGBMClassifier(**params)
    model_cv.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    va_pred = model_cv.predict_proba(X_va)[:, 1]
    metrics = {
        "valid_season": 2024,
        "auc": roc_auc_score(y_va, va_pred),
        "logloss": log_loss(y_va, va_pred),
        "accuracy@0.5": accuracy_score(y_va, (va_pred >= 0.5).astype(int)),
        "best_iteration": model_cv.best_iteration_,
    }
    print("Validation metrics:", json.dumps(metrics, indent=2))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- feature importance ----
    imp = pd.Series(model_cv.feature_importances_, index=X_tr.columns)
    imp = imp.sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))

    # ---- 전체 데이터(2019~2024)로 최종 모델 재학습 ----
    print("\nRefit on full data with best_iteration...")
    final_params = dict(params)
    final_params["n_estimators"] = max(model_cv.best_iteration_, 1)
    model_final = lgb.LGBMClassifier(**final_params)
    model_final.fit(X_all, y_all)

    joblib.dump(model_final, os.path.join(MODEL_DIR, "lgbm.pkl"))
    # 학습 시 사용한 컬럼 순서 / 범주형 컬럼 / 범주값 목록을 저장해
    # 추론 시 동일한 category 코드 매핑을 재현한다 (test에 없는 범주는 NaN 처리).
    cat_categories = {c: list(X_all[c].cat.categories) for c in CAT_COLS}
    joblib.dump(
        {"columns": list(X_all.columns), "cat_cols": CAT_COLS,
         "cat_categories": cat_categories},
        os.path.join(MODEL_DIR, "feature_meta.pkl"),
    )
    print(f"Saved model to {MODEL_DIR}/lgbm.pkl")


if __name__ == "__main__":
    main()
