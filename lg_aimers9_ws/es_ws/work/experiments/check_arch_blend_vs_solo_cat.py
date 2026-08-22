"""train_arch_blend_bagged.py와 동일한 calib carve-out(random_state=42, 5%)에서
CatBoost 6시드 단독 calibrated BSS를 계산해, 블렌드(2080.01)와 직접 비교(apples-to-apples).
재학습 없음 -- jh_ws 저장된 모델 로드 + 예측만 수행."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
JH_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(_BASE)), "jh_ws", "v18_seed_bagging", "model")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]
RAW_ID_COLS = ["pitcher_id", "batter_id"]


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


with open(os.path.join(JH_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
    jh_meta = json.load(f)
id_mappings = jh_meta["id_mappings"]
cat_models = []
for seed in jh_meta["seeds"]:
    m = CatBoostClassifier()
    m.load_model(os.path.join(JH_MODEL_DIR, f"catboost_seed{seed}.cbm"))
    cat_models.append(m)

df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
context = joblib.load(os.path.join(_BASE, "model", "trackman_context.pkl"))
for spec in context.values():
    df = df.merge(spec["table"], on=spec["keys"], how="left")
y_all = df[TARGET_COL]

X = df.drop(columns=[ID_COL, TARGET_COL])
for c in RAW_ID_COLS:
    X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
for c in CAT_COLS:
    X[c] = X[c].astype(str)

idx_train, idx_calib = train_test_split(df.index, test_size=0.05, stratify=y_all, random_state=42)
X_calib, y_calib = X.loc[idx_calib], y_all.loc[idx_calib]

cat_raw = np.mean([m.predict_proba(X_calib)[:, 1] for m in cat_models], axis=0)
a, b = fit_platt_scaling(cat_raw, y_calib)
cat_calib = apply_platt_scaling(cat_raw, a, b)
print(f"CatBoost 6시드 단독 (동일 calib carve-out): raw={bss_score(cat_raw, y_calib):.2f}  calibrated={bss_score(cat_calib, y_calib):.2f}")
print("블렌드(cat 0.85/lgb 0.1/xgb 0.05) 결과(로그 참고): raw=2064.70  calibrated=2080.01")
