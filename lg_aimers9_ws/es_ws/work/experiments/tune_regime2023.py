"""2023 game_type(F/R) 레짐 변화를 명시적 피처로 추가했을 때 효과 검증 --
8/18 워크로그 다음 세션 후보 2순위.

배경: train.csv에서 season별 game_type=F의 control_success 평균이
2019~2022엔 R보다 뚜렷이 높다가(+0.06~+0.20) 2023부터 역전됨(-0.03, 2024도
동일 방향 -0.03). game_type=R은 2019(0.549)->2024(0.490)로 완만히
단조감소하는 것과 달리 F는 2023에서 급락(0.709->0.473)한다 -- 점진적 추세가
아니라 레짐 자체가 바뀐 패턴.

지금 프로덕션(pipeline/train_catboost.py)은 season(수치형)과 game_type(범주형)을
각각 독립 피처로만 넣는다. CatBoost 트리는 원칙적으로 season<2023 분기 뒤
game_type 분기를 순차로 학습할 수 있지만, game_type 자체의 ordered target
statistic(카테고리 인코딩)은 학습 데이터 전체 시간축에서 누적되므로 최근
레짐 변화가 희석될 수 있다. 이 스크립트는 season>=2023 x game_type 교차를
새 범주형 피처로 명시적으로 추가하면 도움이 되는지 확인한다.

검증 방식: tune_rawid_cv_robust.py와 동일한 3-fold walk-forward
(2022/2023/2024 검증) + Platt 보정 후 BSS. 다만 이번 후보는 "2023부터
바뀐 레짐이 2025(test)에도 이어지는지"가 핵심 불확실성이므로, 8/18
워크로그에 못박은 대로 fold1(->2023)과 fold2(->2024) 둘 다에서 방향이
일관돼야 신뢰. fold1은 다른 실험들에서 "이상 폴드"로 취급돼 왔지만, 이번
후보 자체가 그 이상 현상을 피처화하려는 시도이므로 여기서는 fold1도
정상적으로 가중치를 둔다(다른 실험들과 성격이 다름에 주의).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../es_ws/work
DATA_DIR = os.path.join(os.path.dirname(_BASE), "open", "data")
MODEL_DIR = os.path.join(_BASE, "model")
OUT_DIR = os.path.join(_BASE, "output")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
REGIME_COL = "regime_gametype"

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold1(train<=2022,valid2023)", "train_seasons": [2019, 2020, 2021, 2022], "valid_season": 2023},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_p, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def load_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df, with_regime):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in ["pitcher_id", "batter_id"]:  # 현재 프로덕션과 동일하게 항상 raw id 포함
        uniq = sorted(X[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        X[c] = X[c].astype(str).map(mapping).astype(int)
    cat_cols = list(CAT_COLS)
    if with_regime:
        regime = pd.Series(
            np.where(df["season"].to_numpy() >= 2023, "post23", "pre23"), index=df.index)
        X[REGIME_COL] = regime + "_" + df["game_type"].astype(str)
        cat_cols = cat_cols + [REGIME_COL]
    for c in cat_cols:
        X[c] = X[c].astype(str)
    return X, cat_cols


def fit_catboost_calibrated(X_tr, y_tr, X_va, y_va, cat_cols):
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, **BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    raw_pred = model.predict_proba(X_va)[:, 1]
    calib_pred = apply_platt(raw_pred, fit_platt(raw_pred, y_va))
    return bss_score(raw_pred, y_va), bss_score(calib_pred, y_va)


def run_fold(fold, df):
    tr_mask = df["season"].isin(fold["train_seasons"])
    va_mask = df["season"] == fold["valid_season"]
    y_tr, y_va = df.loc[tr_mask, TARGET_COL], df.loc[va_mask, TARGET_COL]

    t0 = time.time()
    X_base, cat_base = build_features(df, with_regime=False)
    bss_raw_base, bss_calib_base = fit_catboost_calibrated(
        X_base[tr_mask], y_tr, X_base[va_mask], y_va, cat_base)

    X_regime, cat_regime = build_features(df, with_regime=True)
    bss_raw_regime, bss_calib_regime = fit_catboost_calibrated(
        X_regime[tr_mask], y_tr, X_regime[va_mask], y_va, cat_regime)
    dt = time.time() - t0

    delta_calib = bss_calib_regime - bss_calib_base
    print(f"  [{fold['name']}] ({dt:.0f}s) n_train={tr_mask.sum()} n_valid={va_mask.sum()}")
    print(f"    base:   raw={bss_raw_base:.2f}  calib={bss_calib_base:.2f}")
    print(f"    regime: raw={bss_raw_regime:.2f}  calib={bss_calib_regime:.2f}")
    print(f"    delta(calibrated, regime - base) = {delta_calib:+.2f}")

    return {
        "fold": fold["name"],
        "bss_raw_base": bss_raw_base, "bss_calib_base": bss_calib_base,
        "bss_raw_regime": bss_raw_regime, "bss_calib_regime": bss_calib_regime,
        "delta_calibrated": delta_calib,
    }


def main():
    print("Load data...")
    df = load_base_df()
    print(f" shape={df.shape}")

    print("\n3-fold walk-forward: baseline vs +season>=2023 x game_type regime feature...")
    results = [run_fold(f, df) for f in FOLDS]

    deltas = [r["delta_calibrated"] for r in results]
    fold12_deltas = deltas[1:]  # fold1(->2023), fold2(->2024) -- 이 후보의 핵심 판단 기준
    print(f"\n=== 요약 ===")
    for r in results:
        print(f"  {r['fold']}: delta={r['delta_calibrated']:+.2f}")
    print(f"  fold1+fold2(2023,2024) 모두 양수 = {all(d > 0 for d in fold12_deltas)}")
    print(f"  전체 mean delta = {sum(deltas) / len(deltas):+.2f}")
    print(f"  fold1+fold2 mean delta = {sum(fold12_deltas) / len(fold12_deltas):+.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_regime2023_results.json"), "w") as f:
        json.dump({
            "folds": results,
            "mean_delta_all": sum(deltas) / len(deltas),
            "mean_delta_fold1_fold2": sum(fold12_deltas) / len(fold12_deltas),
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/tune_regime2023_results.json")
    print(
        "\n판단 기준: fold1(->2023)과 fold2(->2024) 둘 다 양수여야 신뢰. "
        "test.csv(2025)가 실제로 이 레짐을 이어받는지는 리더보드 실측 전엔 알 수 없음 -- "
        "로컬 신호가 강해도 test.csv 표본이 game_type=R 위주라면 전이 효과가 작을 수 있음에 유의."
    )


if __name__ == "__main__":
    main()
