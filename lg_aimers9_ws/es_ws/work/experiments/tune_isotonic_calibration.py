"""Isotonic 보정 vs 현재 프로덕션(Platt/sigmoid) 보정 비교 — 8/18 다음 세션
후보 1순위 재시도.

배경(8/18 worklog 섹션 6-1): 지금 프로덕션(`pipeline/train_catboost.py`)은
Platt scaling(2-파라미터 로지스틱 직선 변환)만 쓰는데, 5% iid carve-out에서
raw->calibrated 격차(+11.5~+25.5, raw id 포함 여부에 따라 다름)가 2024
시간분할 홀드아웃 격차(+24.8~+9.48)와 다르게 나온 걸 보면 Platt이 못 잡는
비선형 미보정이 남아있을 가능성이 있다. jh_ws의 isotonic_blend(8/17)는
랜덤 K-Fold+unseen ID 문제가 섞여 있어 isotonic 자체 효과가 검증된 적이
없었다.

이 스크립트가 확인하려는 것 — 두 가지 별개 질문:
  (1) Isotonic이 Platt보다 BSS를 더 개선하는가?
  (2) Isotonic은 비모수(계단함수)라 표현력이 크지만 그만큼 작은 보정용
      홀드아웃에 과적합하기 쉽다 -- "보정을 fit한 데이터로 그대로 평가하면
      안 된다"는 게 지금까지 프로덕션 파이프라인의 원칙(전체 재학습 데이터로
      학습한 모델을 그 데이터로 재보정하면 리키지이므로 별도 carve-out에
      fit하는 것과 동일한 논리). 따라서 각 fold의 검증 시즌 내부를
      calib_fit/calib_eval로 다시 쪼개 fit과 eval을 반드시 분리하고,
      calib_fit 크기(20%/50%/80%)에 따라 isotonic이 얼마나 불안정해지는지
      (홀드아웃 크기 민감도) 확인한다.

FOLDS는 tune_rawid_cv_robust.py / tune_ensemble_cv_robust.py와 동일한
3-fold walk-forward(2022/2023/2024 검증). 각 fold는 한 번만 학습하고(raw
pred 재사용), calib_fit/calib_eval 분할만 여러 크기 x 여러 시드로 반복하므로
재학습 비용 없이 저렴하게 돈다.

주의: fold(->2023)는 8/17/8/18에 이미 기록된 "2023년 game_type F/R 관계
역전" 데이터 이상으로 BSS 자체가 붕괴하는 폴드다 -- 여기서 나오는 델타는
낮은 가중치로 해석할 것.

이 스크립트는 효과 크기만 측정한다. 채택이 결정되면 프로덕션 반영 시
CalibratedClassifierCV/IsotonicRegression 객체를 그대로 pickle하지 않고
(sklearn 버전 의존 리스크, 8/16 교훈) `feature_meta.json`에 정렬된
(x_thresholds, y_thresholds) 배열로 저장 후 추론 시 `np.interp`로 적용해야
Platt scaling과 동일한 무의존성 원칙을 지킨다.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
RAW_ID_COLS = ["pitcher_id", "batter_id"]  # 현재 프로덕션과 동일하게 항상 포함(수치형, label-encoded)

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold1(train<=2022,valid2023)", "train_seasons": [2019, 2020, 2021, 2022], "valid_season": 2023},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]

# calib_fit에 쓸 검증시즌 내 비율. 나머지는 calib_eval(보정이 한번도 못 본 데이터)로 BSS 측정.
CALIB_FIT_SIZES = [0.2, 0.5, 0.8]
SEEDS = [0, 1, 2]


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def fit_platt(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return lambda p: 1.0 / (1.0 + np.exp(-(a * np.asarray(p) + b)))


def fit_isotonic(raw_p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(raw_p), np.asarray(y))
    return lambda p: iso.predict(np.asarray(p))


def load_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        uniq = sorted(X[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        X[c] = X[c].astype(str).map(mapping).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def train_fold(fold, df, X):
    tr_mask = df["season"].isin(fold["train_seasons"])
    va_mask = df["season"] == fold["valid_season"]
    y_tr, y_va = df.loc[tr_mask, TARGET_COL], df.loc[va_mask, TARGET_COL]

    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, **BEST_PARAMS,
    )
    model.fit(X[tr_mask], y_tr, eval_set=(X[va_mask], y_va))
    raw_pred_va = model.predict_proba(X[va_mask])[:, 1]
    return raw_pred_va, y_va.to_numpy()


def run_calib_comparison(raw_pred_va, y_va, fold_name):
    rows = []
    for size in CALIB_FIT_SIZES:
        for seed in SEEDS:
            fit_idx, eval_idx = train_test_split(
                np.arange(len(y_va)), train_size=size, stratify=y_va, random_state=seed,
            )
            p_fit, y_fit = raw_pred_va[fit_idx], y_va[fit_idx]
            p_eval, y_eval = raw_pred_va[eval_idx], y_va[eval_idx]

            bss_raw_eval = bss_score(p_eval, y_eval)
            platt = fit_platt(p_fit, y_fit)
            bss_platt_eval = bss_score(platt(p_eval), y_eval)
            iso = fit_isotonic(p_fit, y_fit)
            bss_iso_eval = bss_score(iso(p_eval), y_eval)

            rows.append({
                "fold": fold_name, "calib_fit_size": size, "seed": seed,
                "n_fit": len(fit_idx), "n_eval": len(eval_idx),
                "bss_raw_eval": bss_raw_eval,
                "bss_platt_eval": bss_platt_eval,
                "bss_isotonic_eval": bss_iso_eval,
                "delta_platt_vs_raw": bss_platt_eval - bss_raw_eval,
                "delta_isotonic_vs_raw": bss_iso_eval - bss_raw_eval,
                "delta_isotonic_vs_platt": bss_iso_eval - bss_platt_eval,
            })
    return rows


def main():
    print("Load data...")
    df = load_base_df()
    X = build_features(df)
    print(f" shape={df.shape}")

    all_rows = []
    for fold in FOLDS:
        t0 = time.time()
        print(f"\n[{fold['name']}] training...")
        raw_pred_va, y_va = train_fold(fold, df, X)
        rows = run_calib_comparison(raw_pred_va, y_va, fold["name"])
        all_rows.extend(rows)
        dt = time.time() - t0
        print(f"  done in {dt:.0f}s, n_valid={len(y_va)}")
        for size in CALIB_FIT_SIZES:
            sub = [r for r in rows if r["calib_fit_size"] == size]
            mean_dp = sum(r["delta_platt_vs_raw"] for r in sub) / len(sub)
            mean_di = sum(r["delta_isotonic_vs_raw"] for r in sub) / len(sub)
            mean_dip = sum(r["delta_isotonic_vs_platt"] for r in sub) / len(sub)
            print(f"    calib_fit_size={size:.1f} (n_fit~{sub[0]['n_fit']}, n_eval~{sub[0]['n_eval']}, "
                  f"{len(SEEDS)} seeds): platt-raw={mean_dp:+.2f}  iso-raw={mean_di:+.2f}  iso-platt={mean_dip:+.2f}")

    print("\n=== 요약: calib_fit_size별 iso-platt delta (fold2=2024만, 프로덕션과 가장 근접) ===")
    fold2_rows = [r for r in all_rows if r["fold"] == FOLDS[-1]["name"]]
    for size in CALIB_FIT_SIZES:
        sub = [r for r in fold2_rows if r["calib_fit_size"] == size]
        vals = [r["delta_isotonic_vs_platt"] for r in sub]
        print(f"  size={size:.1f}: mean={sum(vals)/len(vals):+.2f}  min={min(vals):+.2f}  max={max(vals):+.2f}")

    print("\n=== 요약: fold별 iso-platt delta (calib_fit_size=0.5 고정) ===")
    for fold in FOLDS:
        sub = [r for r in all_rows if r["fold"] == fold["name"] and r["calib_fit_size"] == 0.5]
        vals = [r["delta_isotonic_vs_platt"] for r in sub]
        print(f"  {fold['name']}: mean={sum(vals)/len(vals):+.2f}  min={min(vals):+.2f}  max={max(vals):+.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tune_isotonic_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(
        "\n판단 기준: (a) 모든 fold/size에서 iso-platt delta가 일관되게 양수인가, "
        "(b) calib_fit_size가 작아질수록(0.2) 분산(min~max 폭)이 커지는가 -- 커진다면 "
        "isotonic이 과적합에 취약하다는 뜻이므로 채택 시 carve-out을 5%보다 키워야 함, "
        "(c) fold1(->2023)은 알려진 이상치 폴드이므로 낮은 가중치로 해석."
    )


if __name__ == "__main__":
    main()
