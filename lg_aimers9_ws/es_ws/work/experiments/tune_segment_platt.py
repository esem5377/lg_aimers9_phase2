"""game_type(F/R)별로 Platt scaling을 따로 fit하는 세그먼트 보정이 지금
프로덕션의 전역(단일 a,b) Platt 보정보다 나은지 검증.

배경: 지금까지 나온 것 중 가장 큰 리더보드 레버는 확률보정(Platt, 로컬
+24.8 -> 리더보드 +70)이었다. 그런데 지금은 전체 데이터에 대해 (a, b)
스칼라 하나로 통 보정한다 -- game_type=F/R 사이에 2023년부터 뚜렷한
control_success 레짐 차이가 있는 걸 감안하면(8/19 regime2023 실험에서
피처로는 실패했지만, 그건 "트리 분기용 피처"로서의 실패였지 "보정 곡선이
세그먼트별로 다를 수 있다"는 가설과는 다른 메커니즘), 원시 확률의
miscalibration 정도 자체가 game_type별로 다를 가능성이 있다.

Isotonic(비모수, 8/19 기각 -- 작은 carve-out에 과적합)과 달리 이건
"파라미터 2개짜리 직선 변환을 세그먼트 수만큼(2개, F/R) 나눠서 fit"하는
것이라 표현력 증가폭이 훨씬 작고 과적합 위험도 낮을 것으로 기대.

검증 방식: tune_isotonic_calibration.py와 동일하게 3-fold walk-forward
(2022/2023/2024 검증) 각 폴드 내부를 calib_fit/calib_eval로 분리(보정을
한번도 못 본 데이터에서 평가), 전역 Platt vs game_type별 세그먼트 Platt의
BSS를 비교한다. F가 전체의 ~11%뿐이라 calib_fit_size가 작을 때 세그먼트당
표본이 부족해질 수 있으므로 그 민감도도 같이 본다.

8/19 교훈 반영: fold(->2023)은 알려진 이상치(BSS가 바닥에 붙음)라 낮은
가중치로 해석. fold(->2022)/fold(->2024) 두 폴드가 독립적으로 일치하는지가
채택 여부의 핵심 기준(단일 폴드 근거는 이번 세션에 -6으로 반증된 전례가
있으므로 재사용하지 않는다).
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
SEGMENT_COL = "game_type"  # 세그먼트 보정 기준 컬럼

BEST_PARAMS = dict(
    learning_rate=0.01, depth=8, l2_leaf_reg=20.0,
    bagging_temperature=1.0, random_strength=1.0, border_count=32,
)

FOLDS = [
    {"name": "fold0(train<=2021,valid2022)", "train_seasons": [2019, 2020, 2021], "valid_season": 2022},
    {"name": "fold1(train<=2022,valid2023)", "train_seasons": [2019, 2020, 2021, 2022], "valid_season": 2023},
    {"name": "fold2(train<=2023,valid2024)", "train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_season": 2024},
]

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


def fit_segment_platt(raw_p, y, seg):
    raw_p, y, seg = np.asarray(raw_p), np.asarray(y), np.asarray(seg)
    fns = {}
    for s in np.unique(seg):
        mask = seg == s
        fns[s] = fit_platt(raw_p[mask], y[mask])

    def predict(p, seg_eval):
        p, seg_eval = np.asarray(p), np.asarray(seg_eval)
        out = np.empty_like(p, dtype=float)
        for s, fn in fns.items():
            mask = seg_eval == s
            if mask.any():
                out[mask] = fn(p[mask])
        return out

    return predict, {s: len(np.asarray(seg)[np.asarray(seg) == s]) for s in fns}


def load_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(os.path.join(MODEL_DIR, "trackman_context.pkl"))
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_features(df):
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in ["pitcher_id", "batter_id"]:  # 현재 프로덕션과 동일하게 항상 raw id 포함
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
    seg_va = df.loc[va_mask, SEGMENT_COL].to_numpy()

    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(
        iterations=2000, loss_function="Logloss", eval_metric="AUC", random_seed=42,
        cat_features=cat_idx, early_stopping_rounds=100, verbose=False, **BEST_PARAMS,
    )
    model.fit(X[tr_mask], y_tr, eval_set=(X[va_mask], y_va))
    raw_pred_va = model.predict_proba(X[va_mask])[:, 1]
    return raw_pred_va, y_va.to_numpy(), seg_va


def run_calib_comparison(raw_pred_va, y_va, seg_va, fold_name):
    rows = []
    for size in CALIB_FIT_SIZES:
        for seed in SEEDS:
            fit_idx, eval_idx = train_test_split(
                np.arange(len(y_va)), train_size=size, stratify=y_va, random_state=seed,
            )
            p_fit, y_fit, seg_fit = raw_pred_va[fit_idx], y_va[fit_idx], seg_va[fit_idx]
            p_eval, y_eval, seg_eval = raw_pred_va[eval_idx], y_va[eval_idx], seg_va[eval_idx]

            bss_raw_eval = bss_score(p_eval, y_eval)
            global_fn = fit_platt(p_fit, y_fit)
            bss_global_eval = bss_score(global_fn(p_eval), y_eval)
            seg_fn, seg_n = fit_segment_platt(p_fit, y_fit, seg_fit)
            bss_seg_eval = bss_score(seg_fn(p_eval, seg_eval), y_eval)

            rows.append({
                "fold": fold_name, "calib_fit_size": size, "seed": seed,
                "n_fit": len(fit_idx), "n_eval": len(eval_idx), "seg_n_fit": seg_n,
                "bss_raw_eval": bss_raw_eval,
                "bss_global_platt_eval": bss_global_eval,
                "bss_segment_platt_eval": bss_seg_eval,
                "delta_global_vs_raw": bss_global_eval - bss_raw_eval,
                "delta_segment_vs_raw": bss_seg_eval - bss_raw_eval,
                "delta_segment_vs_global": bss_seg_eval - bss_global_eval,
            })
    return rows


def main():
    print("Load data...")
    df = load_base_df()
    X = build_features(df)
    print(f" shape={df.shape}")
    print(f" segment({SEGMENT_COL}) value_counts:\n{df[SEGMENT_COL].value_counts()}")

    all_rows = []
    for fold in FOLDS:
        t0 = time.time()
        print(f"\n[{fold['name']}] training...")
        raw_pred_va, y_va, seg_va = train_fold(fold, df, X)
        rows = run_calib_comparison(raw_pred_va, y_va, seg_va, fold["name"])
        all_rows.extend(rows)
        dt = time.time() - t0
        print(f"  done in {dt:.0f}s, n_valid={len(y_va)}")
        for size in CALIB_FIT_SIZES:
            sub = [r for r in rows if r["calib_fit_size"] == size]
            mean_dg = sum(r["delta_global_vs_raw"] for r in sub) / len(sub)
            mean_ds = sum(r["delta_segment_vs_raw"] for r in sub) / len(sub)
            mean_dsg = sum(r["delta_segment_vs_global"] for r in sub) / len(sub)
            print(f"    calib_fit_size={size:.1f} (n_fit~{sub[0]['n_fit']}, n_eval~{sub[0]['n_eval']}, "
                  f"seg_n_fit~{sub[0]['seg_n_fit']}, {len(SEEDS)} seeds): "
                  f"global-raw={mean_dg:+.2f}  segment-raw={mean_ds:+.2f}  segment-global={mean_dsg:+.2f}")

    print("\n=== 요약: calib_fit_size별 segment-global delta (fold2=2024만, 프로덕션과 가장 근접) ===")
    fold2_rows = [r for r in all_rows if r["fold"] == FOLDS[-1]["name"]]
    for size in CALIB_FIT_SIZES:
        sub = [r for r in fold2_rows if r["calib_fit_size"] == size]
        vals = [r["delta_segment_vs_global"] for r in sub]
        print(f"  size={size:.1f}: mean={sum(vals)/len(vals):+.2f}  min={min(vals):+.2f}  max={max(vals):+.2f}")

    print("\n=== 요약: fold별 segment-global delta (calib_fit_size=0.5 고정) ===")
    for fold in FOLDS:
        sub = [r for r in all_rows if r["fold"] == fold["name"] and r["calib_fit_size"] == 0.5]
        vals = [r["delta_segment_vs_global"] for r in sub]
        print(f"  {fold['name']}: mean={sum(vals)/len(vals):+.2f}  min={min(vals):+.2f}  max={max(vals):+.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tune_segment_platt_results.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(
        "\n판단 기준(8/19 교훈 반영): (a) fold0(->2022)과 fold2(->2024) 두 폴드가 "
        "독립적으로 비슷한 크기·같은 방향으로 일치해야 신뢰(단일 폴드 근거는 채택 안 함), "
        "(b) calib_fit_size가 작아질수록 분산이 커지는지(과적합 취약성, 특히 F 세그먼트 "
        "표본이 작아서 더 취약할 수 있음), (c) fold1(->2023)은 알려진 이상치이므로 참고만."
    )


if __name__ == "__main__":
    main()
