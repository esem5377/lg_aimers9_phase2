"""974 레시피(CAT_COLS/RAW_ID_COLS/trackman context) 그대로, 모델만
LightGBM으로 바꿔서 CatBoost와 동일한 방법론(Optuna, BSS 직접 최적화,
fold0/fold2 dual-axis)으로 하이퍼파라미터 탐색. 8/21 `diag_ensemble.py`
비교에서 LGBM이 "합리적으로 보이는 기본값"(튜닝 안 됨)만으로 CatBoost에
크게 뒤졌던 것(754.98 vs 833.08)이 튜닝 부족 때문인지, 아니면 이 데이터
자체가 CatBoost의 ordered target encoding에 구조적으로 유리한 것인지
확인하는 게 목적. CatBoost용 `optuna_bss_search.py`와 동일한 검증 설계
(fold0=train<=2021/eval==2022, fold2=train<=2023/eval==2024, 정직한
calibration, id_mappings는 fold TRAIN 파티션에서만 생성, 탐색 속도용
서브샘플링)를 그대로 재사용.
"""
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\USER\Desktop\open\data"
CONTEXT_PATH = (
    r"C:\Users\USER\Desktop\lg_aimers9_phase2\lg_aimers9_ws\es_ws\work\model\trackman_context.pkl"
)
OUT_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(OUT_DIR, "optuna_lgbm_search_log.jsonl")
RESULT_PATH = os.path.join(OUT_DIR, "optuna_lgbm_search_result.json")

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
RAW_ID_COLS = ["pitcher_id", "batter_id"]

# 8/21 diag_ensemble.py에서 쓴 "튜닝 안 된 기본값" -- 비교 기준점으로 trial 0에 enqueue
BASELINE_PARAMS = dict(
    num_leaves=63, learning_rate=0.02, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3,
)

SEARCH_ESTIMATORS = 1500
SEARCH_TRAIN_CAP = 300_000
N_TRIALS = 30
SEED = 42


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


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    context = joblib.load(CONTEXT_PATH)
    for spec in context.values():
        df = df.merge(spec["table"], on=spec["keys"], how="left")
    return df


def build_id_mappings(df):
    mappings = {}
    for c in RAW_ID_COLS:
        uniq = sorted(df[c].astype(str).unique())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def build_features(df, id_mappings):
    """LGBM 네이티브 categorical(category dtype)용 -- CatBoost의 str과 다름."""
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for c in RAW_ID_COLS:
        X[c] = X[c].astype(str).map(id_mappings[c]).fillna(-1).astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str).astype("category")
    return X


print("Loading data + trackman context...", flush=True)
t0 = time.time()
DF = load_data()
print(f" shape={DF.shape}  ({time.time()-t0:.1f}s)", flush=True)

FOLD_SPECS = [
    {"name": "fold0_2022", "train_mask": DF["season"] <= 2021, "eval_mask": DF["season"] == 2022},
    {"name": "fold2_2024", "train_mask": DF["season"] <= 2023, "eval_mask": DF["season"] == 2024},
]

PREPARED_FOLDS = []
for spec in FOLD_SPECS:
    train_df = DF[spec["train_mask"]]
    eval_df = DF[spec["eval_mask"]]
    id_mappings = build_id_mappings(train_df)
    X_tr_full = build_features(train_df, id_mappings)
    y_tr_full = train_df[TARGET_COL].reset_index(drop=True)
    X_tr_full = X_tr_full.reset_index(drop=True)
    X_ev = build_features(eval_df, id_mappings)
    y_ev = eval_df[TARGET_COL].reset_index(drop=True)
    X_ev = X_ev.reset_index(drop=True)

    if len(X_tr_full) > SEARCH_TRAIN_CAP:
        X_search, _, y_search, _ = train_test_split(
            X_tr_full, y_tr_full, train_size=SEARCH_TRAIN_CAP,
            stratify=y_tr_full, random_state=SEED,
        )
        X_search = X_search.reset_index(drop=True)
        y_search = y_search.reset_index(drop=True)
    else:
        X_search, y_search = X_tr_full, y_tr_full

    PREPARED_FOLDS.append({
        "name": spec["name"],
        "X_search": X_search, "y_search": y_search,
        "X_ev": X_ev, "y_ev": y_ev,
    })
    print(f" [{spec['name']}] train_full={len(X_tr_full)} -> search={len(X_search)}  eval={len(X_ev)}", flush=True)


def eval_params(params, trial_idx):
    fold_scores = {}
    for pf in PREPARED_FOLDS:
        X_tr, X_calib, y_tr, y_calib = train_test_split(
            pf["X_search"], pf["y_search"], test_size=0.05,
            stratify=pf["y_search"], random_state=SEED,
        )
        model = LGBMClassifier(
            n_estimators=SEARCH_ESTIMATORS, objective="binary",
            random_state=SEED, n_jobs=-1, verbosity=-1,
            **params,
        )
        model.fit(
            X_tr, y_tr, eval_set=[(X_calib, y_calib)],
            categorical_feature=CAT_COLS,
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        calib_raw = model.predict_proba(X_calib)[:, 1]
        a, b = fit_platt(calib_raw, y_calib)

        ev_raw = model.predict_proba(pf["X_ev"])[:, 1]
        ev_calib = apply_platt(ev_raw, a, b)

        fold_scores[pf["name"]] = {
            "bss_calibrated": bss_score(ev_calib, pf["y_ev"]),
            "bss_raw": bss_score(ev_raw, pf["y_ev"]),
            "best_iteration": model.best_iteration_,
        }
    return fold_scores


def objective(trial):
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
    )
    t_start = time.time()
    fold_scores = eval_params(params, trial.number)
    elapsed = time.time() - t_start

    avg_bss = np.mean([v["bss_calibrated"] for v in fold_scores.values()])
    record = {
        "trial": trial.number, "params": params, "fold_scores": fold_scores,
        "avg_bss_calibrated": avg_bss, "elapsed_sec": elapsed,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"[trial {trial.number}] avg_bss={avg_bss:.2f}  "
        f"fold0={fold_scores['fold0_2022']['bss_calibrated']:.2f}  "
        f"fold2={fold_scores['fold2_2024']['bss_calibrated']:.2f}  "
        f"({elapsed:.1f}s)  params={params}",
        flush=True,
    )
    return avg_bss


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(BASELINE_PARAMS)

    print(f"\nStarting Optuna search: {N_TRIALS} trials...", flush=True)
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print("\n=== SEARCH DONE ===", flush=True)
    print(f"Best trial: #{best.number}  avg_bss={best.value:.2f}", flush=True)
    print(f"Best params: {best.params}", flush=True)

    baseline_trial = study.trials[0]
    print(f"\nBaseline(trial 0, 8/21 diag_ensemble.py 미튜닝 값): avg_bss={baseline_trial.value:.2f}", flush=True)
    print(f"Delta (best - baseline): {best.value - baseline_trial.value:+.2f}", flush=True)
    print(f"참고: 같은 방식 CatBoost 30-trial 탐색 결과 -- fold0/fold2 동시개선 조합 0개,"
          f" 기존 BEST_PARAMS가 이미 fold0 2위권", flush=True)

    top5 = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9, reverse=True)[:5]
    result = {
        "baseline_avg_bss": baseline_trial.value,
        "baseline_params": BASELINE_PARAMS,
        "best_trial_number": best.number,
        "best_avg_bss": best.value,
        "best_params": best.params,
        "delta_vs_baseline": best.value - baseline_trial.value,
        "top5": [{"trial": t.number, "avg_bss": t.value, "params": t.params} for t in top5],
        "n_trials": N_TRIALS,
        "search_estimators": SEARCH_ESTIMATORS,
        "search_train_cap": SEARCH_TRAIN_CAP,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
