"""팀 최고 기록(909점, jh_ws/submit_jh_ensemble.zip)의 CatBoost 컴포넌트는
지금까지 한 번도 하이퍼파라미터 탐색을 거치지 않았다 (train.py/train_v5.py
전부 depth=6/lr=0.03/iterations=1000 고정값). Optuna 앙상블 블렌드에서
CatBoost가 가장 큰 가중치(원본 62.5%~77%)를 받는 지배적 컴포넌트인데도
그렇다 — es_ws의 CatBoost 랜덤서치(tune_catboost.py, 전혀 다른 피처셋/검증
방식)와 별개로, 이 파이프라인(진짜 pitcher_id group k-fold, jh 선택 피처
45개) 안에서는 완전히 미탐색 영역.

train_v5.py의 1~3단계(데이터 로딩/피처 엔지니어링/피처 선택)를 그대로
재현해서 동일한 X/y/selected_features를 얻은 뒤, CatBoost 하이퍼파라미터만
랜덤서치. 시간 절약을 위해 탐색 단계는 5-fold 중 fold 0~2만 사용하고,
상위 후보만 5-fold 전체로 재검증(폴드 전체 일관성 확인 — 오늘 두 번 겪은
"국소 검증에서만 좋아 보이는" 함정을 피하기 위함).
"""
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

DATA_DIR = './data'
SEED = 42
N_TRIALS = 10
SEARCH_FOLDS = [0, 1, 2]  # 5-fold 중 탐색용 부분집합

PARAM_GRID = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 10, 20],
    "bagging_temperature": [0, 0.5, 1, 2],
    "random_strength": [0.5, 1, 2, 5],
    "border_count": [32, 64, 128, 254],
}
BASELINE_PARAMS = dict(learning_rate=0.03, depth=6, l2_leaf_reg=3, bagging_temperature=1,
                        random_strength=1, border_count=254)  # train_v5.py 현재 값


def prepare_data():
    print("[1/3] 데이터 로딩 및 사전 집계 (train_v5.py와 동일)...", flush=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    tm_path = os.path.join(DATA_DIR, 'trackman_history.csv')

    if os.path.exists(tm_path):
        tm_df = pd.read_csv(tm_path)
        hist_p_col = [c for c in tm_df.columns if 'pitcher' in c.lower()][0]
        rel_x = [c for c in tm_df.columns if 'rel' in c.lower() and ('x' in c.lower() or 'side' in c.lower())][0]
        rel_z = [c for c in tm_df.columns if 'rel' in c.lower() and ('z' in c.lower() or 'height' in c.lower())][0]
        agg_df = tm_df.groupby(hist_p_col).agg(
            avg_rel_x=(rel_x, 'mean'), avg_rel_z=(rel_z, 'mean'),
            std_rel_x=(rel_x, 'std'), std_rel_z=(rel_z, 'std')
        ).reset_index()
        train_p_col_raw = [c for c in train_df.columns if 'pitcher' in c.lower()][0]
        train_df = train_df.merge(agg_df, left_on=train_p_col_raw, right_on=hist_p_col, how='left')

    if 'avg_rel_x' in train_df.columns and 'rel_x' in train_df.columns:
        train_df['release_dev'] = np.sqrt(
            (train_df['rel_x'] - train_df['avg_rel_x']) ** 2 + (train_df['rel_z'] - train_df['avg_rel_z']) ** 2)
    if 'balls_before' in train_df.columns and 'strikes_before' in train_df.columns:
        train_df['cnt_diff'] = train_df['strikes_before'] - train_df['balls_before']
        train_df['is_strike_pressured'] = (train_df['balls_before'] == 3).astype(int)
        train_df['is_two_strike'] = (train_df['strikes_before'] == 2).astype(int)
    if 'li' in train_df.columns and 'asof_pitcher_middle_rate' in train_df.columns:
        train_df['leverage_middle_risk'] = train_df['li'] * train_df['asof_pitcher_middle_rate']

    ignore_cols = ['row_id', 'control_success']
    train_p_col = 'pitcher_id'
    features = [c for c in train_df.columns if c not in ignore_cols]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]

    for c in cat_cols:
        train_df[c] = train_df[c].astype(str).fillna('missing')
        mapping = {v: i for i, v in enumerate(sorted(train_df[c].unique()))}
        train_df[c] = train_df[c].map(mapping)

    X = train_df[features].copy()
    y = train_df['control_success'].copy()

    print("[2/3] 다중공선성 제거 및 피처 선별 (train_v5.py와 동일)...", flush=True)
    num_cols = [c for c in features if c not in cat_cols and c != train_p_col]
    corr_matrix = X[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > 0.95)]
    if to_drop:
        X = X.drop(columns=to_drop)
        features = [f for f in features if f not in to_drop]

    import lightgbm as lgb
    selector_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    selector_model.fit(X, y)
    importance_df = pd.DataFrame({'feature': features, 'importance': selector_model.feature_importances_})
    selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
    X = X[selected_features].copy()
    print(f"  selected_features n={len(selected_features)}", flush=True)

    groups = train_df[train_p_col]
    cat_feature_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    return X, y, groups, cat_feature_idx


def fit_eval(X, y, groups, cat_feature_idx, params, fold_ids, splits, seed_offset=0):
    aucs = []
    for fold in fold_ids:
        train_idx, val_idx = splits[fold]
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = CatBoostClassifier(
            iterations=1500, random_seed=42 + fold + seed_offset,
            cat_features=cat_feature_idx, early_stopping_rounds=100, verbose=False, thread_count=-1,
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        pred = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, pred))
    return aucs


def main():
    t0 = time.time()
    X, y, groups, cat_feature_idx = prepare_data()

    print("[3/3] 5-fold 진짜 group k-fold 스플릿 생성...", flush=True)
    gkf = StratifiedGroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=groups))

    rng = random.Random(SEED)
    trials = [BASELINE_PARAMS]
    for _ in range(N_TRIALS):
        trials.append({k: rng.choice(v) for k, v in PARAM_GRID.items()})

    print(f"\n=== 탐색 단계: fold {SEARCH_FOLDS} 기준, {len(trials)}개 config (baseline 포함) ===", flush=True)
    results = []
    for i, params in enumerate(trials):
        t1 = time.time()
        aucs = fit_eval(X, y, groups, cat_feature_idx, params, SEARCH_FOLDS, splits)
        mean_auc = sum(aucs) / len(aucs)
        tag = "BASELINE" if params is BASELINE_PARAMS else f"trial{i}"
        print(f"[{tag}] mean_auc(fold{SEARCH_FOLDS})={mean_auc:.5f} fold_aucs={['%.5f' % a for a in aucs]} "
              f"params={params} ({time.time()-t1:.0f}s)", flush=True)
        results.append({"tag": tag, "params": params, "search_mean_auc": mean_auc, "search_fold_aucs": aucs})

    baseline_search_auc = results[0]["search_mean_auc"]
    results_sorted = sorted(results[1:], key=lambda r: -r["search_mean_auc"])
    top2 = results_sorted[:2]

    print(f"\n=== 상위 후보 전체 5-fold 재검증 (baseline 포함) ===", flush=True)
    all_folds = [0, 1, 2, 3, 4]
    baseline_full = fit_eval(X, y, groups, cat_feature_idx, BASELINE_PARAMS, all_folds, splits)
    baseline_full_mean = sum(baseline_full) / len(baseline_full)
    print(f"[BASELINE full5fold] mean_auc={baseline_full_mean:.5f} fold_aucs={['%.5f' % a for a in baseline_full]}", flush=True)

    for cand in top2:
        full_aucs = fit_eval(X, y, groups, cat_feature_idx, cand["params"], all_folds, splits)
        full_mean = sum(full_aucs) / len(full_aucs)
        deltas = [a - b for a, b in zip(full_aucs, baseline_full)]
        consistent = all(d > 0 for d in deltas)
        print(f"[{cand['tag']} full5fold] mean_auc={full_mean:.5f} fold_aucs={['%.5f' % a for a in full_aucs]} "
              f"delta_vs_baseline={full_mean-baseline_full_mean:+.5f} fold_deltas={['%+.5f' % d for d in deltas]} "
              f"모든폴드에서baseline보다좋음={consistent} params={cand['params']}", flush=True)

    print(f"\ntotal elapsed: {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
