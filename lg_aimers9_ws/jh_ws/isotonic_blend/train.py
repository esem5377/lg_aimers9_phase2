import os
import time
import random
import warnings

# NOTE: lightgbm/catboost는 반드시 pandas보다 먼저 import해야 한다.
# 이 Windows 환경에서 pandas를 먼저 import하면 lightgbm의 네이티브 DLL과
# 충돌해 Dataset 생성 시 access violation이 발생하는 것을 확인했다 (import 순서 문제).
import lightgbm as lgb
from catboost import CatBoostClassifier

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')

DATA_DIR = 'data'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_THREADS = 8


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def brier_score(p, y):
    return np.mean((p - y) ** 2)


def bss_score(p, y):
    r = y.mean()
    baseline = r * (1 - r)
    bs = brier_score(p, y)
    return max(0.0, 100000 * (1 - bs / baseline))


def main():
    seed_everything()

    log("[1/6] train.csv 로딩...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    log(f"  shape={train_df.shape}")

    # -----------------------------------------------------------------
    # 피처 구성
    #   - trackman_history.csv는 pitcher_id(train) <-> pitcher_trackman_id
    #     간 공유 ID가 전혀 없어(overlap 0) 신뢰 가능한 조인이 불가하므로 사용하지 않는다.
    #   - 대회 제공 asof_* 컬럼은 투구 직전 시점까지의 과거 기록으로 사전 계산된
    #     공식 피처이므로 안전하게 사용 가능하다 (leakage 없음).
    # -----------------------------------------------------------------
    ignore_cols = ['row_id', 'control_success']
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    cat_cols = ['top_bottom', 'game_type', 'base_state', 'pitcher_id', 'batter_id',
                'pitcher_team_id', 'batter_team_id', 'pitcher_hand', 'batter_hand']
    cat_cols = [c for c in cat_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    log(f"  feature 수={len(feature_cols)} (범주형={len(cat_cols)}, 수치형={len(num_cols)})")

    # -----------------------------------------------------------------
    # 범주형 인코딩: train.csv만으로 매핑을 만들고, 학습에 없던 값은
    # 별도의 'unknown' 코드로 고정 배정한다 (test 정보 사용 없음).
    # -----------------------------------------------------------------
    cat_mappings = {}
    for c in cat_cols:
        train_df[c] = train_df[c].astype(str)
        uniques = sorted(train_df[c].unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        cat_mappings[c] = mapping
        unk_code = len(mapping)
        train_df[c] = train_df[c].map(mapping).astype(int)
        train_df[c] = pd.Categorical(train_df[c], categories=list(range(unk_code + 1)))

    X = train_df[feature_cols].copy()
    y = train_df['control_success'].astype(int).copy()

    log(f"  target rate r={y.mean():.5f}")

    # -----------------------------------------------------------------
    # 5-Fold CV. 각 fold에서 OOF 예측은 해당 fold 학습에 전혀 관여하지 않은
    # 데이터로만 산출한다. Early stopping용 검증셋은 OOF fold와 별도로
    # train 쪽에서만 떼어내어(carve-out), OOF/캘리브레이션 시 이중사용(leak)을 막는다.
    # -----------------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    lgb_models = []
    cat_models = []

    log("[2/6] 5-Fold LightGBM + CatBoost 학습 시작...")
    for fold, (trainval_idx, oof_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_trainval, y_trainval = X.iloc[trainval_idx], y.iloc[trainval_idx]
        X_oof, y_oof = X.iloc[oof_idx], y.iloc[oof_idx]

        X_tr, X_es, y_tr, y_es = train_test_split(
            X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=SEED
        )

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            objective='binary', random_state=SEED + fold, n_jobs=N_THREADS, verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr, eval_set=[(X_es, y_es)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
        oof_lgb[oof_idx] = lgb_model.predict_proba(X_oof)[:, 1]
        lgb_models.append(lgb_model)

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=3000, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='Logloss',
            random_seed=SEED + fold, thread_count=N_THREADS, verbose=0
        )
        cat_model.fit(
            X_tr, y_tr, eval_set=(X_es, y_es),
            cat_features=cat_cols, early_stopping_rounds=100
        )
        oof_cat[oof_idx] = cat_model.predict_proba(X_oof)[:, 1]
        cat_models.append(cat_model)

        fold_bss_lgb = bss_score(oof_lgb[oof_idx], y_oof.values)
        fold_bss_cat = bss_score(oof_cat[oof_idx], y_oof.values)
        log(f"  └ Fold {fold} 완료 ({time.time()-t0:.0f}s) | "
            f"LGB best_iter={lgb_model.best_iteration_} BSS={fold_bss_lgb:.1f} | "
            f"CAT best_iter={cat_model.get_best_iteration()} BSS={fold_bss_cat:.1f}")

    joblib.dump(lgb_models, os.path.join(MODEL_DIR, 'lgb_models.pkl'))
    joblib.dump(cat_models, os.path.join(MODEL_DIR, 'cat_models.pkl'))

    log("[3/6] OOF 기준 앙상블 최적 가중치 탐색...")
    best_w, best_bs = 0.5, np.inf
    for w in np.arange(0.0, 1.0001, 0.01):
        blend = w * oof_lgb + (1 - w) * oof_cat
        bs = brier_score(blend, y.values)
        if bs < best_bs:
            best_bs, best_w = bs, w
    log(f"  └ best_w(lgb)={best_w:.2f}, best_w(cat)={1-best_w:.2f} | OOF Brier={best_bs:.6f} | OOF BSS={bss_score(best_w*oof_lgb+(1-best_w)*oof_cat, y.values):.1f}")

    oof_blend = best_w * oof_lgb + (1 - best_w) * oof_cat

    log("[4/6] OOF 예측 기반 확률 보정(Isotonic Regression) 학습...")
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calibrator.fit(oof_blend, y.values)
    oof_calibrated = calibrator.predict(oof_blend)
    log(f"  └ 보정 후 OOF Brier={brier_score(oof_calibrated, y.values):.6f} | "
        f"보정 후 OOF BSS={bss_score(oof_calibrated, y.values):.1f}")

    log("[5/6] 아티팩트 저장...")
    clip_eps = 1e-3
    meta = {
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'cat_mappings': cat_mappings,
        'best_w_lgb': float(best_w),
        'best_w_cat': float(1 - best_w),
        'clip_eps': clip_eps,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, 'meta.pkl'))
    joblib.dump(calibrator, os.path.join(MODEL_DIR, 'calibrator.pkl'))

    log(f"[6/6] 완료! 최종 OOF BSS(캘리브레이션 적용) = {bss_score(np.clip(oof_calibrated, clip_eps, 1-clip_eps), y.values):.1f}")


if __name__ == '__main__':
    main()
