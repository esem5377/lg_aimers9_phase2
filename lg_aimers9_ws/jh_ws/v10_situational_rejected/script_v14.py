import os
import glob
import joblib
import numpy as np
import pandas as pd


def load_model():
    lgb_models = [joblib.load(p) for p in sorted(glob.glob(os.path.join('model', 'lgb_fold*.pkl')))]
    cat_models = [joblib.load(p) for p in sorted(glob.glob(os.path.join('model', 'cat_fold*.pkl')))]
    xgb_models = [joblib.load(p) for p in sorted(glob.glob(os.path.join('model', 'xgb_fold*.pkl')))]
    meta_info = joblib.load(os.path.join('model', 'meta_info.pkl'))
    return {'lgb': lgb_models, 'cat': cat_models, 'xgb': xgb_models, 'meta': meta_info}


def load_data():
    return pd.read_csv(os.path.join('data', 'test.csv'))


def create_features(df, meta_info):
    data = df.copy()

    sit_cols = meta_info['sit_cols']
    hand_map = meta_info['hand_map']
    fallback = meta_info['sit_global_fallback']
    sit_stats = pd.DataFrame(meta_info['sit_stats'])

    data['pitcher_hand'] = data['pitcher_hand'].astype(str)
    data['batter_hand'] = data['batter_hand'].astype(str)
    data = data.merge(sit_stats, on=sit_cols, how='left')
    data['sit_fastball_rate'] = data['sit_fastball_rate'].fillna(fallback['sit_fastball_rate'])
    data['sit_breaking_rate'] = data['sit_breaking_rate'].fillna(fallback['sit_breaking_rate'])
    data['sit_offspeed_rate'] = data['sit_offspeed_rate'].fillna(fallback['sit_offspeed_rate'])
    data['sit_n'] = data['sit_n'].fillna(0)

    if 'balls_before' in data.columns and 'strikes_before' in data.columns:
        data['cnt_diff'] = data['strikes_before'] - data['balls_before']
        data['is_strike_pressured'] = (data['balls_before'] == 3).astype(int)
        data['is_two_strike'] = (data['strikes_before'] == 2).astype(int)

    if 'li' in data.columns and 'asof_pitcher_middle_rate' in data.columns:
        data['leverage_middle_risk'] = data['li'] * data['asof_pitcher_middle_rate']

    return data


def predict(models_dict, data):
    meta_info = models_dict['meta']
    X_test = create_features(data, meta_info)

    selected_features = meta_info['features']
    cat_mappings = meta_info['cat_mappings']
    opt_w = meta_info['opt_weights']

    for c in selected_features:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test_feat = X_test[selected_features].copy()

    for col, cat_map in cat_mappings.items():
        if col in X_test_feat.columns:
            vals = [str(v) if pd.notna(v) else 'missing' for v in X_test_feat[col]]
            X_test_feat[col] = [cat_map.get(v, -1) for v in vals]

    lgb_preds = np.mean([m.predict_proba(X_test_feat)[:, 1] for m in models_dict['lgb']], axis=0)
    cat_preds = np.mean([m.predict_proba(X_test_feat)[:, 1] for m in models_dict['cat']], axis=0)
    xgb_preds = np.mean([m.predict_proba(X_test_feat)[:, 1] for m in models_dict['xgb']], axis=0)

    final_preds = (lgb_preds * opt_w['w_lgb']) + (cat_preds * opt_w['w_cat']) + (xgb_preds * opt_w['w_xgb'])
    final_preds = np.clip(final_preds, 0.005, 0.995)
    return final_preds


def save_results(test_df, predictions):
    os.makedirs('output', exist_ok=True)
    id_col = [c for c in test_df.columns if c.lower() in ['row_id', 'id']]
    id_col = id_col[0] if id_col else test_df.columns[0]

    submission = pd.DataFrame({
        id_col: test_df[id_col],
        'control_success': predictions
    })

    if submission['control_success'].isnull().sum() > 0:
        submission['control_success'] = submission['control_success'].fillna(0.5)

    submission.to_csv('output/submission.csv', index=False)


if __name__ == "__main__":
    models = load_model()
    test_data = load_data()
    preds = predict(models, test_data)
    save_results(test_data, preds)
    print("추론 완료! output/submission.csv 생성 성공")
