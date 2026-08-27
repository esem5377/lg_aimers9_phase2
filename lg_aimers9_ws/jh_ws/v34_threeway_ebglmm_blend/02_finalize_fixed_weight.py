"""
그리드서치 결과 best_w_ebglmm=0.00(=v26과 동일)이 나왔지만, 사용자가 로컬 신호가
약해도 실제 리더보드에서 직접 확인하고 싶어해 w_ebglmm을 작게 고정(0.05)해 v34
제출 패키지를 완성. 01_build_v34.py가 저장해둔 raw 예측 캐시를 재사용(재계산 없음).
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

V34_DIR = os.path.dirname(__file__)
V34_MODEL_DIR = os.path.join(V34_DIR, "model")
V34_OUT_DIR = os.path.join(V34_DIR, "output")
W_CATBOOST_RETRIEVAL = 0.7
W_EBGLMM_FIXED = 0.05


def fit_platt_scaling(raw_p, y):
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(raw_p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt_scaling(raw_p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw_p) + b)))


def bss_score(p, y):
    r = np.asarray(y).mean()
    baseline = r * (1 - r)
    bs = np.mean((np.asarray(p) - np.asarray(y)) ** 2)
    return max(0.0, 100000 * (1 - bs / baseline))


def main():
    cb_calib_raw = np.load(os.path.join(V34_OUT_DIR, "cb_calib_raw.npy"))
    nca_calib_raw = np.load(os.path.join(V34_OUT_DIR, "nca_calib_raw.npy"))
    eb_calib_raw = np.load(os.path.join(V34_OUT_DIR, "eb_calib_raw.npy"))
    y_calib = np.load(os.path.join(V34_OUT_DIR, "y_calib.npy"))

    base_calib_raw = W_CATBOOST_RETRIEVAL * cb_calib_raw + (1 - W_CATBOOST_RETRIEVAL) * nca_calib_raw
    blend_calib_raw = (1 - W_EBGLMM_FIXED) * base_calib_raw + W_EBGLMM_FIXED * eb_calib_raw
    a, b = fit_platt_scaling(blend_calib_raw, y_calib)
    calib_bss = bss_score(apply_platt_scaling(blend_calib_raw, a, b), y_calib)
    ref_bss = bss_score(apply_platt_scaling(base_calib_raw, *fit_platt_scaling(base_calib_raw, y_calib)), y_calib)
    print(f"w_ebglmm={W_EBGLMM_FIXED} 고정: calib_bss={calib_bss:.2f}  a={a}  b={b}")
    print(f"참고(w_ebglmm=0, 순수 CatBoost+retrieval 0.7:0.3): calib_bss={ref_bss:.2f}")
    print(f"delta(calib 기준)={calib_bss - ref_bss:+.2f}")

    with open(os.path.join(V34_MODEL_DIR, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["blend_weight_ebglmm"] = W_EBGLMM_FIXED
    meta["calibration"] = {"method": "platt_sigmoid", "a": a, "b": b}
    with open(os.path.join(V34_MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nfeature_meta.json 업데이트 완료: blend_weight_ebglmm={W_EBGLMM_FIXED}, calibration a/b 갱신")

    with open(os.path.join(V34_OUT_DIR, "metrics_v34_final.json"), "w", encoding="utf-8") as f:
        json.dump({
            "w_catboost_retrieval": W_CATBOOST_RETRIEVAL,
            "w_ebglmm_fixed": W_EBGLMM_FIXED,
            "calib_bss": calib_bss,
            "calib_bss_reference_no_ebglmm": ref_bss,
            "calib_delta": calib_bss - ref_bss,
            "calibration_a": a, "calibration_b": b,
            "note": "grid search best was w_ebglmm=0.00 (identical to v26); "
                    "0.05 chosen deliberately by user to test on real leaderboard "
                    "despite weak/negative local signal.",
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
