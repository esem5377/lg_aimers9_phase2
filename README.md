# LG Aimers 9 해커톤 - 투구 제구 성공 확률 예측

투구 직전까지 알 수 있는 경기 상황 / 선수 정보 / 과거 이력 피처로 해당 투구의
제구 성공 확률(`control_success`)을 예측하는 과제.

## 폴더 구조

```
lg_aimers9_ws/
  open/
    data_description.md      # 데이터 설명서 (컬럼 정의, 제출 규칙 등 필독)
    competition_rules.md     # 대회 규칙 (사전학습모델/외부 API/외부 데이터 사용 제한 등)
    data/                     # 원본 데이터 (.gitignore 처리, 각자 로컬에 준비 필요)
      train.csv               # 학습 데이터 2019~2024, 1,475,092행 x 49컬럼
      test.csv                 # 형식 확인용 5행 샘플 (실제 평가는 서버가 2025시즌 데이터로 교체)
      sample_submission.csv
      trackman_history.csv    # 2019~2024 Trackman 투구 실측 로그 (아직 미사용)
    baseline_submit/          # 주최측 제공 베이스라인 예시 (RandomForest)

  work/
    pipeline/                 # 현재 운영 중인 학습/추론 파이프라인
      build_trackman_context.py # trackman_history.csv → 상황(카운트/투타/이닝) 단위 통계 룩업 테이블 생성
      train.py                  # LightGBM 학습 (시간 기반 검증: 2019~2023 학습 / 2024 검증)
      train_catboost.py         # CatBoost 학습 (동일 피처/검증, 현재 채택 모델)
      predict.py                 # 로컬 추론/검증용 (LightGBM)

    experiments/               # 튜닝/피처 실험 스크립트 (채택 여부 무관, 전부 참고용 보존)
      eda.py                     # 기초 EDA
      tune.py                    # LightGBM 하이퍼파라미터 랜덤서치 (개선 미미, 기각)
      tune_coldstart.py          # asof_* 콜드스타트 베이지안 스무딩 (효과 없음, 기각)
      tune_v2.py                 # trackman 교차그룹 + LGB/CatBoost 앙상블 비교 (CatBoost 채택 계기)
      build_team_history.py      # 팀 단위 as-of 이력 피처 (효과 없음, 기각)
      tune_catboost.py           # CatBoost 하이퍼파라미터 랜덤서치 (v4에 반영)
      tune_catboost_rawid.py     # CatBoost에 raw pitcher_id/batter_id 재도입 (과적합, 기각)
      tune_blend.py              # LightGBM+CatBoost 블렌드 가중치 정밀 탐색 (노이즈 수준, 기각)
      tune_cross2.py             # 저카디널리티 trackman 교차 피처 3종 (로컬 개선, 리더보드 하락으로 기각)

    submissions/                # 제출 패키지 원본 및 zip
      submit/                    # LightGBM 제출 패키지 원본 (822점)
      submit_catboost/           # CatBoost 제출 패키지 원본 (현재 최선 버전 소스)
      submit_v2.zip               # LightGBM 제출 zip (822점)
      submit_v3_catboost.zip      # CatBoost 제출 zip (875점)
      submit_v4_catboost.zip      # CatBoost + 튜닝 제출 zip (889점, 현재 최선)

    model/                     # LightGBM 모델 (lgbm.pkl, feature_meta.pkl, trackman_context.pkl)
    model_catboost/            # CatBoost 모델 (catboost.cbm, feature_meta.json)

  worklog/
    2026-08-15.md             # 작업 기록 (~875점까지)
    2026-08-16.md             # 작업 기록 (889점 반영, 이후 실험 기각)
```

## 데이터 준비

`lg_aimers9_ws/open/data/*.csv`는 용량이 커서(train 352MB, trackman_history 338MB)
레포에는 포함되어 있지 않습니다. 대회 배포본에서 받아 `lg_aimers9_ws/open/data/`
아래에 그대로 넣어주세요.

## 환경 세팅

```bash
pip3 install --user --break-system-packages pandas scikit-learn lightgbm joblib catboost
```

`python3 -m venv`가 이 환경에서 sudo 없이 정상 동작하지 않아 `--user
--break-system-packages`로 설치함. 다른 환경이면 그냥 venv 써도 무방.

## 사용법

```bash
# trackman_history 기반 상황 단위 통계 피처 생성 (최초 1회, 또는 trackman_history.csv 갱신 시)
python3 lg_aimers9_ws/work/pipeline/build_trackman_context.py

# 학습 (train.csv 필요, model/trackman_context.pkl 필요)
python3 lg_aimers9_ws/work/pipeline/train.py

# 로컬 추론 확인 (test.csv 5건)
python3 lg_aimers9_ws/work/pipeline/predict.py

# CatBoost 학습 (현재 최선 버전, model/trackman_context.pkl 필요)
python3 lg_aimers9_ws/work/pipeline/train_catboost.py

# 제출 zip 재생성 (CatBoost, 현재 최선 — 학습 후 model_catboost/*를 submissions/submit_catboost/model/에 복사한 뒤)
cd lg_aimers9_ws/work/submissions/submit_catboost && zip -r ../submit_v4_catboost.zip script.py model requirements.txt

# 제출 zip 재생성 (LightGBM, 이전 버전)
cd lg_aimers9_ws/work/submissions/submit && zip -r ../submit_v2.zip script.py model requirements.txt
```

## 제출 규칙 요약 (자세한 건 data_description.md, competition_rules.md)

- 평가 데이터는 행 단위 독립 예측만 허용 — test.csv 내부 행을 이용한 집계/타겟인코딩/롤링 피처 금지.
- 현재 투구의 사후 정보(실제 코스, 판정, 구종, Trackman 실측값), 2025년 Trackman 데이터 사용 금지.
- 제공된 `asof_*` 컬럼(사전 계산된 과거 이력 피처)은 사용 가능.
