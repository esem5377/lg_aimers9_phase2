# LG Aimers 9 해커톤 - 투구 제구 성공 확률 예측

투구 직전까지 알 수 있는 경기 상황 / 선수 정보 / 과거 이력 피처로 해당 투구의
제구 성공 확률(`control_success`)을 예측하는 과제.

## 폴더 구조

```
lg_aimers9_ws/
  open/
    data_description.md      # 데이터 설명서 (컬럼 정의, 제출 규칙 등 필독)
    data/                     # 원본 데이터 (.gitignore 처리, 각자 로컬에 준비 필요)
      train.csv               # 학습 데이터 2019~2024, 1,475,092행 x 49컬럼
      test.csv                 # 형식 확인용 5행 샘플 (실제 평가는 서버가 2025시즌 데이터로 교체)
      sample_submission.csv
      trackman_history.csv    # 2019~2024 Trackman 투구 실측 로그 (아직 미사용)
    baseline_submit/          # 주최측 제공 베이스라인 예시 (RandomForest)

  work/
    eda.py                    # 기초 EDA
    build_trackman_context.py # trackman_history.csv → 상황(카운트/투타/이닝) 단위 통계 룩업 테이블 생성
    train.py                  # LightGBM 학습 (시간 기반 검증: 2019~2023 학습 / 2024 검증)
    predict.py                # 로컬 추론/검증용
    model/                    # 학습된 모델 (lgbm.pkl, feature_meta.pkl, trackman_context.pkl)
    submit/                   # 리더보드 제출용 패키지 원본 (script.py + model + requirements.txt)
    submit_v2.zip             # 제출용 zip (trackman 컨텍스트 피처 + ID 컬럼 제거 반영 버전)

  worklog/
    2026-08-15.md             # 작업 기록
```

## 데이터 준비

`lg_aimers9_ws/open/data/*.csv`는 용량이 커서(train 352MB, trackman_history 338MB)
레포에는 포함되어 있지 않습니다. 대회 배포본에서 받아 `lg_aimers9_ws/open/data/`
아래에 그대로 넣어주세요.

## 환경 세팅

```bash
pip3 install --user --break-system-packages pandas scikit-learn lightgbm joblib
```

`python3 -m venv`가 이 환경에서 sudo 없이 정상 동작하지 않아 `--user
--break-system-packages`로 설치함. 다른 환경이면 그냥 venv 써도 무방.

## 사용법

```bash
# trackman_history 기반 상황 단위 통계 피처 생성 (최초 1회, 또는 trackman_history.csv 갱신 시)
python3 lg_aimers9_ws/work/build_trackman_context.py

# 학습 (train.csv 필요, model/trackman_context.pkl 필요)
python3 lg_aimers9_ws/work/train.py

# 로컬 추론 확인 (test.csv 5건)
python3 lg_aimers9_ws/work/predict.py

# 제출 zip 재생성
cd lg_aimers9_ws/work/submit && zip -r ../submit_v2.zip script.py model requirements.txt
```

## 제출 규칙 요약 (자세한 건 data_description.md)

- 평가 데이터는 행 단위 독립 예측만 허용 — test.csv 내부 행을 이용한 집계/타겟인코딩/롤링 피처 금지.
- 현재 투구의 사후 정보(실제 코스, 판정, 구종, Trackman 실측값), 2025년 Trackman 데이터 사용 금지.
- 제공된 `asof_*` 컬럼(사전 계산된 과거 이력 피처)은 사용 가능.
