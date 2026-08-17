# 진행 기록 — 2026-08-17

> jh_ws의 v8(그룹 k-fold, LGB+CAT+XGB 3모델 앙상블)과는 별개로 진행한 실험 트랙입니다.
> `model/cat_models.pkl`은 454MB로 GitHub 업로드 제한(100MB)을 초과해 이 저장소에는 포함하지 않았습니다.
> 재현하려면 `train.py`를 다시 돌리면 됩니다. `submit.zip`/`submit_fixed.zip`(각 165MB)도 같은 이유로 제외했습니다.

## 1. 대회 개요
- 과제: KBO 투구 데이터 기반 "제구 성공(`control_success`)" 확률 예측 (이진 분류)
- `train.csv`: 2019~2024시즌, 1,475,092행 × 49컬럼
- `test.csv`: **2025시즌**(미래 시점). 배포본에는 형식 확인용 5건만 포함, 실제 평가 시 서버에서 비공개 데이터로 교체됨
- 평가지표: Brier Skill Score(BSS), ×100000 스케일
- `trackman_history.csv`: 2019~2024년 Trackman 로그(1,793,078행 × 30컬럼) — 아래 4번에서 사용 불가 확인

## 2. 만든 모델
- **구조**: LightGBM + CatBoost 5-Fold(StratifiedKFold, 랜덤 셔플) 앙상블
- 각 fold 학습 시 early stopping용 검증셋은 OOF fold와 별도로 train 쪽에서만 carve-out(이중사용 방지)
- OOF 예측 기준으로 Brier score 최소화하는 블렌딩 가중치 탐색 → **LGB 0.08 : CAT 0.92**
- Isotonic Regression으로 최종 확률 보정(calibration)
- **OOF BSS = 2212.3**
- 산출물: `model/lgb_models.pkl`, `model/cat_models.pkl`, `model/calibrator.pkl`, `model/meta.pkl`
- 코드: `train.py`(학습), `script.py`(추론 및 제출 파일 생성)

## 3. 제출 zip 버그 — 발견 및 수정 완료
- **증상**: 제출 후 평가 서버에서 `FileNotFoundError: model/lgb_models.pkl` 발생
- **원인**: `submit.zip`의 zip 엔트리가 `model\lgb_models.pkl`처럼 **백슬래시** 경로로 기록되어 있었음. zip 표준은 `/`를 구분자로 써야 하는데, 리눅스 평가 서버(`/app`)에서 압축을 풀면 `model/` 디렉토리가 생성되지 않고 파일명 자체에 백슬래시가 포함된 이상한 단일 파일이 생성됨
- **조치**: forward-slash 경로로 재압축한 `submit_fixed.zip` 생성, 실제로 압축을 풀어서 `model/` 하위에 4개 pkl 파일이 정상적으로 들어가는 것까지 검증 완료

## 4. 실제 리더보드 점수 격차 진단
- `submit_fixed.zip` 제출 후 실제 점수 **658** (OOF 2212.3 대비 대폭 낮음)
- **원인 ①**: train은 2019~2024만 포함, test는 2025(완전 미래 시즌). 학습 때 쓴 랜덤 StratifiedKFold는 이 시간적 분포 이동(temporal shift)을 전혀 시뮬레이션하지 못해 OOF가 실제 성능을 과대평가함
- **원인 ②**: `pitcher_id`/`batter_id`를 고카디널리티 범주형 피처로 사용. train의 `pitcher_id` 최댓값은 24633인데, test 샘플(5건 중)의 신인 선수 ID가 이 범위를 벗어나 "unknown" 코드로 처리됨(표본 기준 약 40%). OOF는 랜덤 셔플이라 거의 모든 선수가 이미 train fold에 존재해 이 문제가 드러나지 않았음
- **부가 요인**: BSS 지표 자체가 baseline 대비 아주 작은 오차 차이에도 극도로 민감한 스케일이라(OOF Brier 0.2439 → 실측 추정 0.2477, 차이 0.0038인데 BSS는 2212→658로 70% 하락) 체감 낙폭이 더 커 보임

## 5. trackman_history.csv 조인 불가 — 실제 테스트로 확인
- `pitcher_id`(train, 20700~24633, 792명) vs `pitcher_trackman_id`(trackman_history, 50008~71,775,155, 906명): **직접 겹치는 값 0개**
- 투수별 집계 후 `pitcher_id` 기준 Left Join을 실제로 실행 → **147.5만행 중 매칭 0건(0.0000%)**
- 팀 코드도 한쪽은 익명 정수(train `pitcher_team_id`: 12~25), 한쪽은 실제 KBO 구단 약어 문자열(trackman `pitcher_team`: `'KIA_TIG'` 등 26종, 팀 개편 이력 포함)이라 크로스워크 복원도 신뢰 불가
- **결론**: `trackman_history.csv`는 선수 단위 피처로 사용 불가능. 데이터 설명서에도 "1:1로 직접 결합되는 테이블이 아니다"라고 명시돼 있음

## 6. 다음 단계 (미결정, 재개 시 먼저 결정할 것)
재학습 방향 두 가지 옵션 제시했으나 아직 선택 안 함:
- **(a) 권장**: 시간 기반 검증(2019~2023 학습 / 2024 검증)으로 전환 + `pitcher_id`/`batter_id` 등 ID 피처 비중 축소 또는 제외 — 실제 미래 시즌 일반화 성능을 제대로 추정하고, 신인 선수 대응력을 높임
- **(b)**: 검증 방식만 시간 기반으로 바꾸고 ID 피처는 유지 — 구조 변경이 적어 빠르게 확인 가능

다음 세션에서 이어서 진행할 때는 이 옵션 선택부터 시작.
