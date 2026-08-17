# v9 — 시간 기반 검증 도입, 리더보드 924점 (2026-08-17)

v8(`StratifiedGroupKFold(groups=pitcher_id)`, 3모델 앙상블, 리더보드 909점) 기반으로
CV 방식만 시간 기반 검증으로 바꾼 버전. **실제 리더보드 924점**으로 v8 갱신.

## 문제의식

test.csv는 2025시즌(완전 미래)인데, v8까지 쓰던 `StratifiedGroupKFold`는 "같은
투수가 train/val에 안 겹치게" 만들 뿐, "2019~2024로 학습해서 2025를 맞혀야 하는
시간적 분포 이동(temporal shift)"은 전혀 시뮬레이션하지 못한다.

## 변경 내용

- `train.py`: CV를 `season<=2023` 학습 / `season==2024` 검증인 단일 시간 분할로 교체.
  블렌드 가중치 탐색(Optuna)도 val 구간 기준으로만 평가하도록 수정
  (원래 코드 그대로 시간분할만 넣으면, oof 배열이 val 구간에만 채워지는데 나머지
  80%가 예측값 0으로 잡혀 최적화가 깨지는 버그가 생겨서 같이 고쳤음).
- 결과: **2024 홀드아웃 BSS = 738.0** (LightGBM 689.4 / CatBoost 737.5 / XGBoost 665.4,
  최적 블렌드 가중치 LGBM 0.086 : CatBoost 0.914 : **XGBoost 0.000**)
  - 랜덤 K-Fold였던 초기 실험(OOF 2212, 실제 658점)보다 훨씬 정직하고,
    GroupKFold(909점)보다도 실제로 나은 신호였음.
- `train_final.py`: 위에서 확정한 구조/가중치를 그대로 고정한 채, 실제 제출 모델은
  **2019~2024 전체 데이터**로 재학습 (early stopping/calibration용으로만 5%
  carve-out, 블렌드 가중치는 재탐색하지 않고 위 값 고정).
- `build_submit.py`: `submit.zip` 패키징 스크립트. zip 엔트리를 forward-slash
  경로로 강제 (과거 다른 실험에서 백슬래시 경로 때문에 평가 서버 압축 해제 실패한
  적 있어서 재발 방지).

## 결과

전체 데이터 재학습 후 제출 → **리더보드 924점** (v8의 909점 대비 +15).

## 다음 후보

- XGBoost는 가중치 0으로 사실상 기여가 없음 — 제거하면 학습 시간 단축 가능.
- v8과 마찬가지로 `pitcher_id` 기준 trackman merge는 겹치는 값이 0개라 죽은 코드
  (`release_dev` 피처가 항상 결측). 정리 여지 있음.
- 상황(카운트/레버리지) 기반 구종 확률 같은 trackman population-level 피처는
  아직 시도 안 함.
