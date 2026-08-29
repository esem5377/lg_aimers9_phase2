# v40: CatBoost에서 pitcher_id/batter_id를 native categorical로 전환 — 931점, 기각 (역대 최악 회귀)

## 배경
- 2026-08-29: "완전 새로운 방향 탐색" 요청. 조사 결과 CatBoost `max_ctr_complexity`를
  그냥 올리는 건 es_ws가 2026-08-16에 이미 시도해 완전 무효 확인(기존 7개 저카디널리티
  범주형 컬럼만으로는 기본값에서 이미 포화, worklog/2026-08-16.md L38).
- 그런데 그 실험은 `RAW_ID_COLS`(pitcher_id=792종/batter_id=830종, 고카디널리티)를
  건드리지 않았음 -- 프로젝트 전체 이력에서 이 두 컬럼은 항상 label-encoded 정수로
  CatBoost에 **numeric** 피처로만 들어갔고, CatBoost의 leak-safe ordered CTR 조합
  탐색 범위(`cat_features`)에 포함된 적이 없었음. 이번 실험은 이 두 컬럼을
  `cat_features`에 추가해 CatBoost 자체의 카운터 기반 prior 스무딩 + ordered boosting
  (타겟 누출 방지)으로 pitcher×batter/pitcher×situational 조합을 자동 학습하게 함.
- retrieval encoder는 v26 것을 그대로 재사용(재학습 없음), EB-GLMM은 v34와 동일 방식
  (last1season)으로 재학습. **사용자가 fold0/fold2 walk-forward 검증 없이 바로 실제
  제출까지 진행하기로 명시적으로 결정**(lg_aimers9_walkforward_methodology 규칙 우회).

## 결과
- carve-out(랜덤 5%, 7.4만행): calibrated BSS **2268.86** (w_ebglmm=0.00으로 수렴 --
  EB-GLMM이 모델링하던 pitcher/batter 랜덤효과를 CatBoost가 직접 흡수한 것으로 해석)
  — v34 기준값 2082.78 대비 **+186.08**, 이 프로젝트 역대 최대 carve-out delta
  (직전 기록은 v37의 +40, 4.6배 차이).
- CatBoost 모델 파일 크기: 8.4MB(baseline, numeric id) → **212.8MB**(treatment, 25배)
  -- pitcher×batter 조합 CTR 통계 테이블이 그만큼 방대해졌다는 신호.
- **실제 리더보드: 931점**
- v34(현재 팀 최고, 1024점) 대비 **-93점** — 이 프로젝트 역대 최악의 실제 리더보드
  회귀(직전 최악은 v36의 -20점).

## 결론 — 이 프로젝트 방법론 규칙에 대한 가장 강력한 반증 사례
- **carve-out은 랜덤 5% 분할이지 시간 분할이 아님.** pitcher_id/batter_id를 범주형으로
  바꾸면 같은 투수/타자의 다른 행이 train_sub와 calib에 동시에 섞여 들어갈 수 있고,
  CatBoost의 ordered CTR이 사실상 "이 투수/타자 조합의 과거 평균 성공률"을 거의
  암기하는 효과를 냈을 가능성이 높음 -- carve-out에서는 극적으로 좋아 보이지만 진짜
  미래 예측에는 전혀 재현되지 않음(오히려 대폭 악화).
- v36(-20)/v37(-3~-4)/v38b(-7)에 이어 **네 번째 연속 "walk-forward 우회 후 실제
  리더보드 역행" 사례**이자 지금까지 중 가장 큰 델타(+186)가 가장 큰 손해(-93)로
  이어진 사례. `lg_aimers9_walkforward_methodology` 메모리의 "carve-out delta가
  클수록 더 의심해야 한다"는 원칙을 가장 강하게 재확인함.
- **다음부터는 carve-out 신호가 아무리 커도(오히려 클수록 더 강하게) fold0/fold2
  walk-forward를 완주한 뒤에만 실제 제출하는 것을 예외 없는 기본값으로 삼아야 함.**
  `session_2026-08-29_walkforward_v40_id_as_cat/`에 검증 스크립트를 만들어뒀으나
  실제 제출을 먼저 진행하느라 완주하지 못함(체크포인트로 fold2 baseline만 재사용 준비된
  상태, fold0/treatment는 미실행) -- 재개 시 이어서 실행 가능.
- 현재 팀 최고는 **v34(1024점)로 변동 없음**. v40은 기각.
