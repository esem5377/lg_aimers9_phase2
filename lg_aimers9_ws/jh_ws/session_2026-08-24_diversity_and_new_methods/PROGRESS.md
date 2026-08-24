# 2026-08-24 세션 — "다양성 탐색" + "완전히 새로운 방법" 실험 기록

전부 로컬 스크리닝(fold0: season<=2021 학습/season==2022 검증, fold2: season<=2023
학습/season==2024 검증)에서 기각, 실제 제출은 v27/v28(별도 폴더)만 진행했음.
CatBoost 분류기는 이전 세션 캐시(`catboost_cache_fold0_2022.pkl`/
`catboost_cache_fold2_2024.pkl`, 987 baseline과 정확히 일치 확인됨)를 재사용해
재학습 생략.

## 1. rmse_blend_screening.py — CatBoostRegressor(RMSE) 블렌드
987 레시피 그대로 RMSE loss로 CatBoost 회귀 학습, 기존 분류기와 블렌드.
- 단독 성능은 분류기와 거의 동일(fold0 auc 0.5792 vs 0.5789, fold2 auc 0.5513 vs 0.5515)
- 블렌드 delta: fold0 **+4.79**, fold2 **±0.00**(회귀 기여 0%)
- 결론: 실력은 비슷한데 같은 트리 계열이라 오류 패턴 다양성이 부족, 기각.

## 2. bayes_shrinkage_screening.py — 베이지안 shrinkage(Beta-Binomial 부분 풀링)
`shrunk_rate=(n*rate+kappa*global_mean)/(n+kappa)`를 투수/타자 asof_n/success_rate에
적용, kappa 그리드서치 후 작은 로지스틱회귀로 결합.
- 단독 성능이 분류기보다 뚜렷이 낮음(fold0 auc 0.5653, fold2 auc 0.5333)
- 블렌드 delta: 두 폴드 다 **정확히 0.00**
- 결론: 피처 2개만 써서 정보량 자체가 CatBoost의 부분집합, 기각.

## 3. extratrees_blend_screening.py / _v2.py — ExtraTrees(배깅 계열) 블렌드
1차: OOB 기준 그리드서치(4조합) → `max_depth=None`(무정규화) 선택 → 파국적 결과
(fold0 delta -67.05, fold2 **-430.77**, 시즌경계 과적합).
2차(v2): 정규화 강화(depth 6/8/10) + eval_df 기준 직접 선택으로 수정 →
과적합은 해결됐으나 단독 성능이 여전히 CatBoost 못 미침(fold2 auc 0.5275,
거의 랜덤) → 블렌드 delta 두 폴드 다 **±0.00**.
- 핵심 교훈: OOB/calib_df는 train_sub와 같은 시기(랜덤 분할)라 시즌경계
  일반화 실패를 못 잡음 — 방법론 결함, 이후 실험(calibration_compare)에도
  똑같이 반복됨.

## 4. calibration_compare.py — Platt vs Isotonic calibration 비교 (production 스케일)
v26의 학습된 CatBoost/retrieval을 재학습 없이 재사용, calib_df(73,755행)를
80:20으로 쪼개 4가지 calibration 방식 비교.
- **D(모델별 개별 isotonic 후 블렌드)가 holdout 기준 +15.23으로 최고** → v28로
  실제 제출까지 진행했으나 **983점(v26 대비 -40)으로 하락**.
- 원인: 이 80:20 홀드아웃도 calib_df(2019~2024 전체 랜덤 5%)를 다시 랜덤
  분할한 것이라 시즌 경계를 안 넘음 — ExtraTrees 1차 시도와 동일한 방법론적
  함정. Isotonic(breakpoint 70~95개)이 Platt(파라미터 2개)보다 훨씬 유연해
  2019~2024 calibration curve를 과적합, 2025 실제 시즌엔 안 맞았을 가능성.
- **새 규칙**: calib_df를 다시 랜덤으로 쪼갠 홀드아웃은 시즌경계 일반화
  실패를 못 잡는 무효한 검증법 — 향후 반드시 fold0/fold2로 스크리닝할 것.

## 5. trend_baseline_screening.py / _v3_noseason.py — 타겟 디트렌딩 + 잔차 부스팅
EDA로 확인된 control_success 6년 연속 하락(-7.9%p)을 season 단독 로지스틱회귀로
분리해 CatBoost의 네이티브 `baseline`(Pool offset) 파라미터로 잔차만 학습.
- **1차 버그**: season 원본값(2019~2024)을 `LogisticRegression(C=1e10)`에
  그대로 넣어 lbfgs 수렴 실패(계수가 0에 수렴) — `season-2019` 중앙화로 수정
  (`trend_baseline_screening.log`=버그 버전 출력, `trend_baseline_screening_v2.log`
  =수정 후 재실행 출력, 스크립트 파일 자체는 수정된 최종본만 남아있음).
- 수정 후 결과: season을 입력 피처로 유지한 채 baseline만 추가 → 두 폴드 다
  나쁨(fold0 **-63.38**, fold2 **-64.65**), AUC는 거의 안 변함(calibration만 깨짐).
- 후속 검증(`_v3_noseason.py`, season을 baseline에서만 쓰고 입력 피처에서 제거):
  가설과 반대로 **훨씬 더 나쁨**(fold0 -247.26, fold2 **-603.39**, auc도
  0.5515→0.5393 붕괴) — season을 일반 피처로 남겨야 트리가 비선형 패턴
  (2023 game_type 레짐 변화 등)까지 학습 가능한데 그걸 뺏겨서 더 손해.
- 결론: 두 변형 다 기각, CatBoost의 end-to-end 학습이 이미 트렌드를 충분히
  잘 흡수하고 있었음 — baseline 분해가 오히려 유연성을 제한.

## 6. adversarial_validation.py — 불안정한 피처 전수조사 (진단 도구, 실험 아님)
season<=2023 vs season==2024를 구분하는 분류기로 "어떤 피처가 연도를 가장
쉽게 누설하는지" 스캔.
- **adversarial AUC = 0.9953** (거의 완벽하게 연도 구분 가능, 심각한 드리프트 확인)
- 상위 피처: pitcher_id(15.82) > asof_batter_n(10.56) > batter_id(9.71) >
  asof_batter_middle_rate(7.53) > game_month(7.11) > batter_team_id(6.36) >
  asof_pitcher_n(6.08) > asof_pitcher_pitchmix_n(5.61)
- 해석: 드리프트의 대부분이 "선수 정체성(로스터가 매년 바뀜)"과 "표본수
  컬럼(커리어가 쌓일수록 증가하는 시계 역할)"에서 옴 — 둘 다 모델의 핵심
  피처라 제거 불가(asof_*_n 제거는 8/23 v24에서 이미 실패 확인, -12).
  `control_risk_score`(이 프로젝트 유일한 성공 피처, +10)도 드리프트 랭킹에
  뜬다는 게 "불안정=제거해야 함"이 아니라는 걸 재확인시켜줌.
- 특이사항(미해결): `game_month`가 5위로 꽤 높음 — 월 값은 매년 반복되는데도
  연도 구분력이 있다는 건 2024년 KBO 일정 구조가 달랐을 가능성, 추가 조사 안 함.

## 종합 결론
이번 세션에 시도한 8개 축(RMSE 회귀/베이지안 shrinkage/ExtraTrees x2/
isotonic calibration/타겟 디트렌딩 x2/adversarial 진단) 전부 팀 최고
1023점(v26, `jh_ws/v26_retrieval_blend_w07`)을 넘지 못함. 실제 제출까지
간 건 v27(992)/v28(983) 둘 다 하락. 다음 세션 최우선 후보는 여전히
retrieval 다중 시드 확장(시드 배깅 + retrieval, 이미 검증된 두 성공
메커니즘의 결합).
