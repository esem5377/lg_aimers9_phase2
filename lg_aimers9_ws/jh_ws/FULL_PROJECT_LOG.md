# LG Aimers 9 Phase 2 — 제구 성공 확률 예측 프로젝트 전체 로그

이 문서는 2026-08-16(시작) ~ 2026-08-23까지 진행된 전체 실험 이력을 시간순으로 정리한 종합 기록이다. 세션별 개별 worklog(`worklog_*.md`)의 요약이자, 팀 전체(jh_ws + es_ws) 리더보드 이력과 이 프로젝트에서 반복 확인된 방법론적 교훈을 한 곳에 모은 것.

## 과제 개요

- **목표**: 투구 직전 상황(카운트, 주자, 점수차, 선수/팀 정보, 과거 이력 피처)을 바탕으로 해당 투구의 제구 성공 확률(`control_success`)을 예측.
- **평가지표**: Brier Skill Score — `Score = max(0, 100000 × (1 − BrierScore / 평균제구율BrierScore))`.
- **수료 기준선**: Public Score 549.51 이상(baseline RandomForest 코드 기준).
- **제약**: 외부 데이터/API 금지, row-wise 독립 추론 원칙(`test.csv` 다른 행 참조 금지), 비상업 라이선스 사전학습 모델만 허용, 완전 오프라인 추론 환경.
- **데이터**: `train.csv`(1,475,092행 x 49컬럼), `trackman_history.csv`(2019~2024 과거 로그, train/test와 직접 결합 불가 — `pitcher_id`/`batter_id`와 `pitcher_trackman_id`/`batter_trackman_id`의 overlap이 0임을 확인).
- **협업 구조**: GitHub 저장소 `esem5377/lg_aimers9_phase2`, 두 워크스페이스(`jh_ws`=사용자, `es_ws`=팀원 eunsoo)가 독립적으로 실험하고 서로 pull하며 발전시킴.

## 팀 전체 리더보드 최고점 갱신 이력

| 날짜 | 점수 | 주체/버전 | 핵심 변화 |
|---|---|---|---|
| ~8/17 | 909 | jh_ws | 3모델(LGB+CatBoost+XGBoost) 앙상블, StratifiedGroupKFold |
| 8/17 | 924 | jh_ws v9 | GroupKFold → 시간 기반(season) 검증으로 교체 |
| 8/18 | 959 | es_ws v6 | CatBoost 단일 + sigmoid(Platt) 확률보정 추가 |
| 8/18 | 974 | es_ws v7 | + raw pitcher_id/batter_id(label-encoded) 재도입 |
| 8/19 | 974.9 | es_ws v7 | 정밀값 확정(이후 여러 실험이 이 값을 못 넘고 버팀) |
| 8/21 | 979 | jh_ws v18 | 974 레시피를 시드 3개(42/7/123)로 배깅(순수 분산감소) |
| 8/21 | 982 | jh_ws v18 | 시드 6개(+1/99/777)로 확장 |
| 8/22 | 986 | es_ws v9 | CatBoost 6시드(v18 재사용) + LGBM/XGBoost 3시드 아키텍처 블렌드 |
| 8/22 | **992** | jh_ws v20 | control_risk_score 피처(기존 asof_* 재조합) 추가 |
| 8/22 | 993 | es_ws v11 | 992 레시피 + LGBM 3시드 블렌드(XGBoost 가중치 0이라 제외) |

**현재 팀 전체 최고: 993점**(es_ws v11, `submit_v11_riskscore_blend`).

## 924 → 974.9 격차 원인 규명 (8/18)

jh_ws(924)와 es_ws(974.9)가 같은 날 갈라진 원인을 통제 실험으로 규명:
- **트랙맨 상황 컨텍스트(기여 약 80%)**: es_ws는 `trackman_history.csv`를 `count_state`(balls/strikes/outs)/`hand_matchup`(투타손)/`inning_state`(이닝/초말) 3개 독립 저차원 groupby로 집계해 구종비율+물리량(구속/스핀/무브먼트) 평균을 상황 키로 병합. jh_ws는 `pitcher_trackman_id` 직접 조인을 시도했으나 overlap=0이라 사실상 죽은 코드였음. 트랙맨 유무만 통제한 ablation에서 delta +40.0(보정후 기준) 확인.
- **categorical 네이티브 인코딩 + raw id + 확률보정**: 나머지 격차 설명.
- **핵심 원칙 확립**: 로컬 검증 시 AUC와 BSS(실제 채점 지표)가 반대로 움직이는 경우가 있어 반드시 BSS로 확인해야 함. 확률보정(Platt)은 AUC엔 안 잡히는 신호를 증폭시킴(로컬 +24.8 → 실제 +70).

## [핵심 발견] 캘리브레이션 정답 유출 버그 (8/18)

기존 로컬 검증 스크립트 전체가 `calib.fit(X_val, y_val)`로 **평가 대상 시즌의 진짜 정답으로 확률보정**한 뒤 같은 시즌으로 채점하는 구조적 leak을 갖고 있었음. 학습기간 내부 carve-out으로만 보정하는 정직한 방식으로 재검증하니 델타가 대부분 사라짐(예: platoon+risp 델타 +23.3 → +5.7로 4분의 3 축소). **이후 모든 실험은 "calibration은 반드시 학습기간 내부 carve-out에서만 fit, 평가 대상 구간 정답은 절대 안 봄" 원칙을 지킴.**

## "로컬 개선 → 실제 하락" 반복 패턴 (8/17~8/21)

이 프로젝트에서 가장 중요하게 반복 확인된 현상. 로컬 season 단일 홀드아웃(주로 2024)에서 개선처럼 보였던 피처가 실제 리더보드에서는 반증되는 사례가 계속 나옴:

| 실험 | 로컬 delta | 실제 delta |
|---|---|---|
| 상황 기반 구종 확률(sit_*, jh_ws) | +13.8 | **-18**(924→906) |
| CatBoost 네이티브 categorical | +7.2 | **-9**(924→915) |
| platoon(투타상성)+risp(득점권) | +36.6(3모델 블렌드 기준) | **-73**(924→851, 이 프로젝트 최악) |
| trackman METRIC_COLS 4→8 확장(v15) | +20.81 | **-14.9**(974.9→960) |
| game_month 4번째 축 추가(v17) | +18.39 | **-18.9**(974.9→956) |
| freq974(raw id 학습셋 빈도) | +3.23/+5.59(fold0/fold2 이중 통과) | **-19**(974→955) |

**원인 가설(확정)**: `control_success` 전체 평균이 2019년 0.5647→2024년 0.4861로 6년 연속 단조 하락하고, `game_type`(F/R)과 target의 관계가 2023년부터 역전되는 등 **연도별 레짐 변화**가 실재함. 새로운 정보를 추가하는 피처는 학습 데이터(2019~2024) 안의 특정 시기 패턴에 과적합되기 쉽고, 그 패턴이 실제 평가 시즌(2025, 미래)에는 다를 수 있음. `season==2024` 단일 홀드아웃을 반복 재사용한 선택편향 가설은 별도 실험(SEALED GATE 설계)으로 **반박됨** — 완전히 격리된 검증 설계로도 851점 실패를 사전에 감지하지 못했으므로, 진짜 원인은 선택편향이 아니라 학습기간과 평가 시즌 사이의 실제 분포 변화로 결론.

**예외(신뢰 가능했던 축)**: 확률보정(+24.8→+70), raw id(+9.48→+15), 시드 배깅(+4.34/+3.98→+4.1, 이후 +3.3→+3), control_risk_score(-3.74/+7.04→+10) — 공통점은 **"기존 정보의 재구성/분산감소"**이지 "새로운 시즌별 패턴을 학습하는 피처"가 아니었다는 것.

## 시드 배깅 — 순수 분산감소가 처음으로 로컬-실제 정확히 일치한 사례 (8/21)

974 레시피를 완전히 동일하게 두고 랜덤시드만 3개(42/7/123)로 학습해 raw 확률을 평균 → Platt 1회 fit. 새 정보를 전혀 추가하지 않는 가장 보수적인 앙상블.
- 로컬(fold0/fold2 walk-forward) delta: fold0 +4.34, fold2 +3.98.
- **실제 리더보드: 974.9 → 979(+4.1)**, 로컬 예측과 거의 정확히 일치 — 이 프로젝트에서 크기까지 신뢰할 수 있었던 첫 사례.
- 시드 6개로 확장(3개 재사용+3개 신규 학습): carve-out 기준 +3.3, 실제 982(+3)로 재차 일치.
- **해석**: 시드 배깅은 새 정보를 추가하지 않고 순수하게 모델 분산만 줄이므로, 이 프로젝트에서 반복 실패했던 "새 정보의 계절별 과적합" 메커니즘이 원천적으로 성립하지 않음.

## 1150점 목표와 5대 "큰 시도" 전부 실패 (8/20~8/21)

팀 최고(당시 974.9~982)와 사용자가 확인한 목표 1150 사이 격차(당시 +175~+206)를 메우기 위해 시도한 5갈래, **전부 막힘**:

1. **DeepFM 임베딩**(pitcher/batter 16차원 임베딩을 CatBoost 피처로 추가): OOF AUC 0.5712→0.5657로 5-fold 전부 악화. 실제 제출 없이 기각.
2. **팀 매핑**(trackman의 실제 KBO 팀 코드 26개 ↔ train의 익명 team_id 13개): 모든 팀이 거의 동일한 정규시즌 일정을 따라 상관계수가 0.87~0.98에 뭉쳐있어 사실상 구분 불가 → 매칭 근거 없어 포기.
3. **trackman METRIC_COLS 확장**(extension/rel_height/rel_side/zone_speed 추가): 로컬 +20.81 → 실제 **-14.9**.
4. **game_month 4번째 독립 축 추가**: 로컬 +18.39 → 실제 **-18.9**.
5. **LGBM/XGBoost 앙상블 재검증**(트랙맨+raw id 포함된 강한 베이스 위): 그리드서치 최적 블렌드도 단일 최고 대비 +2.90뿐, 실제 제출 갈 가치 없다고 판단해 로컬에서 종료.

**결론(8/21 시점)**: "974.9~982가 이 레시피 계열(CatBoost 단일/앙상블 + situational trackman + raw id)의 현실적 상한"이라는 판단이 5개 독립 실패로 굳어짐.

## 8/21 세션 재개 — 추가 3개 신규 축(정규화/시드배깅/monotonic)

- **정규화 강화 가설**(depth=5, l2=50, bagging_temp=2.5, random_strength=5.0): fold0/fold2/GroupKFold **세 축 전부** -97~-168로 뚜렷하게 악화 → 명확히 기각(실제 제출 없이 로컬에서 걸러짐).
- **시드 배깅**: 위 "시드 배깅" 섹션 참고, 성공(974.9→979→982).
- **Monotonic constraints**(방향성 명확한 asof_* 8개에 단조 제약): fold0 -36.87/fold2 -15.41, 두 폴드 다 악화 → 기각. CatBoost 네이티브 처리가 수동 개입보다 나은 패턴이 여기서도 재확인됨.

## Optuna BSS 직접 최적화 하이퍼파라미터 탐색 (8/21)

기존 BEST_PARAMS가 "AUC 기준 랜덤서치" 결과였을 뿐 BSS를 직접 최적화한 적이 없었음을 발견 → 30 trial TPE 탐색(fold0/fold2 두 축). **30개 trial 전체에서 fold0/fold2를 동시에 개선한 조합이 단 하나도 없었음**(전부 트레이드오프) — 기존 BEST_PARAMS가 이미 fold0 기준 2위권으로 균형 잡힌 지점이었음이 확인됨. 기각.

## 8/22 세션 — control_risk_score 성공(992)과 그 여파

### control_risk_score 발견 (982 → 992, +10)

외부 가이드 문서 검토 중 발견한 유일하게 새로 검증된 아이디어. 기존 `asof_pitcher_reverse_rate`+`middle_rate`+`ball_rate`(전부 원래 피처로 존재)를 단순합/가중합으로 재조합:
```python
control_risk_score = asof_pitcher_reverse_rate + asof_pitcher_middle_rate + asof_pitcher_ball_rate
control_risk_score_weighted = 0.4*reverse + 0.3*middle + 0.3*ball
```
- 로컬(fold0/fold2, iterations=1000): fold0 **-3.74**, fold2 **+7.04** — 방향이 엇갈리는 애매한 신호.
- 982(6시드) 레시피에 그대로 얹어 6시드 재학습(iterations=2000) → **실제 리더보드 992점(+10)**. "소폭이고 방향이 갈리는 로컬 신호도 완전 기각하지 말고 실제 제출 후보로 남겨두자"는 방침이 처음으로 적중한 사례.

### 사후 분석 — 왜 성공했나

1. **연도별 안정성**: `control_success` target과의 상관계수를 연도별로 따로 구해보면, 세 원재료 중 가장 불안정한(`ball_rate`, range=0.062, 부호flip 있음) 것도 **셋을 합치면 range=0.025로 모든 개별 성분보다 더 안정**해짐(세 성분의 연도별 요동이 서로 다른 방향이라 상쇄).
2. **`platoon`(투타상성)과의 비교**: platoon도 "기존 컬럼 재조합"이지만 원재료(`pitcher_hand`/`batter_hand`, 연도간 변동폭 0.08~0.09)부터 이미 불안정했고, 재조합해도 불안정성이 줄지 않아 실제 -73으로 대실패. **"재조합=안전"이 아니라 "재조합이 원재료보다 더 안정해지는가"가 진짜 구분선**.
3. **원재료 유지 vs 제거**: control_risk_score는 원재료(reverse/middle/ball_rate)를 그대로 둔 채 파생 2개를 추가하는 방식(71→73피처)이었음. 원재료를 제거하고 파생만 남기면(70피처) fold0 -14.52/fold2 +8.87로 방향이 갈려 판단 보류 상태였으나, 실제 제출(v22, 1시드)로는 **987점** — v21(1시드, 원재료 유지+matchup+recent_form, 977점)보다는 높았지만 992(6시드)와는 시드 개수가 달라 직접 비교 불가.

### asof_* 전체 컬럼 상관계수 순위 정리 (8/23)

`asof_*` 19개 컬럼을 target과의 전체 상관계수 절대값 순으로 정리 — 강함(0.06↑): `success_rate`(+0.084)/`prev5_success`(+0.082)/`reverse_rate`(-0.080)/`prev3_success`(+0.078). 중간(0.05~0.06): `prev1_success`(+0.062)/`batter_success`(+0.059). 약함 이하: 나머지 13개(`middle_rate`류/`ball_rate`/`n`류/`strike_rate`+0.004/`fastball_rate`-0.0002 등 사실상 무의미한 것 포함).

**정제된 재조합 판단 기준(세 조건)**:
1. 원재료들의 개별 상관계수가 **비슷한 등급끼리** 모여있을 것(강한 것과 무의미한 것을 섞으면 강한 신호 희석).
2. 원재료끼리 **상호상관이 낮을 것**(높으면 압축해도 상쇄 이득 적음).
3. 연도별 range가 **전체 원재료보다 작아질 것**(control_risk_score 기준 통과 사례).

### 재조합 계열 3연속 실패 (8/22~23)

같은 원리를 다른 컬럼 그룹에 적용해봤으나 전부 실패:
- **matchup_skill_gap**(=success_rate-batter_success_rate) + **recent_form_gap**(=prev1_success-success_rate): 스크리닝은 통과(range 0.030/0.027, 부호flip 없음)했으나 fold 실측에서 개별로 애매/음수, "both"만 fold0 +5.30/fold2 -1.05. 실제 제출(v21, 1시드): **977점**(-15 vs 992).
- **control_quality_score**(=success_rate+strike_rate) + **pitcher_net_control**(=success_rate-control_risk_score): 스크리닝 스펙은 오늘 중 최강(range 0.021/0.031, corr 0.070/0.084)이었으나 fold0에서 뚜렷이 나쁨(-9.26~-11.22). "both"도 baseline 못 넘음. quality 단독+원재료 제거는 fold0 **-28.16**(이 세션 최악) — 강한 신호(success_rate, corr 0.084)를 무의미한 신호(strike_rate, corr 0.004)와 섞어 압축한 게 원인으로 추정.
- **pitcher_middle_sum**(4개 middle_rate 합): 스크리닝부터 애매(원재료끼리 상호상관 이미 높음, prev3↔prev5=0.846) → fold0 -14.26/fold2 -6.68로 기각.

**결론**: control_risk_score는 예외적 성공 사례였고, 재조합 피처 탐색 방향은 여기서 소진됨.

## 신경망/구조적 레버 시도 — 전부 실패

- **DeepFM 임베딩**(jh_ws, 8/20): 위 참고, 실패.
- **MLP(no-identity)**(es_ws, 8/18): GroupKFold 5-fold 평균 AUC 0.56917 vs CatBoost baseline 0.57171 — CatBoost보다 낮음.
- **CatBoost grow_policy=Lossguide**(비대칭 leaf-wise 트리, 8/23): control_risk_score+원재료제거 베이스 위에서 기존 Symmetric 대비 fold0 -11.41/fold2 **-27.61** — 뚜렷하게 악화. 대칭 트리의 내재적 정규화가 이 데이터에 유리한 것으로 해석(8/21 정규화 강화 실패와 같은 계열의 결론).
- **진짜 OOF 스태킹**: jh_ws는 스크립트만 작성하고 실행 안 함(사용자 판단으로 중단). **es_ws가 독립적으로 실제 실행 → 단일 최고 모델보다 -9.28~-29.21로 뚜렷이 나빠 기각**(그리드서치 블렌드가 스태킹보다 나음). "모델간 상관 높아 기대 낮음"이라는 사전 가설이 실측으로 확인됨.

## GRU/Transformer 시퀀스 모델 — 구현 불가 확정 (8/21)

외부에서 받은 설계 문서 검토 결과, 규칙 위반(`test.csv` 행 순서 기반 rolling/expanding 피처 금지)과 데이터 구조상 불가능(`game_id`/`pitch_no`/`game_date`가 train/test에 없어 같은 투수의 투구를 시간순으로 복원할 방법이 없고, `trackman_history.csv`는 그런 식별자가 있지만 `pitcher_trackman_id`-`pitcher_id` overlap=0이라 조인 불가) 둘 다에 걸려 착수 자체가 불가능함을 확정.

## 캘리브레이션/전처리 계열 세부 실험 (8/19~8/20, 다수 기각)

- **Isotonic calibration**: 사용자 본인(-66.79)과 팀원(3-fold walk-forward, -11~-28) 양쪽에서 독립적으로 기각.
- **D 앙상블**(분류기+sigmoid 50% + CatBoostRegressor(RMSE) 50%): 924 스케일에서는 유망해 보였으나(+31.0) 974 레시피 위에서는 정반대(-1.53)로 기각 — 베이스 레시피가 강해질수록 결과가 뒤집히는 사례.
- **iterations 2000→1000→500**: 961점(-13), 923점(-51) — 학습 데이터 스케일과 iterations를 맞추지 않으면 손해가 커짐을 확인.
- **시즌 recency 가중 학습**(sample_weight 지수감쇠): fold0/fold2 기준 미충족, 기각.
- **선수별(pitcher_id) 잔차 보정**(글로벌 모델+shrinkage 후처리): calib_carveout2 단계에서부터 손해, 기각.
- **버킷팅/winsorize/cold-start smoothing**(저위험 전처리 3종): 918(-5)/917(-6)/974.8(-0.1, 사실상 무변화) — 셋 다 974.9를 못 넘음. CatBoost 네이티브 결측/이상치 처리가 이미 충분히 좋다는 패턴 반복 확인.
- **pitcher×batter 매치업 히스토리**: cold-start 비율이 절반에 달해(fold2 -7.61) 기각.
- **raw id 학습셋 빈도 피처(freq974)**: fold0/fold2 이중 양수(+3.23/+5.59)로 이 프로젝트 기준 첫 통과였으나 **실제 -19로 반증** — "로컬 이중 통과도 안전하지 않다"는 경고 사례로 남음.
- **_isna 결측 플래그(16개)**, **calibration carve-out 5%→10%**, **median 블렌드(6모델)**, **체크포인트(스냅샷) 평균**: 전부 소폭 음수(-1.56~-9.80)로 로컬에서 걸러짐, 실제 제출까지는 안 감(982 확정 이후 "소폭 delta 그룹"으로 분류, 아직 재검증 여지 있음).

## 데이터 구조 관련 핵심 사실 (반복 참고용)

- `pitcher_id`/`batter_id`(train.csv, 익명 숫자)와 `pitcher_trackman_id`/`batter_trackman_id`(trackman_history.csv)의 overlap = **0**. trackman을 활용하려면 반드시 **상황 기반 키**(count_state/hand_matchup/inning_state 등)로 population-level 집계해서 병합해야 함.
- `asof_*` 결측(16개 컬럼, 전체의 약 2%)은 표본 0(cold-start)일 때 분모 0으로 NaN 전파되는 구조. CatBoost `nan_mode="Min"` 네이티브 처리가 이미 "표본 없음" 정보를 암묵적으로 보존하고 있어, 수동 개입(전역평균 대치, 플래그 추가)이 거의 항상 무의미하거나 손해.
- `control_success` 전체 평균이 2019~2024 단조 하락(레짐 변화), `game_type`(F/R)과 target 관계는 2023년 기점 역전.
- eval 시즌 시점의 `asof_pitcher_n`/`asof_batter_n`(누적 이력 표본)이 시즌이 늦을수록 두꺼워짐(2022시즌 평균 3194.7/3827.1 → 2024시즌 3928.2/5402.5) — 초기 fold(fold0→2022)일수록 원재료 자체의 노이즈가 커서 재조합 피처가 상대적으로 불리한 경향의 원인으로 추정(다만 절대 법칙은 아님, 반례도 있음).

## 1시드 실제 제출 3종 비교 — control_risk_score 원재료 제거(987)가 최고 (8/22~23)

6시드 풀 재학습(3시간)을 매번 거치지 않고, 새 피처 후보를 빠르게(단일 시드, 약 30분) 실제 제출로 검증하는 방식을 도입. 992 레시피(control_risk_score, 원재료 유지)를 기준으로 세 갈래 시도:

| 버전 | 구성 | 실제 점수 |
|---|---|---|
| v21 | control_risk_score(원재료 유지) + matchup_skill_gap + recent_form_gap | 977 |
| **v22** | **control_risk_score만, 원재료(reverse/middle/ball_rate) 제거** | **987**(1시드 중 최고) |
| v23 | v22 베이스 + control_quality_score + pitcher_net_control(원재료 유지) | 971 |

- v21/v23 둘 다 v22(987)보다 낮아 기각. quality_score/net_control은 원재료 유지/제거, risk_score 베이스 유지/제거를 조합한 모든 경우의 수(로컬 4가지 + 실제제출 1가지)를 다 시도했고 전부 실패로 확정.
- **1시드 실험 전용 기준점**: 앞으로 "6시드 대신 1시드로 빠르게" 검증하는 후보는 993/992(6시드)가 아니라 **987(v22, 1시드)을 비교 기준**으로 삼기로 함 — 시드 개수를 맞춰야 apples-to-apples 비교가 됨.
- **참고**: control_risk_score의 원재료를 유지(992, 6시드)하는 게 맞는지 제거(987 계열)하는 게 맞는지는 아직 같은 시드 수로 직접 비교한 적이 없어 미확정 상태.

## Brier Score 직접 최적화 calibration 실험 (8/23)

Isotonic(비모수, 두 번 기각: -66.79, -11~-28)과 기존 Platt(2파라미터 sigmoid, logloss/MLE 최적화) 사이의 변형 -- 똑같이 2파라미터 sigmoid를 쓰되 목적함수를 실제 채점 지표인 Brier Score(squared error)로 직접 최소화(`scipy.optimize.minimize`). 987점 베이스(drop_ingredients) 위에서 fold0/fold2 비교.
- **결과**: fold0 delta=**0.00**(완전히 동일), fold2 delta=**-0.51**(노이즈 수준) — 사실상 차이 없음.
- **해석**: Isotonic이 실패한 이유는 "지나치게 유연해서 과적합"이었는데, 이 실험은 파라미터 수를 Platt과 동일(2개)하게 유지한 채 목적함수만 바꾼 것이라 애초에 표현력에 차이가 없었음. 저차원 파라미터 공간에서는 logloss와 Brier 최적점이 사실상 같은 곳으로 수렴함. 기각.

## 외부 전략 문서 2건 검토 — 대부분 재탕이거나 규정 위반 소지 (8/23)

사용자가 외부에서 받아온 "1150점 돌파 전략" 문서 2건(`LG_Aimers9_Phase2_1150_Breakthrough_Strategy.md`, `LG_Aimers9_Execution_Strategy_1150.md`)을 프로젝트 실제 이력과 대조 검토.

- **규정 위반 위험**: 1번 문서의 "Test 예측값 평균을 목표치로 sweeping"하는 Global Mean Shift는 `competition_rules.md`의 "평가 데이터 전체 분포를 이용한 사후 보정 금지" 조항과 정확히 충돌 -- 실격 사유가 될 수 있음을 경고. 2번 문서는 이를 인지하고 "train.csv 추세만으로 계산한 고정 상수" 방식으로 스스로 수정했으나, 그마저도 8/18 `season_trend_prior` 실험에서 "baseline 모델이 이미 예측평균-실제평균 gap 0.8%p로 추세를 잘 흡수하고 있다"는 게 확인된 바 있어 불필요할 가능성이 높다고 판단, 실행 안 함.
- **데이터에 없는 것을 가정**: `in_zone_rate`/`whiff_rate`/`first_pitch_strike_rate` 제안은 `trackman_history.csv`에 스윙/판정/코스 위치 컬럼이 전혀 없어 애초에 계산 불가능함을 확인.
- **이미 실패한 것과 겹침**: RMSE 회귀 블렌딩(=8/19 "D 앙상블", -1.53), Bayesian smoothing(=8/20 cold-start smoothing, -0.1), 최근 3년 서브모델(=8/20 recency 가중, 기각), trackman 상황 키 확장(=v17 month축, -18.9)과 각각 동일 계열.
- **유일하게 새로 시도할 가치가 있던 것**: Brier-최적화 Platt(위 섹션) -- 실행해봤으나 무의미했음.
- **결론**: 두 문서 모두 실질적으로 새로운 방향을 제시하지 못함, 실행 안 함(Brier-최적화 Platt 제외).

## 현재 상태 (2026-08-23 기준, 최신)

- **팀 전체 최고: 993점**(es_ws v11, `submit_v11_riskscore_blend`).
- jh_ws 단독 최고(6시드): 992점(`v20_control_risk_score`).
- jh_ws 1시드 실험 최고: 987점(`v22_drop_ingredients_1seed`).
- 1150(과거 확인된 타 팀 리더보드 기준) 대비 격차 약 157~179점, 지금까지 시도한 어떤 단일 실험도 이 정도 규모의 개선을 낸 적이 없음. 이 목표 자체가 8/20~21 확인 시점의 스냅샷이라 재확인 필요.
- **탐색이 거의 소진된 방향**: 재조합 피처(control_risk_score 제외 전부 실패, quality/net_control 모든 조합 소진), 신경망(DeepFM/MLP 둘 다 실패), 트리 구조 변경(Lossguide 실패), 진짜 스태킹(es_ws가 실측으로 실패 확인), 정규화 강도 조정(실패), 팀 매핑(매칭 근거 없음), 시퀀스 모델(규칙상 불가), calibration 변형(Isotonic/Brier-최적화 Platt 둘 다 무의미/악화), 외부 전략 문서 2건(대부분 재탕 또는 규정 위반 소지).
- **아직 안 해본 것**: 남은 "소폭 delta 그룹"(_isna 플래그/carve-out 비율/median 블렌드/체크포인트 평균 재검증), LGBM/XGBoost를 jh_ws의 Optuna 튜닝값으로 교체한 블렌드 재검증, control_risk_score 원재료 유지 vs 제거를 같은 시드 수(6시드)로 직접 비교.
