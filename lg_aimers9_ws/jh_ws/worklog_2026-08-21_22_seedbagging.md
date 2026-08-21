# 작업 로그 (2026-08-21 ~ 2026-08-22, jh_ws)

기준선: 세션 시작 시점 974.9점(es_ws v7). **이 세션 결과로 팀 전체 최고가 982점(jh_ws v18_seed_bagging)으로 갱신됨.**

## 결론부터

**새 팀 최고: 982점** — `jh_ws/v18_seed_bagging/v18_6seed_bagging.zip`
974 레시피(CAT_COLS 네이티브/RAW_ID_COLS/trackman context/BEST_PARAMS)는 전부 그대로 두고, **완전히 동일한 모델을 랜덤시드만 6개(42/7/123/1/99/777) 바꿔 학습해 raw 확률을 평균낸 뒤 Platt 보정 1회**만 다르게 적용. 새 피처나 정보는 전혀 추가하지 않음 — 순수하게 모델 분산(variance)만 줄이는 방식.

- 3시드 버전: 실제 제출 974.9 → **979**(+4.1)
- 6시드로 확장(기존 42/7/123 재사용 + 1/99/777 신규 학습): 실제 제출 979 → **982**(+3)

**중요한 발견**: 이 프로젝트에서 반복됐던 "로컬 개선 → 실제 반증" 패턴(sit_*/platoon+risp/regime2023/freq974 등)이 시드 배깅에서는 처음으로 깨짐. 로컬 fold0/fold2 예측 delta(+4.34/+3.98)와 실제 리더보드 delta(+4.1)가 **크기까지 거의 정확히 일치**했음. 해석: 시드 배깅은 새로운 정보/피처를 추가하지 않고 순수하게 노이즈만 줄이는 방식이라, 다른 실패들의 공통 원인이었던 "특정 시즌 분포에 대한 과적합"이 원천적으로 발생하지 않음.

## 세션 전체 실험 목록 (전부 이 폴더에 스크립트+결과 json 첨부)

| # | 실험 | 파일 | 결과 | 판정 |
|---|---|---|---|---|
| 1 | 정규화 강화(depth=5,l2=50 등) | `reg_strength_hypothesis.py` | fold0/fold2/GroupKFold **3축 전부 -97~-168** | 기각 |
| 2 | **시드 배깅 (3개, 로컬 검증)** | `seed_bagging.py` | fold0 +4.34, fold2 +3.98 | **채택 → 실제 제출 979점** |
| 3 | Monotonic constraints (asof_* 8개) | `monotonic_constraints.py` | fold0 -36.87, fold2 -15.41 | 기각 |
| 4 | 부트스트랩 배깅(데이터도 리샘플) | `bootstrap_bagging.py` | 기존 시드배깅 대비 fold0 -7.27, fold2 -24.93 | 기각 |
| 5 | calibration carve-out 5%→10% | `carveout_ratio.py` | fold0 -7.85, fold2 -2.59 | 기각 |
| 6 | 투수 오토인코더 임베딩(비지도, 4dim) | `autoencoder_embedding.py` | fold0 **-103.61**, fold2 **-197.42** | 기각(이 세션 최악) |
| 7 | 체크포인트(스냅샷) 평균 (재학습無) | `checkpoint_avg.py` | carve-out -9.80 | 기각 |
| 8 | median 블렌드 (mean 대신) | `checkpoint_avg.py`(같은 스크립트) | carve-out -1.56 | 기각 |
| 9 | LGBM Optuna 튜닝(30 trial) | `optuna_lgbm_search.py` | 미튜닝比 +109(1307→1416)이나 **CatBoost 미튜닝(1481)에도 못 미침** | 참고용, 채택 안 함 |
| 10 | 로지스틱 회귀 sanity check | `logistic_baseline.py` | fold0 1851(CatBoost 2386 대비 열세), fold2 **0.00**(AUC 0.522, 붕괴) | 기각 |
| 11 | control_risk_score(reverse+middle+ball rate 합) | `control_risk_score.py` | fold0 -3.74, fold2 **+7.04** | 트레이드오프, 기각 |
| 12 | `_isna` 결측 플래그(16개) | `isna_flags.py` | fold0 -3.84, fold2 -2.75 | 기각 |

모든 로컬 실험은 season walk-forward(fold0: train≤2021/eval==2022, fold2: train≤2023/eval==2024), iterations=1000, 정직한 calibration(TRAIN 파티션 내부 5% carve-out에만 Platt fit) 원칙 준수. GroupKFold(미본 투수) 축은 시간 관계상 대부분 스킵(#1만 포함).

## 프로덕션 (실제 제출로 검증된 것)

`jh_ws/v18_seed_bagging/`:
- `train_final.py`: 3시드 초기 버전 (974 레시피 + 시드 42/7/123, iterations=2000 전체데이터)
- `train_add_seeds.py`: 6시드 확장 (기존 3개 모델 재사용, 신규 3개만 추가 학습)
- `script.py`: N개 모델 전부 로드 → 각각 예측 → raw 평균 → Platt 보정 (seeds는 `feature_meta.json`에서 동적으로 읽음)
- `v18_6seed_bagging.zip`: **현재 제출 파일 (982점)**

## 다음에 시도해볼 만한 것 / 이미 막힌 것 (다음 세션 참고)

**막힌 것(반복 검증 불필요)**:
- 새 피처 추가류(situational/platoon/risk_score/isna 등) 전부 이 프로젝트에서 반복 실패 — 사실상 이 레시피에 새 정보를 추가하는 방향은 거의 다 막혔다고 봐도 됨
- 비지도 임베딩(DeepFM, 오토인코더) 둘 다 명확히 손해
- 로지스틱회귀/CART 단일트리 계열은 이론+실측 둘 다 CatBoost에 크게 못 미침
- 시퀀스 모델(GRU/LSTM/Transformer)은 `game_id`/`pitch_no` 부재 + 규칙 위반으로 구조적으로 불가능

**아직 안 해본 것**:
- 시드를 6→9~12개로 더 늘리기 (한계효용 체감 중이라 기대치는 낮음, carve-out 기준 3→6이 +3.3)
- 진짜 OOF 기반 스태킹(메타러너 학습) — 다만 CatBoost/LGBM/XGBoost 상관성이 높아서 그리드서치 블렌드(+2.90) 대비 큰 이득 기대 어려움

**현재 판단**: 982점이 이 레시피 계열(CatBoost + situational trackman + raw id + 시드배깅)의 현실적 상한에 가까워 보임. 팀원 쪽(es_ws) 새 아이디어 있으면 공유 바람.
