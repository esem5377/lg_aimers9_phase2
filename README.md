# LG Aimers 9 Phase 2 — 투구 제구 성공 확률 예측 📊

**최종 팀 최고 스코어 1024점** (baseline RandomForest 549.51 대비, 시작 시점 909 대비 +115)

**최종 리더보드 순위 398등**

- **과제**<br />투구 직전까지 알 수 있는 경기 상황 / 선수 정보 / 과거 이력 피처로
  해당 투구의 제구 성공 확률(`control_success`)을 예측
- **평가지표**<br />Brier Skill Score — `max(0, 100000 × (1 − BrierScore / baseline BrierScore))`
- **기간**<br />2026. 08. 16. ~ 09. 01., 약 40회 이상의 실험 사이클
- **데이터**<br />`train.csv` 1,475,092행 × 49컬럼 (2019~2024), `trackman_history.csv` 1,793,078행 × 30컬럼
- **제약**<br />외부 데이터/API 금지, row-wise 독립 추론(`test.csv` 타 행 참조 금지), 완전 오프라인 추론
- **협업 구조**<br />두 워크스페이스(`jh_ws`, `es_ws`)가 독립적으로 실험하고 서로 pull하며 발전

<br />

<div align="center">
<table>
<tr>
<td align="center" valign="top" width="180">
  <a href="https://github.com/jeongjaehyeog331-del"><img src="https://github.com/jeongjaehyeog331-del.png?size=200" width="110" height="110" alt="" /></a><br />
  <a href="https://github.com/jeongjaehyeog331-del"><b>jh_ws</b></a><br />
  <sub>모델 아키텍처<br />Retrieval&nbsp;/&nbsp;앙상블 설계<br />시드 배깅</sub>
</td>
<td align="center" valign="top" width="180">
  <a href="https://github.com/esem5377"><img src="https://github.com/esem5377.png?size=200" width="110" height="110" alt="" /></a><br />
  <a href="https://github.com/esem5377"><b>es_ws</b></a><br />
  <sub>확률보정&nbsp;/&nbsp;검증 방법론<br />피처 엔지니어링<br />클린룸 검증</sub>
</td>
</tr>
</table>
</div>

<br />

## 결과 요약

| 날짜 | 점수 | 버전 | 핵심 변화 |
|---|---|---|---|
| ~8/17 | 909 | jh_ws | LGB+CatBoost+XGBoost 3모델 앙상블 |
| 8/17 | 924 | jh_ws v9 | GroupKFold → 시간 기반(season) 검증으로 교체 |
| 8/18 | 959 | es_ws v6 | CatBoost 단일 + Platt 확률보정 |
| 8/18 | 974 | es_ws v7 | + raw pitcher_id/batter_id(label-encoded) 재도입 |
| 8/21 | 982 | jh_ws v18 | 6시드 배깅 |
| 8/22 | 993 | es_ws v11 | control_risk_score 피처 + LGBM 블렌드 |
| 8/23 | 1020 | jh_ws v25 | CatBoost + retrieval(ModernNCA 스타일 인코더, 133만행 exact kNN) 블렌드 도입 |
| 8/24 | 1023 | jh_ws v26 | 블렌드 가중치 그리드서치 (CatBoost:retrieval = 0.7:0.3) |
| 8/29 | **1024** | jh_ws v34 | + EB-GLMM(last-season shrinkage) 3-way 블렌드 |

이후(v35~v40, 8종) 및 es_ws의 RMSE-loss 실험까지 포함해 두 자릿수 이상의 추가 실험이
있었지만 전부 1024를 넘지 못했거나(9/1, 대회 종료로 제출 못함), 실제 리더보드에서
역행해 기각됐다. 자세한 이력은 [`lg_aimers9_ws/jh_ws/FULL_PROJECT_LOG.md`](lg_aimers9_ws/jh_ws/FULL_PROJECT_LOG.md)
와 각 워크스페이스의 `worklog/`, `session_*/` 참고.

## 아키텍처 (jh_ws v34, 최종 채택)

```
raw features ─┬─ control_risk_score 재조합 (원재료 3종 → 위험도 스코어)
              ├─ raw pitcher_id / batter_id (label-encoded, CatBoost numeric)
              └─ as-of 누적률(투수/타자 성공률, 최근 N경기, 구종 믹스) ── 대회 제공

        ┌───────────────┐   ┌────────────────────────┐   ┌───────────────────┐
        │  CatBoost      │   │  Retrieval (ModernNCA)  │   │  EB-GLMM            │
        │  (GBDT, 70%)   │   │  인코더 임베딩 +          │   │  직전 시즌 empirical- │
        │                │   │  133만행 exact kNN(30%) │   │  Bayes shrinkage(5%) │
        └───────┬────────┘   └───────────┬─────────────┘   └──────────┬──────────┘
                └────────────────┬────────┘────────────────────────────┘
                                 weighted blend
                                       │
                                 Platt calibration
                                       │
                              control_success 확률
```

트리 기반(CatBoost)과 거리 기반(retrieval)이라는 서로 다른 귀납 편향을 가진 두 축을
블렌드한 것이 993 → 1023 도약의 핵심(8/23~24). EB-GLMM은 그 위에 소폭 얹은 세 번째 축.

```
lg_aimers9_ws/jh_ws/v34_threeway_ebglmm_blend/
  script.py             # 실제 제출용 추론 엔트리포인트 (test.csv -> submission.csv)
  01_build_v34.py       # CatBoost + retrieval + EB-GLMM 3-way 블렌드 학습
  02_finalize_fixed_weight.py
  model/                # catboost_seed42.cbm, retrieval_encoder.pt, ebglmm_state.pkl 등
```

## 이 프로젝트에서 배운 것: 로컬 검증 방법론

40회 이상의 실험 사이클에서 가장 값진 결과는 사실 스코어 자체보다 **"어떤 로컬 검증을
믿을 수 있는가"**에 대한 반복 검증이었다.

- **랜덤 5% carve-out(`train_test_split(stratify=y)`)은 믿을 수 없다.** 시간 순서를
  무시한 분할이라, 같은 투수/타자의 다른 투구가 train/calib 양쪽에 섞여 들어가며
  leak성 이득을 준다. 이 프로젝트에서 최소 5번, 랜덤 carve-out에서 좋아 보였던 변경이
  실제 리더보드에서 역행했다 (가장 큰 사례: 8/29 pitcher/batter id를 CatBoost
  categorical로 전환 — carve-out 델타 **+186**(역대 최대) → 실제 **-93점**(역대 최악)).
- **시간 기준 walk-forward(train≤2021→eval 2022, train≤2023→eval 2024, 2개 fold)가
  기준선.** 두 fold가 모두 같은 방향·비슷한 크기로 동의해야 실제 전이를 신뢰할 수
  있었다 (예: raw id 피처 — fold별 +9.47/+9.48 → 실제 +15). 한 fold만 보고 지른
  경우(v13, v29)는 실제로 반증됐다.
- **AUC가 아니라 BSS(Brier Skill Score, 대회 채점 지표)로 검증해야 한다.** 확률보정
  같은 변경은 AUC엔 안 잡히지만 BSS/실제 스코어에는 크게(로컬 +24.8 → 실제 +70) 반영된다.
- 그래도 **fold가 다 동의해도 실패하는 경우가 있었다** (v29: fold0 +21.94/fold2
  +8.60 둘 다 동의했으나 실제 -10 — 원인은 fold 전용으로 축소 재학습한 encoder와
  프로덕션 encoder의 용량/데이터량 불일치). 즉 walk-forward도 필요조건이지 충분조건은
  아니다.

## 구성

```
lg_aimers9_ws/
  es_ws/                       # eunsoo 워크스페이스
    open/
      data_description.md      # 데이터 설명서 (컬럼 정의, 제출 규칙 등)
      competition_rules.md     # 대회 규칙 (사전학습모델/외부 API/외부 데이터 사용 제한 등)
      data/                     # 원본 데이터 (.gitignore 처리, 각자 로컬에 준비 필요)
    work/
      pipeline/                 # 학습/추론/walk-forward 검증 스크립트
      experiments/               # 튜닝/피처 실험 스크립트 (채택 여부 무관, 전부 보존)
      submissions/                # 제출 패키지 원본 + zip (v2~v15, v_rmse_retrieval)
      model*/                    # 버전별 학습된 모델 아티팩트
    worklog/                    # 날짜별 작업 기록

  jh_ws/                        # 친구 워크스페이스
    v1~v40, session_*/          # 버전/세션별 실험 (스크립트 + 모델 + 결과)
    FULL_PROJECT_LOG.md          # 팀 전체 종합 로그 (리더보드 이력 + 방법론 교훈)
    worklog_*.md                 # 세션별 작업 기록
```

## 실행 방법

```bash
# 1. 데이터 준비 — 용량 초과(train 352MB, trackman_history 338MB)로 레포 미포함,
#    대회 배포본을 lg_aimers9_ws/es_ws/open/data/ 아래에 그대로 배치
# 2. 환경 세팅
pip3 install --user --break-system-packages pandas scikit-learn lightgbm joblib catboost torch

# 3. 최종 채택 파이프라인(jh_ws v34) 학습 재현
python3 lg_aimers9_ws/jh_ws/v34_threeway_ebglmm_blend/01_build_v34.py
python3 lg_aimers9_ws/jh_ws/v34_threeway_ebglmm_blend/02_finalize_fixed_weight.py

# 4. 제출 zip의 추론 엔트리포인트 단독 실행 (test.csv -> submission.csv)
cd lg_aimers9_ws/jh_ws/v34_threeway_ebglmm_blend && python3 script.py
```

`python3 -m venv`가 이 환경에서 sudo 없이 정상 동작하지 않아 `--user
--break-system-packages`로 설치함. 다른 환경이면 그냥 venv 써도 무방.

## 제출 규칙 요약

자세한 건 [`data_description.md`](lg_aimers9_ws/es_ws/open/data_description.md),
[`competition_rules.md`](lg_aimers9_ws/es_ws/open/competition_rules.md) 참고.

- 평가 데이터는 행 단위 독립 예측만 허용 — test.csv 내부 행을 이용한 집계/타겟인코딩/롤링 피처 금지.
- 현재 투구의 사후 정보(실제 코스, 판정, 구종, Trackman 실측값), 2025년 Trackman 데이터 사용 금지.
- 제공된 `asof_*` 컬럼(사전 계산된 과거 이력 피처)은 사용 가능.
