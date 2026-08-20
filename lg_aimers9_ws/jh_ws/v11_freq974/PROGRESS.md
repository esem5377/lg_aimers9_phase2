# v11_freq974 — raw id 학습셋 빈도 피처, 실제 제출 974→955 하락(-19), 기각

## 배경
8/19~8/20 세션에서 es_ws 974점 레시피(`submit_v7_catboost_calibrated_rawid`)에
대한 반증 시도를 여러 개 진행(EWM/D앙상블/Isotonic/regime화 x2/iterations
x2/segment-Platt/recency가중/corr+importance필터링, 전부 기각/보류). 그중
마지막으로 남아있던 낮은 우선순위 후보 `pitcher_id`/`batter_id` 학습셋
총빈도 파생피처(`pitcher_train_freq`/`batter_train_freq`)만 유일하게 로컬
3-fold walk-forward(iterations=1000)에서 **fold0(→2022) +3.23, fold2(→2024)
+5.59로 두 폴드 독립 양수 일치** — 이 프로젝트에서 raw id 도입(+9.47/+9.48)
성공 이후 계속 신뢰해온 채택 기준을 처음으로 통과한 사례였음.

## 실행
- `train_final.py`: es_ws 974 레시피(CAT_COLS 네이티브 categorical + raw
  pitcher_id/batter_id label-encoded + trackman context 3종 + BEST_PARAMS)
  그대로에 `pitcher_train_freq`/`batter_train_freq`(전체 학습 데이터
  season<=2024 내 등장 횟수, OOV=0)를 추가. 정식 스케일(iterations=2000)
  홀드아웃 재확인 없이 사용자 판단으로 바로 프로덕션 학습 진행 — early
  stopping과 calibration을 5% stratified carve-out 하나로 통합한 단일 fit
  구조로 단순화(별도 time-split CV 단계 생략).
- carve-out(5%) 검증: AUC 0.578, BSS calib 2067.24, best_iteration=1999.
- `script.py`: v7 script.py에 빈도 매핑 조회(없는 id는 0) 로직만 추가.
- 클린룸 2회 검증(스테이징 실행 + zip 재추출 재실행) 통과, 예측값 완전
  동일, OOV pitcher_id 3건/batter_id 2건으로 v7과 일치.

## 결과 — 실제 리더보드 974 -> **955 (-19), 기각**
현재 팀 전체 최고 제출은 변동 없이 `es_ws/work/submissions/submit_v7_catboost_calibrated_rawid.zip`
(974점)으로 유지. 이 브랜치는 production에 반영하지 않음.

## 중요한 발견
이 프로젝트에서 "fold0/fold2(2022/2024) 독립 양수 일치"는 raw id 도입과
확률보정, 두 성공 사례의 공통 근거였던 채택 기준인데, 이번이 **그 기준을
이중으로 만족한 첫 사례였음에도 실패**했다. 지금까지의 다른 실패
사례(sit_*, native cat_features, platoon+risp, regime2023, segment-Platt,
recency가중, corr+importance 필터링)는 전부 단일 폴드 의존이었거나 그
기준 자체를 통과하지 못했던 경우였다는 점에서 이번 결과는 결이 다르다.

원인 가설(미검증): `pitcher_train_freq`는 2019~2024라는 고정 선수 풀
안에서의 총 등장 횟수 스냅샷인데, walk-forward 검증 폴드(→2022, →2024)도
전부 같은 2019~2024 폐쇄 풀 안의 시즌이라 "빈도가 높다=베테랑/주전"이라는
신호가 검증 시점엔 안정적으로 보였을 뿐이다. 반면 실제 test(2025시즌)는
로스터 구성 자체가 바뀌는(신인 데뷔, 트레이드, 은퇴) 첫 해라, 이
프로젝트의 walk-forward 설계 자체가 "선수 풀 구성 변화"라는 축의 일반화는
폴드가 아무리 일치해도 검증할 수 없는 구조였을 가능성이 있다.

## 다음 세션 권고
이걸로 974점에 대한 반증 시도가 사용자+팀원 합쳐 10개 이상 쌓였고, 이번엔
프로젝트의 검증 설계 자체(이중 폴드 일치 기준)까지 반증됐다. 로컬 검증만
으로 974점을 넘어설 수 있다는 기대치를 상당히 낮춰야 하며, 추가 실험보다
974점을 최종안으로 확정하고 제출물/코드/PPT 정리로 넘어가는 것을 적극
권장한다.
