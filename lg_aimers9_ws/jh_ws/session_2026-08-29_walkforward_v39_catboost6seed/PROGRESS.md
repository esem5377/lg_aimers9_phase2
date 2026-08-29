# v39 후보: CatBoost 6-seed 배깅 walk-forward 검증 — 중단, 결과 없음

## 배경
- es_ws가 독립적으로 진짜 walk-forward(fold2_2024, carve-out 아님)로 CatBoost
  1시드->6시드 배깅을 확인: solo CB delta=+18.06, CB+retrieval(0.7:0.3) 블렌드
  delta=+10.88 (`es_ws/work/pipeline/v13_walkforward_check.log`). jh_ws의 현재
  최고(v34, 1024점)는 여전히 CatBoost 1시드(seed42)만 씀. jh_ws 규칙상 fold0/fold2
  둘 다 양성이어야 제출 후보 자격 -- es_ws는 fold2만 확인했으므로 fold0도 검증하려고
  시작.

## 상태 (중단됨, 미완료)
- fold2용 CatBoost seed42 + retrieval(batch1024) 체크포인트는 v37 walk-forward에서
  복사해와 재사용 준비(`checkpoints/fold2_2024/`), seed=7 학습 도중 사용자가 "하고
  있는거 멈춰봐"로 중단 요청 -> 프로세스 kill. seed=7 이후 신규 시드는 하나도 완료
  못함(체크포인트 없음).
- 사용자가 이후 "아예 새로운 방법 탐색해"로 방향을 전환(v40: pitcher_id/batter_id
  categorical)했기 때문에 이 검증은 현재 우선순위 아님.

## 재개 방법
`v39_walkforward.py` 재실행 시 fold2 seed=42/retrieval은 체크포인트로 스킵되고
seed=7부터 이어서 진행. 재개할 가치가 있는지는 v40 결과(931점, 역대 최악 회귀,
`v40_id_as_cat_1seed/PROGRESS.md` 참고) 이후 사용자 판단 필요.
