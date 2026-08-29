# v37: retrieval encoder in-batch negative pool 확대 (BATCH_SIZE 1024->4096) — 1020점, 기각

## 배경
- 2026-08-28 세션: "cosine similarity로 전환" 요청 확인 중 RowEncoder.encode()가 v25부터
  이미 L2 정규화 후 z@z.T(=cosine similarity)를 쓰고 있었음을 재확인, 델타 없는 실험으로
  기각. 대신 아직 안 건드린 negative sampling 축에서 in-batch negative pool을
  1024->4096(4배)로 확대하는 실험으로 방향 전환(사용자 선택).
- v35(용량 확대: embed_out 32->64, hidden [256,128]->[512,256])는 carve-out calib_bss가
  v26 대비 거의 그대로였음(2082.07 vs 2082.78) -- 용량은 병목이 아니라는 신호였고, 이번
  실험은 그 축(negative pool 크기)만 분리해서 봄.
- **walk-forward 검증 결과를 기다리지 않고 사용자 요청으로 production 빌드 + 제출을
  먼저 진행**(lg_aimers9_walkforward_methodology 규칙 명시적 우회). CatBoost는 v26
  재사용, retrieval encoder만 BATCH_SIZE=4096으로 전체 데이터 재학습.

## 결과
- carve-out(랜덤 5%, 7.4만행): calibrated BSS **2122.76** (best_w_catboost=0.60)
  — v26(w=0.70) 2082.78 대비 **+39.98**, v35(용량확대) 2082.07 대비 **+40.69**
- **실제 리더보드: 1020점**
- v26(동일 구조의 2-way CatBoost+retrieval 블렌드, 1023점) 대비 **-3점**
- 현재 팀 최고인 v34(CatBoost+retrieval+EB-GLMM 3-way, 1024점) 대비 **-4점**
- carve-out에서 가장 크게 개선됐던 실험(+40, 이 프로젝트 기록상 이례적으로 큰 로컬
  델타)이 실제로는 소폭 역행함 -- v32(carve-out 1위 importance, fold2 -89.76)와 같은
  계열의 "carve-out만의 신호는 시간 기반 분포 변화에 취약하다"는 패턴 재확인.

## 진행 중이던 walk-forward 검증 (미완료, 중단)
- `session_2026-08-28_walkforward_v37_negpool/`: fold2는 완료(batch1024 eval_bss=848.21,
  batch4096 eval_bss는 retrieval 추론 직전에 사용자 요청으로 프로세스 중단), fold0은
  시작 전. 체크포인트(`checkpoints/fold2_2024/`)에 CatBoost + 두 배치 변형 인코더가
  모두 저장돼 있어 재개 시 retrieval 추론부터 이어서 가능.
- 실제 리더보드가 이미 -3으로 나왔기 때문에 이 walk-forward를 마저 돌리는 건 "제출
  여부 결정"으로서는 의미가 없어졌지만, **fold2가 이 역행을 미리 잡아냈을지는 여전히
  방법론 검증 가치가 있음**(v36처럼 walk-forward가 실측과 일치했는지, 아니면 이번엔
  fold까지도 놓쳤는지 확인 필요).

## 결론
- in-batch negative pool 확대(배치 크기만 키우는 방식)는 기각. carve-out에서의 대형
  개선(+40)을 실제 제출 근거로 삼기엔 위험하다는 사례가 하나 더 쌓임.
- **방법론 시사점**: 이번 세션은 walk-forward 게이트를 의도적으로 우회하고 제출까지
  갔는데 실제로 역행했음 -- v36에 이어 두 번째로, "carve-out만 보고 우회 제출"이
  손해로 이어진 사례. 다음부터는 carve-out 델타가 아무리 커도(오히려 이례적으로 클수록)
  fold0/fold2 walk-forward를 완주한 뒤 제출하는 쪽을 기본값으로 되돌리는 게 맞아 보임.
