# 야구 투구 제구력 예측 모델 고도화 방안 (Pitch Command Modeling Proposals)

본 문서는 전통적인 정적 수치 분류 모델의 한계를 극복하고, 야구공의 역학적 궤적, 투수-타자 상성 관계망, 투구 메커니즘 일관성, 투수별 개인화 특성을 반영하기 위한 **4가지 차세대 딥러닝/머신러닝 고도화 아키텍처**를 정리한 기술 제안서입니다.

---

## 1. 물리 기반 딥러닝 (Physics-Informed Deep Learning)

단순한 정적 테이블 피처 기반 예측을 넘어, 공기역학 및 비행 궤적 물리 법칙을 모델 구조 및 손실 함수(Loss Function)에 직접 융합하는 접근법입니다.

### 1.1 3차원 물리적 궤적 복원 (Trajectory Reconstruction)
- **입력 파라미터**: 릴리스 포인트($X_0, Y_0, Z_0$), 초기 구속($V_0$), 3축 가속도/무브먼트($a_x, a_y, a_z$), 스핀 레이트 및 회전축
- **궤적 방정식 수식화**:
  $$ec{r}(t) = ec{r}_0 + ec{v}_0 t + rac{1}{2}ec{a}t^2 + \int ec{F}_{	ext{Magnus}}(t) dt$$
- **입체적 판정 접점 도출**:
  - 홈플레이트 전면 및 3차원 스트라이크 존(Strike Zone Volume)과의 교차 지점 좌표($(X_{	ext{plate}}, Z_{	ext{plate}})$)를 물리적으로 역추적/순방향 계산.
  - 단순 통계적 분류가 아닌, **물리적 제구 성공 허용 오차 한계(Command Error Margin)**를 동적으로 산출하여 타깃 피처로 결합.

### 1.2 딥러닝 기반 Tabular 모델 도입 (TabNet / FT-Transformer)
- **기존 트리 모델(XGBoost, LightGBM)의 한계**: 연속적인 물리량 사이의 복합 비선형 상호작용 및 시공간적 연속성을 분할(Split) 방식으로만 학습하여 디테일 손실 발생.
- **적용 방안**:
  - **TabNet**: 순차적 어텐션(Sequential Attention) 메커니즘을 통해 각 투구 결정에 기여한 핵심 물리 변수(릴리스 높이, 수직 무브먼트 등)를 단계별로 마스킹/선택하여 해석력 및 예측력 극대화.
  - **FT-Transformer (Feature Tokenizer Transformer)**: 모든 수치형/범주형 물리 피처를 임베딩 토큰화한 후 Self-Attention을 적용해 구속-회전수-릴리스 위치 간의 고차원 상호작용 포착.

---

## 2. 그래프 신경망 (Graph Neural Networks, GNN) 기반 접근

투구 이벤트를 단일 행(Row)으로 독립 처리하지 않고, 투수-포수-타자-볼카운트-경기 상황 간의 복합 네트워크 그래프로 모델링합니다.

### 2.1 그래프 노드 및 엣지 정의
| 구성 요소 | 항목 | 설명 |
| :--- | :--- | :--- |
| **노드 (Nodes)** | **투수 (Pitcher)** | 투수 고유 ID 및 기본 프로필 임베딩 |
| | **타자 (Batter)** | 상대 타자 특성 (좌/우타, 존별 핫/콜드존) |
| | **구종 (Pitch Type)** | 패스트볼, 슬라이더, 체인지업 등 |
| | **상황 (Context)** | 볼카운트, 아웃카운트, 주자 상황, 이닝 |
| **엣지 (Edges)** | **과거 매치업 기록** | 특정 타자/상황 대비 제구 성공/실패 이력 가중치 |
| | **상성/시퀀스 연결** | 직전 투구 구종 및 로케이션 전이 관계 |

### 2.2 원리 및 학습 방식
- **Graph Convolutional Network (GCN) / Graph Attention Network (GAT)**:
  - 투수의 고유 핑거프린트(투구 스타일, 피로도, 릴리스 안정성)가 특정 타자 및 볼카운트 노드와의 상호작용을 거쳐 전달되는 과정을 임베딩으로 학습.
  - GAT 어텐션 계수를 통해 "특정 볼카운트에서 특정 타자를 상대로 제구 불안정성이 커지는 관계 패턴"을 직접 시각화 및 정량화.

---

## 3. 이상치 탐지 (Anomaly Detection) 및 분포 기반 분류

"제구 성공(1/0)"을 일반적인 이진 분류로 직접 풀기 전에, **"투수 본인의 정상 메커니즘 기준 분포에서 얼마나 벗어났는가(Mechanical Anomaly Index)"**를 측정하여 모델의 핵심 특성으로 활용합니다.

### 3.1 Autoencoder 기반 릴리스/무브먼트 재구성 오차
- **정상 베이스라인 학습**:
  - 투수별 **성공 투구(Good Command/In-Zone Execution)** 데이터셋만을 추출하여 Autoencoder 학습.
  - 입력 피처: 릴리스 포인트($X, Y, Z$), 릴리스 각도, 스핀 액시스, 체감 속도 등.
- **재구성 오차(Reconstruction Error) 도출**:
  $$	ext{Anomaly Score} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$
  - 새로운 투구 입력 시 재구성 오차가 임계값 이상이면 메커니즘 붕괴(제구 실패 확률 급증)로 판단.

### 3.2 Mahalanobis Distance & Isolation Forest
- **마할라노비스 거리 (Mahalanobis Distance)**:
  - 투수별 물리 지표의 다변량 정규분포 공분산 행렬($\mathbf{\Sigma}$)을 기반으로 중심점($oldsymbol{\mu}$)과의 거리 계산:
    $$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - oldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - oldsymbol{\mu})}$$
  - 변수 간 상관관계를 고려한 정밀한 메커니즘 이탈 점수 도출.
- **활용 방안**: 메인 앙상블 모델의 핵심 수치형 Feature 또는 사전 필터링 확률값으로 투입.

---

## 4. 메타 학습 (Meta-Learning / Few-Shot Learning)

투수마다 릴리스 포인트, 폼, 구종 메커니즘이 완전히 상이하므로, 전체 데이터를 단일 모델로 통합 학습할 때 발생하는 개인화 성능 저하를 해결합니다.

### 4.1 MAML (Model-Agnostic Meta-Learning)
- **개념**: "제구 예측 모델" 자체를 고정하는 것이 아니라, **"새로운 투수의 5~10개 투구 샘플(Few-shot)만 보고도 해당 투수에게 최적화되는 초기 가중치($	heta$)"**를 학습.
- **알고리즘 흐름**:
  1. **Inner Loop**: 개별 투수 Task $T_i$에 대해 소량의 투구 데이터로 $k$-step Gradient Descent 수행.
  2. **Outer Loop**: 모든 투수에 대해 적응 후 검증 오차가 최소화되도록 공통 메타 가중치 업데이트.

### 4.2 계층적 베이지안 & 멀티태스크 학습 (Multi-Task Architecture)
- **Shared Backbone Network**: 투구 궤적의 공통 공기역학, 릴리스-홈플레이트 간 물리 법칙 등 보편적 패턴 학습.
- **Pitcher-Specific Head**: 투수 ID별 고유 잠재 벡터(Latent Vector)를 조건부 입력(Conditioning)으로 받아 개별 투수의 제구 경향을 세부 조정(Fine-tuning/Residual Modeling).

---

## 5. 아키텍처 비교 및 권장 파이프라인

| 접근 방식 | 핵심 장점 | 모델링 복잡도 | 주 활용 데이터/피처 |
| :--- | :--- | :---: | :--- |
| **물리 기반 DL** | 물리 법칙 일관성 확보, 도메인 해석력 극대화 | 중~상 | 릴리스 좌표, 무브먼트, 궤적 시계열 |
| **GNN** | 투수-타자 상성 및 볼카운트 맥락 관계 모델링 | 상 | 매치업 기록, 투구 시퀀스, 상황 노드 |
| **이상치 탐지** | 투구 폼 메커니즘 붕괴의 직접적 정량화 | 중 | 릴리스/회전축 다변량 분포, 오차 점수 |
| **메타 학습** | 신규/표본 부족 투수에 대한 빠른 적응력 | 상 | 투수별 Few-shot 투구 기록, 개인화 임베딩 |

### 권장 앙상블 파이프라인
1. **Feature Engineering**: 물리 궤적 방정식 + Autoencoder 메커니즘 이탈 점수($D_M$, Reconstruction Error) 생성
2. **Context Embedding**: GNN을 통해 투수-타자 상성 및 카운트 맥락 임베딩 추출
3. **Core Model**: FT-Transformer / Shared Backbone + Pitcher-Specific Adapter 구조로 최종 제구 성공 확률 예측
