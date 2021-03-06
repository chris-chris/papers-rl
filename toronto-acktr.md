# K-FAC 근사를 사용한 딥 강화학습을 위한 확장가능한 신뢰구간 방법론(ACKTR)

###Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation
```
Yuhuai Wu∗ University of TorontoVector Instituteywu@cs.toronto.edu
Elman Mansimov∗ 
New York University 
mansimov@cs.nyu.edu
Roger GrosseUniversity of Toronto Vector Institute rgrosse@cs.toronto.edu
Shun LiaoUniversity of Toronto Vector Institute sliao3@cs.toronto.edu
Jimmy BaUniversity of Toronto Vector Institute jimmy@psi.utoronto.ca
```
#Abstract
본 연구에서는 최근 제안 된 크론 커를 고려한 곡률 근사법을 이용한 심층적 인 강화 학습에 신뢰 영역 최적화를 적용 할 것을 제안합니다. 우리는 자연 정책 그라디언트의 틀을 확장하고 신뢰 지역과 함께 Kronecker-factored approximate curvature (K-FAC)를 사용하여 배우와 비평가 모두를 최적화하도록 제안합니다. 따라서 우리는 Kronecker-Factored Trust Region (ACKTR)을 사용하여 Actor critic 메서드를 호출합니다. 우리가 알고있는 한, 이것은 배우 - 비평 방식을위한 최초의 확장 가능한 트러스트 영역 자연 그라디언트 방법입니다. 또한 원시 픽셀 입력에서 직접 이산 제어 정책은 물론 연속 제어에서 중요하지 않은 작업을 학습하는 방법입니다. 우리는 MujeCo 환경에서 연속 도메인뿐만 아니라 Atari 게임의 이산 도메인에서 우리의 접근법을 테스트했습니다. 제안 된 방법을 사용하면 이전의 최첨단 온 - 정책 배우 - 비평 방식과 비교하여 평균적으로 표본 효율이 2 ~ 3 배 향상된 개선 효과를 얻을 수 있습니다. 코드는 https://github.com/openai/baselines에서 확인할 수 있습니다.

#1 Introduction

심층 강화 학습 (deep RL) 방법을 사용하는 에이전트는 복잡한 행동 기술을 학습하고 고차원의 원시 감각 상태 공간에서 어려운 작업을 해결하는 데 엄청난 성공을 보였습니다 [25, 18, 13]. Deep RL 방법은 심층 신경 네트워크를 사용하여 제어 정책을 나타냅니다. 인상적인 결과에도 불구하고, 이러한 신경 회로망은 확률 적 구배 강하 (SGD)의 간단한 변형을 사용하여 여전히 학습됩니다. SGD 및 관련 1 차 방법은 비효율적으로 중량 공간을 탐색합니다. 현재의 깊은 RL 방법이 다양한 연속 및 이산 제어 작업을 마스터하는 데 종종 며칠이 걸립니다. 이전에는 환경과 동시에 상호 작용할 수 있도록 여러 에이전트를 실행하여 교육 시간을 줄이기 위해 분산 된 접근 방식이 제안되었지만 병렬 처리 수준이 높아짐에 따라 샘플 효율이 빠르게 감소했습니다.
샘플 효율은 RL에서 지배적 인 관심사입니다. 실제 세계와의 로봇 상호 작용은 일반적으로 계산 시간보다 부족하며 시뮬레이션 환경에서도 시뮬레이션 비용이 종종 알고리즘 자체의 비용을 압도합니다. 샘플 크기를 효과적으로 줄이는 한 가지 방법은 그래디언트 업데이트에 고급 최적화 기술을 사용하는 것입니다. 자연 정책 그라디언트 [11]는 자연 그라디언트 디센트 [1]의 기술을 사용하여 그라디언트 업데이트를 수행합니다. 자연 그라디언트 방법은 피셔 메트릭을 기본 메트릭으로 사용하는 가장 가파른 하강 방향을 따릅니다.이 메트릭은 좌표 선택이 아닌 매니 폴드 (즉, 지표면)에 기반합니다.
그러나 자연 그라디언트의 정확한 계산은 피셔 정보 행렬을 반전해야하기 때문에 다루기가 어렵습니다. TRPO (Trust-region policy optimization) [22]는 피셔 - 벡터 곱을 사용하여 피셔 행렬을 명시 적으로 저장하고 반전하는 것을 방지합니다. 그러나 일반적으로 단일 매개 변수 업데이트를 얻기 위해 많은 단계의 공액 기울기가 필요하며 곡률을 정확하게 예측하려면 각 배치에서 많은 수의 샘플이 필요합니다. 따라서 TRPO는 대형 모델의 경우 실용적이지 못하고 샘플 비효율을 겪습니다.
크로네 커 - 인수 근사 곡률 (K-FAC) [16, 7]은 자연 그라디언트에 대한 확장 가능한 근사입니다. 더 큰 미니 배치를 사용하여 감독 학습에서 다양한 최첨단 대규모 신경 네트워크의 학습을 가속화하는 것으로 나타났습니다. TRPO와는 달리 각 업데이트는 SGD 업데이트와 비교할 수 있으며 곡률 정보의 평균을 유지하므로 작은 배치를 사용할 수 있습니다. 이는 정책 최적화에 K-FAC을 적용하면 현재의 심층 RL 방법의 표본 효율을 향상시킬 수 있음을 시사합니다.
이 논문에서는 actor-critic 방법을위한 확장 가능한 신뢰 영역 최적화 알고리즘 인 Kronecker-factored trust region (ACKTR; 알려진 "actor") 방법을 사용하여 actor-critic을 소개합니다. 제안 된 알고리즘은 그라디언트의 공분산 행렬 (covariance matrix)이 효율적으로 반전 될 수있게하는 자연스러운 정책 그라디언트 (natural policy gradient)에 대한 Kronecker-factored approximation을 사용합니다. 우리가 알고있는 한, Gauss-Newton 근사를 통해 가치 함수를 최적화하기 위해 자연 정책 기울기 알고리즘을 처음으로 확장했습니다. 실제로 ACKTR의 업데이트 당 계산 비용은 SGD 기반 방법보다 10 ~ 25 % 정도 높습니다. 실험적으로, 우리는 ACKTR이 Atari 환경 [4]과 MuJoCo [27] 작업에서 에이전트의 샘플 효율과 최종 성능 모두를 최첨단 온 - 정책 배우 - 비평 방식 인 A2C [18] 그리고 유명한 트러스트 영역 최적화 기인 TRPO [22].
소스 코드는 https://github.com/openai/baselines에서 온라인으로 사용할 수 있습니다. 

#2 배경

##2.1 강화 학습 및 배우 비평 방법

우리는 무한 수평선의 할인 된 Markov Decision Process와 상호 작용하는 에이전트를 고려합니다.
(X, A, γ, P, r). 시간 t에서, 에이전트는 정책 πθ (a | st)에 따라 ∈ A에서의 동작을 선택합니다.
그것의 현재 상태 st ∈ X. 환경은 차례로 보상 r (st, at)을 생성하고
전이 확률 P (s | s, a)에 따라 다음 상태 s로 진행합니다. 에이전트의 목표는 t + 1 t + 1 t t ∞ i입니다.
정책 파라미터 θ에 대해 예상 된 γ- 할인 누적 수익 J (θ) = Eπ [Rt] = Eπ [i≥0γr (st + i, at + i)]를 최대화합니다. 정책 기울기 방법 [30, 26]은 정책 πθ (a | st)를 직접 매개 변수화하고 객관적인 J (θ)를 최대화하도록 매개 변수 θ를 업데이트합니다. 일반적으로 정책 기울기는 [23]과 같이 정의된다.


여기서 Ψt는 주어진 상태에서 각 동작의 값의 상대 척도를 제공하는 이점 함수 Aπ (st, at)로 종종 선택된다. 낮은 분산과 낮은 바이어스 그래디언트 추정을 제공하는 우위 함수 설계에 대한 연구가 활발히 이루어지고있다 [23]. 이것이 우리의 작업의 초점이 아니기 때문에 비동기 우위 애호가 평론 (A3C) 방법 [18]을 따르고 함수 근사를 사용하여 k-step이 반환 할 때 우위 함수를 정의하기 만하면됩니다.

여기서 Vφπ (st)는 다음과 같은 보상의 예상 합계의 예상치를 제공하는 값 네트워크입니다.
정책 π에 주어진 주어진 상태, Vφπ (st) = Eπ [Rt]. 가치 네트워크의 매개 변수를 훈련 시키려면,
시간차 갱신을 수행하여 [18]을 다시 따라 간다. 그래서 제곱을 최소화합니다.
부트 스트랩 된 k-step 간의 차이는 Rt와 예측값 1 || Rt-V π (st) || 2를 반환합니다.

##2.2 Kronecker를 근사화 한 자연 그라디언트

비 응집 함수 J (θ)를 최소화하기 위해, 가장 가파른 하강의 방법은 || Δθ || B <1 인 제약 조건하에 J (θ + Δθ)를 최소화하는 업데이트 Δθ를 계산합니다. · B는
T1
|| x || B = (x Bx) 2에 의해 정의 된 표준이고, B는 양의 반정도 행렬입니다. 그 해결책은
제약 조건 최적화 문제는 다음과 같은 형태를 취한다 : Δθ α -B-1∇θJ, 여기서 ∇θJ는 표준 기울기입니다. 규범이 유클리드 (Euclidean), 즉, B = 1 일 때, 이는 일반적으로 사용되는 구배 강하 방법이된다. 그러나 유클리드 표준의 변화는 매개 변수화 θ에 달려 있습니다. 이는 모델의 매개 변수화가 임의적 인 선택이고 최적화 궤도에 영향을 주어서는 안되기 때문에 바람직하지 않습니다. 자연스러운 그래디언트의 방법은 피셔 정보 행렬 F를 사용하여 표준을 구성합니다.이 행렬은 KL 분기에 대한 로컬 2 차 근사입니다. 이 표준은 확률 분포의 클래스에 대한 모델 매개 변수화 θ와는 독립적이며보다 안정적이고 효과적인 업데이트를 제공합니다. 그러나 현대 신경망에는 수백만 개의 매개 변수가 포함될 수 있으므로 정확한 피셔 행렬 및 그 역을 계산하고 저장하는 것은 비실용적이므로 approximations에 의존해야합니다.
최근 제안 된 기법 인 Kronecker-factored approximate curvature (K-FAC) [16]은 피셔 행렬에 대한 크로네 커 인수 근사법을 사용하여 효율적인 대략적인 자연 그라디언트 업데이트를 수행합니다. 우리는 신경망의 출력 분포를 p (y | x)라고하고, L = log p (y | x)는 로그 우도를 나타낸다. W ∈ RCout × Cin을 l 번째 레이어의 가중치 행렬이라하자. 여기서 Cout과 Cin은 레이어의 출력 / 입력 뉴런 수입니다. ∈ RCin으로 입력 활성 벡터를 레이어에 나타내고 다음 레이어에 대한 사전 활성화 벡터를 s = Wa로 나타냅니다. 무게 기울기는 다음과 같이 주어진다 : ∇W L = (∇L) a Note. K-FAC은이 사실을 이용하고 층 (l)에 대응하는 블록 (F1)을 F1,

여기서 A는 E [aa⊺]를 나타내고 S는 E [∇sL (∇sL) ⊺]를 나타냅니다. 이 근사값은 활성화와 역 전파 파생 정보의 2 차 통계가 상관 관계가 없다는 가정을하는 것으로 해석 할 수 있습니다. 이 근사화를 통해, 자연 구배 업데이트는 다음과 같은 식으로 자연 상태 기울기를 효율적으로 계산할 수 있습니다. P≡Q - 1 = P - 1niQ - 1 및 (P⊗Q) vec (T) = PTQ⊺ :


위의 방정식으로부터 K-FAC 근사 자연 기울기 업데이트는 W와 크기가 비슷한 행렬에 대한 계산 만 필요로한다는 것을 알 수 있습니다. Grosse and Martens [7]는 최근에 길쌈 네트워크를 다루기 위해 K-FAC 알고리즘을 확장했습니다. Ba 등 [2]는 나중에 비동기식 계산을 통해 오버 헤드의 대부분이 완화되는 방법의 분산 버전을 개발했습니다. 분산 K-FAC은 대규모 현대 분류 길쌈 네트워크를 학습 할 때 2 배에서 3 배의 속도 향상을 달성했습니다.

3 방법
3.1 배우 비평가의 자연스러운 그라디언트
자연 그라디언트는 10 년 전에 Kakade [11]에 의해 정책 그라디언트 방법에 적용하기 위해 제안되었다. 그러나 자연 정책 그라디언트의 확장 성, 샘플 효율성 및 범용 인스턴스화는 여전히 존재하지 않습니다. 이 섹션에서는 배우 - 비평 방식에 대한 최초의 확장 가능하고 샘플 효율적인 자연 기울기 알고리즘 인 Kronecker-factored trust region (ACKTR) 방법을 사용하는 배우 평론가를 소개합니다. 우리는 자연 그라디언트 업데이트를 계산하기 위해 Kronecker 인수 근사법을 사용하고 배우와 비평가 모두에게 자연 그라디언트 업데이트를 적용합니다.
강화 학습 목표를위한 피셔 메트릭을 정의하기 위해서는 현 상태에서 주어진 액션에 대한 분포를 정의하는 정책 함수를 사용하고 궤적 분포에 대한 기대치를 취하는 것이 당연한 선택입니다.

여기서 p (τ)는 p (s0) Tt = 0 π (at | st) p (st + 1 | st, at)에 의해 주어진 궤도의 분포입니다. 실제로,
훈련 중에 수집 된 궤적에 대한 난해한 기대치를 근사화합니다.
지금 우리는 비평가를 최적화하기 위해 자연스러운 그라디언트를 적용하는 한 가지 방법을 설명합니다. 평론가를 배우는 것은 움직이는 표적을 가지고 있지만 최소 자승 함수 근사 문제로 생각할 수 있습니다. 최소 자승 함수 근사화의 설정에서 선택 2 차 알고리즘은 Gauss-Newton 행렬 Gauss-Newton 행렬 G : = E [JTJ] (여기서 J는 매개 변수를 출력 [19]. Gauss-Newton 행렬은 Gaussian 관측 모델 [15]에 대한 피셔 행렬과 같습니다. 이 동등성은 K-FAC을 평론가에게도 적용 할 수있게합니다. 특히 비평가 v의 출력은 가우스 분포 p (v | st) ~ N (v; V (st), σ2)로 정의된다고 가정합니다. 평론가를위한 피셔 행렬은이 가우시안 출력 분포와 관련하여 정의됩니다. 실제로, σ를 1로 설정하면 바닐라 가우스 - 뉴튼 방법과 동일합니다.
배우와 평론가가 서로 얽히지 않으면 위에 정의 된 측정 항목을 사용하여 K-FAC 업데이트를 개별적으로 적용 할 수 있습니다. 하지만 교육의 불안정성을 피하기 위해 두 네트워크가 하위 계층 표현을 공유하지만 별개의 출력 계층을 갖는 아키텍처를 사용하는 것이 종종 도움이됩니다 [18, 28]. 이 경우, 우리는 두 개의 출력 분포, 즉 p (a, v | s) = π (a | s) p (v | s)의 독립성을 가정함으로써 정책과 가치 분포의 공동 분포를 정의 할 수있다. 네트워크의 출력을 독립적으로 샘플링해야한다는 점을 제외하면 표준 K-FAC과 다르지 않은 p (a, v | s)와 관련하여 피셔 미터법을 구성합니다. Fisher 행렬 Ep (τ) [∇ log p (a, v | s) ∇ log p (a, v | s) T]를 근사하여 K-FAC을 적용하여 동시에 업데이트를 수행 할 수있다.
또한 [16]에 설명 된 인수 분해 Tikhonov 댐핑 접근법을 사용합니다. 우리는 또한 [2]를 따르고 계산 시간을 줄이기 위해 크로네 커 근사법에 필요한 2 차 통계 및 비 반전의 비동기 계산을 수행합니다.

3.2 단계 크기 선택 및 신뢰 영역 최적화
전통적으로 자연 그라디언트는 SGD와 유사한 업데이트 인 θ ← θ - ηF - 1∇θL로 수행됩니다. 그러나 깊은 RL의 맥락에서 Schulman et al. [22] 그러한 업데이트 규칙은 정책을 대규모로 업데이트 할 수 있으므로 알고리즘이 조기에 가까운 결정 성있는 정책으로 수렴되는 것을 발견했습니다. 그들은 트러스트 영역 접근 방식을 사용하는 대신 정책 배포를 KL 분산 (divergence) 측면에서 지정된 양만큼 수정하도록 업데이트 규모를 축소합니다. 따라서, [2]에 의해 도입 된 K-FAC의 신뢰 영역 공식을 채택하고 유효 스텝 크기 η를
 분 (η
, 2δ), 여기서 학습률 η max Δθ⊺ FΔθ max
신뢰 영역 반경 δ는 하이퍼 파라미터입니다. 만약
 배우와 비평가가 서로 얽혀 있지 않으면, 우리는 ηmax와 δ의 다른 세트를 둘 다 따로 따로 튜닝해야합니다. 비평가 출력 분포에 대한 분산 매개 변수는 바닐라 가우스 - 뉴턴에 대한 학습 속도 매개 변수로 흡수 될 수 있습니다. 다른 한편, 만약 그들이 표현을 공유한다면, 우리는 ηmax, δ의 한 세트와 비평가의 훈련 손실에 대한 가중치 매개 변수를 배우의 그것과 관련하여 조정할 필요가있다.
 
#4 관련 연구

자연 그라디언트 [1]는 Kakade [11]에 의해 정책 기울기 방법에 처음 적용되었다. Bagnell과 Schneider [3]는 [11]에서 정의 된 메트릭이 경로 - 분포 매니 폴드에 의해 유도 된 공변량 메트릭임을 더 증명했습니다. Peters and Schaal [20]은 배우 비평 알고리즘에 자연스러운 그라데이션을 적용했습니다. 그들은 배우의 업데이트에 대한 자연스러운 정책 그라디언트를 수행하고 비평가의 업데이트를 위해 LSTD (least-squares temporal difference) 방법을 사용하도록 제안했습니다. 그러나 자연 그라디언트 방법을 적용 할 때 큰 어려움이 있습니다. 주로 피셔 매트릭스를 효율적으로 저장하는 것은 물론 역수를 계산하는 것과 관련이 있습니다. 다루기 쉽도록 이전의 연구에서는이 방법을 호환 함수 근사 (선형 함수 근사법)를 사용하도록 제한했습니다. TRPO (Trust Region Policy Optimization) [22]는 계산상의 부담을 피하기 위해 Martens [14]의 작업과 유사한 고속 피셔 행렬 - 벡터 곱을 사용하여 공액 그라데이션을 사용하는 선형 시스템을 대략적으로 해결합니다. 이 접근법에는 두 가지 주요 단점이 있습니다. 첫째, Fisher 벡터 제품을 반복 계산해야하므로 Atari 및 MuJoCo의 이미지 관측에서 학습하는 데 사용되는 대형 아키텍처로 스케일링되지 않습니다. 둘째, 곡률을 정확하게 추정하기 위해서는 대량 배치가 필요합니다. K-FAC는 다루기 쉬운 피셔 매트릭스 근사법을 사용하고 교육 도중 곡률 통계의 평균을 유지함으로써 두 가지 문제를 피합니다. TRPO는 Adam [12]과 같은 1 차 옵티 마이저로 교육 된 정책 기울기 방법보다 더 나은 반복 당 진행률을 보여 주지만 일반적으로 샘플 효율이 떨어집니다.
TRPO의 계산 효율을 향상시키기위한 몇 가지 방법이 제안되었다. 피셔 - 벡터 곱을 반복적으로 계산하는 것을 피하기 위해, Wang et al. [28] 정책 네트워크의 실행 평균과 현재 정책 네트워크 사이의 KL 발산의 선형 근사치로 제약 된 최적화 문제를 푸는 것. 신뢰 영역 최적화기에 의해 부과 된 하드 제한 대신 Heess et al. [9] 그리고 Schulman et al. [24]는 목적 함수에 KL 비용을 소프트 제약으로 추가했습니다. 두 논문 모두 샘플 효율성 측면에서 연속 및 이산 제어 작업에 대한 바닐라 정책 기울기보다 약간 개선 된 점을 보여줍니다.
경험 재생 [28], [8] 또는 보조 목표 [10]를 도입하여 표본 효율을 향상시킨 최근에 발표 된 다른 배우 - 비평가 모델이 있습니다. 이러한 접근법은 우리 연구와 직각을 이루며, 샘플 효율을 더욱 향상시키기 위해 ACKTR과 잠재적으로 결합 될 수있다.

#5 Experiments

우리는 다음과 같은 문제를 조사하기 위해 일련의 실험을 수행했습니다. (1) ACKTR은 샘플 효율성 및 계산 효율성과 관련하여 최첨단 온 온 정책 방법 및 일반적인 2 차 옵티 마이저 기준선과 어떻게 비교 되는가? (2) 비평가의 최적화를 위해 더 나은 표준을 만드는 것은 무엇입니까? (3) ACKTR의 성능은 1 차 방법에 비해 배치 크기에 따라 어떻게 달라 집니까?
우리는 제안 된 방법 인 ACKTR을 2 개의 표준 벤치 마크 플랫폼에서 평가했습니다. 먼저 이산 제어를위한 심층적 학습 벤치 마크로 사용되는 Atari 2600 게임의 시뮬레이터 인 Arcade Learning Environment [4]에 의해 시뮬레이션 된 OpenAI Gym [5]에 정의 된 개별 제어 작업에 대해 평가했습니다. 그런 다음 다양한 연속 제어로 평가했습니다.
MuJoCo [27] 물리 엔진에 의해 시뮬레이션 된 OpenAI Gym [5]에서 정의 된 벤치 마크 작업. 우리의베이스 라인은 (a) 비동기 우위 애호가 비평 모델 (A3C) [18]의 동기식 및 일괄 버전, 이후 A2C (advantage actor critic) 및 (b) TRPO [22]입니다. ACKTR과베이스 라인은 Atari 게임의 TRPO베이스 라인을 제외하고는 동일한 모델 아키텍처를 사용합니다.이 아키텍처는 공액 그래디언트 내부 루프를 실행하는 컴퓨팅 부하로 인해 더 작은 아키텍처를 사용하는 것으로 제한됩니다. 다른 실험 세부 사항은 부록을 참조하십시오.

##5.1 이산 제어
먼저 ACKTR에서 얻은 성능 향상을 측정하기 위해 표준 6 개의 Atari 2600 게임에 대한 결과를 제시합니다. 1 천만 개의 타임 스텝에 대해 훈련 된 6 개의 아타리 게임에 대한 결과가 그림 1에 나와 있으며 A2C 및 TRPO2와 비교됩니다. ACKTR은 모든 게임에서 표본 효율 (즉, 타임 스텝 수에 따른 수렴 속도) 측면에서 A2C보다 월등히 뛰어났습니다. 우리는 TRPO가 Seaquest와 Pong이라는 2 개의 게임을 1 천만 개의 타임 스 텝으로 만 배울 수 있으며 샘플 효율면에서 A2C보다 나빴던 것으로 나타났습니다.
표 1에서 우리는 인간의 성과를 달성하는 데 필요한 에피소드의 수와 5 천만 개의 타임 스텝을위한 훈련에서 마지막 100 개의 에피소드의 보상 평균을 제시합니다 [17]. 특히 Beamrider, Breakout, Pong 및 Q-bert 게임에서 A2C는 인간 성능을 달성하기 위해 ACKTR보다 각각 2.7, 3.5, 5.3 및 3.0 배 더 많은 에피소드를 요구했습니다. 또한 Space Invaders에서 A2C의 실행 중 하나가 사람의 성능과 일치하지 못했지만 ACKTR은 1972 년에 평균적으로 인간의 성능 (1652 회)보다 12 배 뛰어났습니다. Breakout, Q-bert 및 Beamrider에서 ACKTR은 A2C보다 26 %, 35 % 및 67 % 더 큰 에피소드 보상을 획득했습니다.
나머지 Atari 게임에서도 ACKTR을 평가했습니다. 전체 결과는 부록 B를 참조하십시오. ACKTR과 Q-learning 방법을 비교 한 결과 44 개의 벤치 마크 중 36 개에서 ACKTR은 Q- 학습 방법과 동등한 것으로 나타 났으며 샘플 효율 측면에서 훨씬 적은 계산 시간을 소비했습니다. 놀랍게도, 아틀란티스 게임에서 ACKTR은 그림 2와 같이 1.3 시간 (600 회)에 2 백만의 보상을 얻는 법을 빨리 배웠습니다. A2C는 동일한 성능 수준에 도달하기 위해 10 시간 (6000 회)을 소비했습니다.

##5.2 지속적인 제어
우리는 MuJoCo [27]에서 시뮬레이션 된 OpenAI Gym [5]에서 정의 된 연속 제어 태스크의 표준 벤치 마크에서 저 차원 상태 공간 표현과 픽셀에서 직접 실험을 실행했습니다. Atari와 달리, 연속적인 제어 작업은 고차원의 작업 공간 및 탐색으로 인해 때로는 더 어려워집니다. 우리 모델은 8 개의 MuJoCo 작업 중 6 개 작업에서 기준선보다 현저히 뛰어난 성능을 보였으며 다른 두 가지 작업 (Walker2d 및 Swimmer)에서는 A2C와 경쟁적으로 수행되었습니다.
우리는 8 건의 MuJoCo 과제에 대한 3 천만 개의 타임 스텝에 대한 ACKTR을 평가했으며, 표 2에서는 훈련에서 상위 10 개의 연속 에피소드의 평균 보상과 [8]에서 정의 된 특정 임계 값에 도달하는 에피소드의 수를 제시합니다. 표 2에 나타난 바와 같이, TRPO가 시료 효율의 4.1 배에 달하는 수영을 제외하고는 모든 작업에서 ACKTR이 지정된 임계 값에 더 빨리 도달합니다. 특히 눈에 띄는 사례는 Ant입니다. 여기서 ACKTR은 TRPO보다 16.4 배 더 효율적입니다. 평균 보상 점수는 3 가지 모델 모두 Walker2d 환경에서 10 % 향상된 보상 점수를 얻는 TRPO를 제외하고는 서로 비교할만한 결과를 얻습니다.
우리는 또한 저 차원 상태 공간을 입력으로 제공하지 않고 픽셀에서 직접 연속 제어 정책을 배우려고했습니다. 픽셀에서 연속 제어 정책을 배우는 것은 Atari에 비해 렌더링 시간이 느려지므로 상태 공간에서 학습하는 것보다 훨씬 어려울 수 있습니다 (MuJoCo에서 0.5 초, Atari에서 0.002 초). 최첨단 배우 비평 방식 인 A3C [18]는 진자, Pointmass2D 및 그리퍼와 같은 비교적 단순한 작업의 픽셀 결과 만보고했습니다. 그림 4에서 볼 수 있듯이 우리 모델은 4 천만 회의 타임 스텝 동안 교육 한 후 최종 에피소드 보상 측면에서 A2C보다 월등히 뛰어납니다. 보다 구체적으로 Reacher, HalfCheetah 및 Walker2d에서 우리 모델은 A2C에 비해 최종 보상이 1.6, 2.8 및 1.7 배 증가했습니다. 픽셀의 숙련 된 정책 동영상은 https : //www.youtube.com/watch?v=gtM87w1xGoM에서 확인할 수 있습니다. 사전 조율 된 모델 무게는 https : //github.com/emansim/acktr에서 확인할 수 있습니다.

##5.3 비평가 최적화를위한 더 나은 규범?
이전의 자연 방침 그라디언트 방법은 액터에만 자연 그라디언트 업데이트를 적용했습니다. 우리의 작업에서 우리는 비평가에게 자연스러운 그라디언트 업데이트를 적용 할 것을 제안합니다. 그 차이점은 우리가 평론가에게 가장 가파른 강하를 수행하기로 선택한 표준에있다. 즉, 표준 || · 2.2 절에서 정의 된 B || B. 이 섹션에서, 우리는 액터에 ACKTR을 적용하고, 비평 최적화를 위해 ACKTR (즉, Gauss-Newton에 의해 정의 된 노름)을 사용하여 1 차 방법 (즉, 유클리드 기준)을 사용하여 비교했습니다. 그림 5 (a)와 (b)는 HalfCheetah와 Atari 게임 브레이크 아웃의 연속 제어 작업 결과를 보여줍니다. 우리는 비평가를 최적화하기 위해 어떤 규범을 사용하는지에 관계없이 기본 A2C와 비교하여 액터에 ACKTR을 적용하여 개선 된 점이 있음을 확인합니다.

그러나 비평가를 최적화하기 위해 Gauss-Newton 규범을 사용함으로써 얻은 개선은 훈련 종료 시점의 표본 효율성 및 에피소드 보상면에서 훨씬 중요합니다. 또한 Gauss-Newton 규범은 유클리드 표준을 사용하는 무작위 종자보다 큰 결과를 관찰 할 때 훈련을 안정시키는 데에도 도움이됩니다.
평론가를위한 피셔 행렬은 비평가의 출력 분포, 분산 σ를 갖는 가우시안 분포를 사용하여 구성된다는 것을 상기하자. vanilla Gauss-Newton에서 σ는 1로 설정됩니다. 우리는 회귀 분석에서 잡음의 분산을 추정하는 것과 유사한 Bellman 오차의 분산을 사용하여 σ를 추정하는 것을 실험했습니다. 우리는이 방법을 적응 형 Gauss-Newton이라고 부릅니다. 그러나 적응 형 Gauss-Newton은 바닐라 Gauss-Newton보다 큰 개선점을 제공하지 못합니다. (부록 D에있는 σ의 선택에 대한 자세한 비교를보십시오).
5.4 ACKTR은 벽시계에서 A2C와 어떻게 비교됩니까?
우리는 벽 시계 시간으로 ACKTR을 기준선 A2C와 TRPO와 비교했습니다. 표 3은 6 개의 Atari 게임과 8 개의 MuJoCo (주 공간) 환경에서 초당 평균 시간 간격을 보여줍니다. 결과는 이전 실험과 동일한 실험 설정으로 얻어집니다. MuJoCo 작업에서 에피소드는 순차적으로 처리되지만 Atari 환경에서는 에피소드가 병렬로 처리됩니다. 따라서 더 많은 프레임이 Atari 환경에서 처리됩니다. 표에서 알 수 있듯이 ACKTR은 계산 시간을 타임 스텝 당 최대 25 % 만 증가시켜 큰 최적화 이점으로 실용성을 입증합니다.

##5.5 ACKTR과 A2C는 다른 배치 크기로 어떻게 수행됩니까?
대규모 분산 학습 설정에서는 큰 배치 크기가 최적화에 사용됩니다. 그러므로, 이러한 환경에서, 배치 크기로 잘 확장 할 수있는 방법을 사용하는 것이 바람직하다. 이 섹션에서는 서로 다른 배치 크기에 대해 ACKTR 및 기본 A2C가 수행하는 작업을 비교합니다. 우리는 배치 크기가 160 및 640 인 것을 실험했습니다. 그림 5 (c)는 시간 단계별 수당 보상을 보여줍니다. 우리는 더 큰 일괄 처리 크기를 가진 ACKTR이 더 작은 일괄 처리 크기로 수행되었음을 발견했습니다. 그러나,보다 큰 배치 크기의 경우, A2C는 시료 효율 측면에서 현저한 저하를 경험했습니다. 이것은 그림 5 (d)의 관찰과 일치하며, 여기서 우리는 업데이트 횟수의 관점에서 훈련 곡선을 그렸다. 우리는 A2C에 비해 ACKTR이 더 큰 배치 크기를 사용할 때 이점이 상당히 증가한다는 것을 알 수 있습니다. 이는 큰 설정을 사용해야하는 분산 환경에서 ACKTR을 사용하여 큰 속도 향상을 가져올 가능성이 있음을 시사합니다. 이것은 [2]의 관찰과 일치합니다.
#6 결론
이 연구에서 우리는 심층적 인 강화 학습을 위해 표본 효율적이고 계산 비용이 저렴한 신뢰 영역 최적화 방법을 제안했습니다. 안정성을 위해 트러스트 영역 최적화를 사용하여 액터 비평 방법에 대한 자연 그라디언트 업데이트를 근사하기 위해 최근에 제안 된 K-FAC이라는 기술을 사용했습니다. 우리가 알고있는 한, 우리는 자연 그라디언트 업데이트를 사용하여 배우와 평론가를 모두 최적화하도록 제안한 첫 번째 사례입니다. 우리는 Atari 게임과 MuJoCo 환경에서 우리의 방법을 테스트했으며, 1 차 구배 법 (A2C)과 반복적 2 차 법 (TRPO)과 비교하여 평균적으로 표본 효율이 2 ~ 3 배 향상된다는 것을 관찰했습니다. . 우리 알고리즘의 확장 성 때문에, 우리는 원시 픽셀 관측 공간에서 연속적으로 제어하는 ​​몇 가지 중요하지 않은 작업을 처음으로 훈련합니다. 이것은 강화 학습에서 Kronecker를 고려한 자연 구배 근사치를 다른 알고리즘으로 확장하는 것이 유망한 연구 방향이라고 제안합니다.
감사 인사
우리는 OpenAI 팀에게 기본 결과와 Atari 환경 선처리 코드 제공에 대한 관대 한 지원에 감사드립니다. 또한 John Schulman에게 도움이되는 토론에 감사드립니다.