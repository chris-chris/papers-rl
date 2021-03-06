# DEEPREINFORCEMENTLEARNING INPARAMETER-IZEDACTIONSPACE

Matthew Hausknecht

Department of Computer Science University of Texas at Austinmhauskn@cs.utexas.edu

Peter Stone

Department of Computer Science University of Texas at Austinpstone@cs.utexas.edu

# ABSTRACT

최근의 연구는 심층 신경망이 지속적인 상태 및 행동 공간을 특징으로하는 강화 학습 영역에서 가치 기능과 정책을 근사 할 수 있음을 보여주었습니다. 그러나 우리가 아는 한, 구조화 된 \(매개 변수화 된\) 연속 동작 공간에서 심 신경 네트워크를 사용하는 데 이전의 연구가 성공하지 못했습니다. 이러한 차이를 보완하기 위해이 백서에서는 시뮬레이트 된 RoboCup 축구의 영역 내에서 Actor는 것에 중점을 둡니다.이 게임에는 연속 변수로 매개 변수화 된 작은 유형의 개별 동작 유형이 있습니다. 최고의 학습자는 2012 RoboCup 챔피언 에이전트보다 더 안정적으로 목표를 기록 할 수 있습니다. 이와 같이,이 논문은 매개 변수화 된 행동 공간 MDP의 클래스에 대한 심층적 인 강화 학습의 성공적인 확장을 나타낸다.

# 1. Introduction

이 논문은 DDPG \(Deep Deterministic Policy Gradients\) 알고리즘 \(Lillicrap et al., 2015\)을 매개 변수화 된 작업 공간으로 확장합니다. 우리는 DDPG 알고리즘의 공개 버전에 대한 수정, 즉 바운딩 액션 공간 구배를 문서화합니다. 우리는이 영역에서 안정적인 학습을 위해 필요한 수정을 발견했으며, 미래의 실무자가 지속적이고 제한적인 행동 공간에서 학습하려고 시도 할 때 유용 할 것입니다.

우리는 골을 넣을 수있는 RoboCup 축구 정책을 처음부터 믿을 수있는 방법으로 학습합니다. 이러한 정책은 낮은 수준의 연속 된 상태 공간과 매개 변수화 된 연속 동작 공간에서 작동합니다. 에이전트는 하나의 보상 기능을 사용하여 공을 찾고 접근하고 목표에 드리블을 부여하며 빈 목표에 점수를 매기는 방법을 학습합니다. 가장 잘 배운 에이전트는 핸드 코딩 된 2012 RoboCup 챔피언보다 느린 속도로 득점 목표에서 더 신뢰할 수 있음을 증명합니다.

RoboCup 2D HFO \(Half-Field-Offense\)는 단일 상담원 학습, 다중 상담원 학습 및 임시 팀 워크를 연구하기위한 연구 플랫폼입니다. HFO는 낮은 수준의 연속적인 상태 공간과 매개 변수화 된 연속 동작 공간을 특징으로합니다. 특히, 매개 변수화 된 작업 공간은 에이전트가 먼저 상위 수준 작업의 개별 목록에서 수행하려는 작업 유형을 선택한 다음 해당 작업을 수반 할 연속 매개 변수를 지정해야합니다. 이 매개 변수화는 순전히 연속적인 동작 공간에서 찾을 수없는 구조를 도입합니다.

이 논문의 나머지 부분은 다음과 같이 구성되어있습니다. HFO 영역은 2 절에서 제시된다. 3 절에서는 상세한 Actor와 Critic 업데이트를 포함한 심층적 인 지속적인 강화 학습에 대한 배경을 제시합니다. 5 장은 행동 공간 구배를 제한하는 방법을 제시합니다. 6 장에서는 실험과 결과를 다룹니다. 마지막으로, 관련 연구는 8 절에서 결론을 제시합니다.


# 2 HALF FIELD OFFENSE DOMAIN

RoboCup은 인공 지능과 로봇에 대한 연구를 촉진시키는 국제 로봇 축구 대회입니다. RoboCup 내에서 2D 시뮬레이션 리그는 플레이어, 공 및 필드가 모두 2 차원 개체 인 축구 추상화 작업을 수행합니다. 그러나 다른 알고리즘을 신속하게 프로토 타이핑하고 평가하려는 연구원의 경우 전체 축구 과제는 시간이 많이 걸리고 결과에 편차가 많으며 프리킥 및 오프사이드와 같은 규칙을 특수하게 처리해야한다는 번거 로움이 있습니다.
하프 필드 오펜스 (Half Field Offense) 도메인은 전체 RoboCup의 어려움을 추상화하고 실험자를 핵심 의사 결정 논리에만 노출시키고 RoboCup 2D 게임의 가장 도전적인 부분 인 득점 및 방어 목표에 중점을 둡니다. HFO에서 각 에이전트는 자체 상태 감각을 수신하고 독립적으로 자체 동작을 선택해야합니다. HFO는 자연스럽게 에피소드 멀티 에이전트 POMDP로 특징 지어집니다. 왜냐하면 에이전트의 부분적으로 연속적인 부분 관찰 및 동작과 점수가 매겨진 목표 또는 플레이 영역을 빠져 나가는 공에서 절정에 달하는 잘 정의 된 에피소드 때문입니다. 각 에피소드를 시작하기 위해 에이전트와 볼은 필드의 공격적인 절반에 무작위로 배치됩니다. 에피소드는 목표 점수가 매겨 지거나 공이 필드를 떠날 때 또는 500 타임 스텝이 지나면 끝납니다. Half Field Offense 게임의 예는 https://vid.me/sNev https://vid.me/JQTw https://vid.me/1b5D에서 볼 수 있습니다. 다음 하위 섹션에서는이 도메인에서 에이전트가 사용하는 하위 수준 상태 및 작업 공간을 소개합니다.

##2.1 스테이트 스페이스

에이전트는 연속적으로 평가되는 58 개의 기능을 사용하여 인코딩 된 낮은 수준의 중심 중심 관점을 사용합니다. 이러한 기능은 헬리오스 - Agent2D의 (아키야마, 2010) 세계 모델을 통해 도출 및 공, 목표, 다른 플레이어와 같은 중요성을 다양한에 필드 객체 각도와 거리를 제공합니다. 그림 1은 에이전트의 인식을 나타냅니다. 가장 관련성이 높은 기능은 다음과 같습니다. 에이전트의 위치, 속도 및 방향, 스테미너. 에이전트가 걷어차는 경우 표시기. 볼, 골, 필드 코너, 페널티 박스 코너, 팀 멤버 및 상대방에게 주어진 각도와 거리. 상태 기능의 전체 목록은 https://github.com/mhauskn/ HFO / blob / master / doc / manual.pdf에서 확인할 수 있습니다.

##2.2 행동 공간

Half Field Offense는 낮은 수준의 매개 변수화 된 작업 공간을 제공합니다. Dash, Turn, Tackle 및 Kick이라는 4 개의 상호 배타적 인 이산 조치가 있습니다. 각 타임 스텝에서 상담원은이 네 가지 중 하나를 선택하여 실행해야합니다. 각 동작에는 1-2 개의 연속 값 매개 변수도 지정해야하며이 매개 변수도 지정해야합니다. 에이전트는 실행하고자하는 개별 동작과 해당 동작에 필요한 연속적으로 가치있는 매개 변수를 모두 선택해야합니다. 매개 변수가있는 작업의 전체 집합은 다음과 같습니다.
대쉬 (힘, 방향) : [0, 100]의 스칼라 힘으로 지시 된 방향으로 움직입니다. 움직임은 옆이나 뒤로보다 빨리 진행됩니다. 방향 전환 : 지시 된 방향으로 회전합니다. 태클 (방향) : 지시 된 방향으로 움직여 볼을 컨테스트합니다. 이 동작은 상대방과 경기 할 때만 유용합니다. 킥 (파워, 방향) : [0, 100]의 스칼라 파워로 지시 된 방향으로 볼을 걷어냅니다. 모든 방향은 [-180, 180]의 범위에서 매개 변수화됩니다.

##2.3 보상 신호

HFO 영역에서의 진정한 보상은 완전한 게임에서 승리 한 것입니다. 그러나, 그러한 보상 신호는 학습자가 견인력을 얻기에는 너무 희박하다. 대신에 우리는 4 가지 구성 요소로 손으로 만들어진 보상 신호를 소개합니다. Move To Ball Reward는 에이전트와 공 간의 거리 변화에 비례하는 스칼라 보상 d (a, b)를 제공합니다. 에이전트가 공을 걷어차기에 충분할 때마다 에피소드 1 회 추가 보상이 주어집니다. 킥오프 목표 리워드는 볼과 목표의 중심 사이의 거리 변화 d (b, g)에 비례합니다. 추가 골은 Igoal 골을 넣는데 주어집니다. 이러한 구성 요소의 가중치 합계를 사용하면 상담원이 먼저 공을 걷어차기에 충분히 접근 한 다음 목표를 향해 걷어 찬 결과를 보상하고 마지막으로 점수를 얻는 단일 보상이 제공됩니다. 공이 에이전트에서 멀어 질수록 move-to-ball 구성 요소는 각 킥 바로 뒤에 네거티브 보상을 생성하기 때문에 보상의 킥 - 투 - 목표 구성 요소에 더 높은 이득을 제공해야합니다. 전반적인 보상은 다음과 같습니다 :


보상 공학이 필요하다는 것은 실망 스럽습니다. 그러나 탐사 작업은 득점 득점만으로 이루어진 보상에 대한 견인력을 얻는 것은 너무 어렵습니다. 무작위로 행동하면 합리적인 시간 내에 단일 목표를 산출 할 가능성이 매우 낮기 때문입니다. 미래의 작업을위한 흥미로운 방향은 큰 주 공간을 탐사하는 더 나은 방법을 찾는 것입니다. 이 방향에 대한 하나의 최근 접근법 인 Stadie et al. (2015)는 시스템 역학 모델을 기반으로 탐사 보너스를 할당했습니다.

#3 BACKGROUND: DEEP REINFORCEMENT LEARNING

딥 뉴럴 네트워크 (deep neural networks)는 감독 학습 (supervised learning) 작업에서 가장 널리 사용되는 숙달 된 범용 함수 근사 (approximate function approximators)입니다. 그러나 최근에는 보강 학습 문제에 적용되어 심층적 인 보강 학습 분야가 생겨났습니다. 이 분야는 심층 신경 네트워크의 진보와 보강 학습 알고리즘을 결합하여 복잡한 환경에서 지능적으로 행동 할 수있는 에이전트를 만듭니다. 이 섹션에서는 연속 행동 공간에서의 심화 학습 학습에 대한 배경을 제시합니다. 표기법은 Lillicrap et al. (2015).

이산 행동 공간에서 모델이없는 깊은 RL은 Mnih 등이 소개 한 Deep Q-Learning 방법을 사용하여 수행 할 수 있습니다. (2015)는 하나의 딥 네트워크 (deep network)를 사용하여 각각의 개별 행동의 가치 함수를 평가하고, 행동 할 때 주어진 상태 입력에 대해 최대로 가치가있는 출력을 선택합니다. DQN의 몇 가지 변형이 탐구되었습니다. Narasimhan et al. (2015)는 붕괴 흔적을 사용했고, Hausknecht & Stone (2015)은 LSTM 재발 성을 조사하였고 van Hasselt et al. (2015)는 이중 Q-Learning을 탐구했다. 이러한 네트워크는 연속적인 상태 공간에서 잘 작동하지만 네트워크의 출력 노드는 연속적으로 연속 동작이 아닌 Q 값 추정치를 출력하도록 훈련되므로 연속 동작 공간에서는 작동하지 않습니다.

Actor / Critic 아키텍처 (Sutton & Barto, 1998)는 가치 학습과 행동 선택을 분리함으로써이 문제에 대한 하나의 해결책을 제시합니다. 두 개의 심 신경 네트워크를 사용하여 표현 된 Actor 네트워크는 Critic가 가치 기능을 평가하는 동안 지속적인 행동을 출력합니다. θμ에 의해 매개 변수화 된 액터 네트워크 μ는 상태 s를 입력으로 취하여 연속적인 동작을 출력합니다. θQ에 의해 매개 변수화 된 비평 네트워크 Q는 상태 s와 동작 a를 입력으로 받아 스칼라 Q 값 Q (s, a)를 출력합니다. 그림 2는 Critic 및 Actor 네트워크를 보여줍니다.

평론 네트워크에 대한 업데이트는 원래 Q-Learning에서 사용 된 표준 시간차 업데이트 (Watkins & Dayan, 1992) 및 이후에 DQN에서 변경되지 않았습니다.

이 방정식을 위에서 설명한 신경 네트워크 설정에 적용하면 다음과 같이 정의 된 손실 함수가 최소화됩니다.

그러나 연속 동작 공간에서이 방정식은 다음 상태 동작을 최대화하는 것과 관련하여 더 이상 다루기가 쉽지 않습니다. 대신 Actor 네트워크에 다음 상태 액션을 제공하도록 요청합니다. = μ (s '| θμ). 이것은 다음과 같은 형식으로 평론가의 손실을 가져옵니다.

평론가의 가치 함수는 θQ에 대한이 손실 함수의 기울기 강하에 의해 학습 될 수있습니다. 그러나 액터가 업데이트 대상에서 다음 상태 액션을 결정하기 때문에이 값 함수의 정확도는 액터 정책의 품질에 크게 영향을받습니다.
Critic의 행동 가치에 대한 지식은 Actor에 대한 더 나은 정책을 Actor기 위해 활용됩니다. 샘플 상태가 주어지면 액터의 목표는 현재 출력 a와 그 상태 a *에서의 최적 동작 간의 차이를 최소화하는 것입니다.

Critic는 다양한 행동의 질에 대한 견적을 제공하기 위해 사용될 수 있지만 순진한 a *를 추정하는 것은 모든 가능한 행동에 대한 Critic의 결과를 극대화하는 것을 포함한다 : a * ≈ arg maxa Q (s, a | θQ). 전지구 적 최대치를 찾는 대신, Critic 네트워크는 행동 공간에서 변화 방향을 나타내는 점수를 제공하여 높은 Q- 값 추정을 유도 할 수있다 : ∇aQ (s, a | θQ). 이러한 그라디언트를 얻으려면 연속 동작 공간에서 최적화 문제를 해결하는 것보다 훨씬 빠르게 비평 네트워크를 한 번 뒤로 통과해야합니다. 이 그라디언트는 매개 변수와 관련하여 일반적인 그라데이션이 아닙니다. 그 대신 NFQCA (Hafner & Riedmiller, 2011)가 이러한 방식으로 처음 사용하는 투입물에 대한 기울기입니다. 액터 네트워크를 업데이트하기 위해 이러한 그라디언트는 액터의 출력 레이어 (대상 대신)에 배치 된 다음 네트워크를 통해 다시 전파됩니다. 주어진 상태에서, Actor는 Critic가 평가하는 행동을 산출하기 위해 앞으로 나아가고, 결과적인 그라디언트는 Actor를 업데이트하는데 사용될 수 있습니다 :

다른 방법으로는 이러한 업데이트가 단순히 Actor와 비평 네트워크를 연결하는 것으로 생각할 수 있습니다. 전달에서 Actor의 결과는 Critic에게 전달되어 평가됩니다. 다음으로, 추정 된 Q 값은 Critic를 통해 백 프로 퍼 게이트되어 Q 값을 높이기 위해 조치가 어떻게 변경되어야 하는지를 나타내는 그라디언트 ∇aQ를 생성합니다. 후방 통과에서, 이러한 구배들은 Critic로부터 Actor를 통해 흐른다. 그런 다음 액터의 매개 변수를 통해서만 업데이트가 수행됩니다. 그림 2는이 업데이트의 예를 보여줍니다.

##3.1 안정적인 업데이트
Critic의 업데이트는 Actor의 정책이 최적의 정책을위한 좋은 프록시라는 가정에 의존합니다. Actor에 대한 업데이트는 정책 개선을위한 Critic의 그라디언트 또는 제안 된 지침이 환경에서 테스트 될 때 유효하다는 가정하에 있습니다. 이 학습 과정을 안정적이고 수렴하기 위해 여러 가지 기술이 필요하다는 것은 놀라운 일이 아닙니다.
Critic의 정책 Q (s, a | θQ)는 Actor와 Critic의 업데이트에 모두 영향을주기 때문에 Critic 정책의 오류로 인해 Actor, Critic 또는 둘 모두의 차이가 발생하는 파괴적인 피드백을 초래할 수 있습니다. 이 문제를 해결하기 위해 Mnih et al. (2015)는 Critic보다 느린 시간 규모로 변화하는 비판 네트워크의 복제물 인 Target-Q-Network Q '를 소개합니다. 이 목표 네트워크는 Critic 업데이트 (식 4)에 대한 다음 상태 목표를 생성하는 데 사용됩니다. 마찬가지로 Target-Actor-Network μ '는 액터의 정책에서 빠르게 변하는 것을 방지합니다.
두 번째 안정화 영향은 에이전트의 최신 경험 (일반적으로 1 백만)으로 구성된 FIFO 대기열 인 재생 메모리 D입니다. 이 메모리에서 균일하게 샘플링 된 미니 일괄 처리 업데이트는 최신 경험에서 독점적으로 업데이트하는 것보다 편견을 줄입니다.

이 두 기술을 사용하면 식 4의 비판적 손실과 식 5의 액터 업데이트를 안정적으로 다음과 같이 다시 표현할 수 있습니다.

마지막으로, 이러한 업데이트는 각 네트워크에 적용됩니다. 여기서 α는 그라디언트 디센트 알고리즘에 의해 결정되는 매개 변수 별 단계 크기입니다. 또한 표적 액터와 표적 비평 네트워크는 요소 τ«1 : 1을 사용하여 Actor와 평론가를 부드럽게 추적하도록 업데이트됩니다.

마지막으로 ADADELTA (Zeiler, 2012), RM-SPROP (Tieleman & Hinton, 2012) 또는 ADAM (Kingma & Ba, 2014)과 같은 적응 학습 속도 방법이 있습니다.

##3.2 네트워크 아키텍처

그림 2에서 Actor와 Critic 모두 동일한 아키텍처를 사용합니다. 58 개의 상태 입력은 1024-512-256-128 단위로 구성된 4 개의 완전히 연결된 레이어로 처리됩니다. 완전히 연결된 각 레이어 다음에는 음의 기울기 10-2를 갖는 정류 된 선형 (ReLU) 활성화 기능이 뒤 따른다. 완전히 연결된 레이어의 가중치는 표준 편차가 10-2 인 가우스 초기화를 사용합니다. 최종 내부 제품 계층에는 두 개의 선형 출력 레이어가 연결됩니다. 하나는 네 개의 개별 작업에 대한 레이어이고, 다른 하나는 이러한 작업에 수반되는 여섯 개의 매개 변수에 대한 레이어입니다. Critic는 58 개의 주 기능 외에도 4 개의 개별 동작과 6 개의 동작 매개 변수를 입력으로 사용합니다. 단일 스. 라 Q 값을 출력합니다. Actor와 Critic 학습 속도가 모두 10-3으로 설정된 ADAM 솔버를 사용합니다. 표적 네트워크는 τ = 10-4를 사용하여 Actor와 Critic를 추적합니다. 우리 대리인을위한 완전한 소스 코드는 https://github.com/mhauskn/dqn-hfo에서, 그리고 HFO 도메인은 https://github.com/mhauskn/HFO/에서 얻을 수 있습니다. 지속적인 행동 공간에서 심층적 인 학습 학습의 배경을 도입 한 후에 매개 변수화 된 행동 공간을 제시합니다.
4 매개 변수 동작 공간 아키텍처
(Masson & Konidaris, 2015)의 표기법에 따라 매개 변수화 된 동작 공간 마르코프 결정 프로세스 (PAMDP)는 이산 동작 세트 Ad = {a1, a2,. . . , ak}. 각 이산 행동 a ∈ Ad은 연속 매개 변수 {pa1,. . . , 파마} ∈ Rma. 작업은 튜플 (a, pa1, ..., pama)로 표시됩니다. 따라서 전체 행동 공간 A = ∪a∈Ad (a, pa1, ..., pama).
Half Field Offense에서, 완전한 매개 변수화 된 행동 공간 (2.2 절)은 A = (Dash, pdash, pdash) ∪ (Turn, pturn) ∪ (태클, ptackle) ∪ (Kick, pkick, pkick)입니다. 그림의 액터 네트워크
123456
2는 작업 공간을 불연속 작업 (Dash, Turn, Tackle, Kick)에 대한 하나의 출력 레이어로, 여섯 개의 연속 매개 변수 (pdash, pdash, pturn, ptackle, pkick, pkick)에 대한 또 다른 요소로 나타냅니다.
123456