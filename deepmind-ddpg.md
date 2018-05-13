# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING(DDPG)

Timothy P. Lillicrap∗
, Jonathan J. Hunt∗
, Alexander Pritzel, Nicolas Heess,
Tom Erez, Yuval Tassa, David Silver & Daan Wierstra
Google Deepmind
London, UK
{countzero, jjhunt, apritzel, heess,
etom, tassa, davidsilver, wierstra} @ google.com

## ABSTRACT

Deep Q-Learning의 성공을 이끄는 아이디어를 지속적인 행동 영역에 적용합니다. 우리는 연속적인 행동 공간에서 작동 할 수있는 결정 론적 정책 구배에 기반한 Actor-Critic가, model-free 알고리즘을 제시합니다. 우리의 알고리즘은 동일한 학습 알고리즘, 네트워크 아키텍처 및 하이퍼 파라미터를 사용하여 CratPole Swing-up,Dexterous manipulation, legged locomotion 및 car driving과 같은 고전적인 문제를 포함 해 20 개 이상의 시뮬레이션 된 물리 작업을 견고하게 해결합니다. 우리의 알고리즘은 도메인 및 그 파생어의 역 동성에 완전히 액세스 할 수있는 계획 알고리즘에 의해 발견 된 정책과 성능이 유사한 정책을 찾을 수 있습니다. 우리는 많은 작업에서 알고리즘이 원시 픽셀 입력에서 직접 "정책"을 배울 수 있음을 보여줍니다.

We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies “end-to-end”: directly from raw pixel inputs.

## 1 INTRODUCTION

인공 지능 분야의 주된 목표 중 하나는 처리되지 않은 고차원의 감각 입력으로부터 복잡한 작업을 해결하는 것입니다. 최근에는 감각 처리 (Krizhevsky et al., 2012)에 대한 심층 학습의 발전과 보강 학습을 결합하여 "Deep Q Network"(DQN) 알고리즘 (Mnih et al., 2015)의 결과로 상당한 진전이 이루어졌습니다. 처리를 위해 픽셀을 사용하여 많은 Atari 비디오 게임에서 인간 수준의 성능을 구현할 수 있습니다. 이렇게하기 위해, 깊은 신경 회로망 근사가 동작 값 함수를 추정하는 데 사용되었습니다.
그러나 DQN은 고차원 관측 공간에 대한 문제를 해결하지만 이산 형 및 저 차원 작업 공간 만 처리 할 수 ​​있습니다. 많은 관심 대상 업무, 특히 물리적 제어 작업에는 연속적인 (실제 가치가있는) 차원 높은 차원의 작업 공간이 있습니다. DQN은 직설적 일 수는 없다.
연속 가치가있는 경우 모든 단계에서 반복적 인 최적화 프로세스가 필요한 동작 값 함수를 최대화하는 동작을 찾는 것에 의존하기 때문에 연속 도메인에 적용됩니다.
DQN과 같은 심층적 인 학습 방법을 연속 영역에 적용하는 명백한 접근법은 단순히 동작 공간을 이산화하는 것입니다. 그러나 이것은 많은 한계를 가지고 있습니다. 특히 차원의 저주입니다 : 행동의 수는 자유도의 수에 따라 기하 급수적으로 증가합니다. 예를 들어, 각 조인트에 대한 가장 큰 이산화 ai ∈ {-k, 0, k}를 갖는 7 자유도 시스템 (사람의 팔에서와 같이)은 차원이있는 작업 공간으로 연결됩니다.
삼
7 = 2187입니다. 상황은 더 미세한 입자 화 된 이산화가 필요하므로 작업을 미세하게 제어해야하는 작업의 경우 더욱 심각해 지므로 개별 작업 수가 폭발적으로 늘어납니다. 이러한 대규모 작업 공간은 효율적으로 탐색하기 어렵 기 때문에 DQN과 유사한 네트워크를 성공적으로 교육하는 것은 어려울 수 있습니다. 또한, 행동 공간의 단순한 이산화 (discretization)는 많은 문제를 해결하는 데 필수적 일 수있는 행동 영역의 구조에 관한 정보를 불필요하게 버린다.
이 작품에서 우리는 고차원적이고 연속적인 행동 공간에서 정책을 학습 할 수있는 심 함수 근사를 사용하여 모형없는 오프 정책 배우 비평 알고리즘을 제시합니다. 우리의 연구는 NFQCA (Hafner & Riedmiller, 2011)와 비슷한 DPG (Deterministic Policy Gradient) 알고리즘 (Silver et al., 2014)을 기반으로하며 유사한 아이디어가 Prokhorov et al., 1997에서 발견 될 수있다. . 그러나 우리가 아래에서 보여 주듯이 신경 기능 근사자를 가진이 actor-critic 방법의 순진한 적용
도전적인 문제에 대해 불안정합니다.
여기서 우리는 DQN (Mightih et al., 2013; 2015)의 최근 성공에서 얻은 통찰력을 바탕으로 배우 - 비평가 접근법을 결합합니다. DQN 이전에는 일반적으로 커다란 비선형 함수 근사기를 사용하여 가치 함수를 학습하는 것이 어렵고 불안정한 것으로 여겨졌습니다. DQN은 두 가지 혁신으로 인해 이러한 함수 근사자를 사용하여 값 함수를 학습 할 수 있습니다. 1. 네트워크는 샘플 간의 상관 관계를 최소화하기 위해 재생 버퍼의 샘플로 오프 정책으로 교육됩니다. 2. 시간차 백업 동안 일관된 목표를 제공하기 위해 목표 Q 네트워크로 네트워크를 교육합니다. 이 작업에서 우리는 깊은 학습의 최근 발전 인 일괄 정규화 (Ioffe & Szegedy, 2015)와 함께 동일한 아이디어를 사용합니다.
우리의 방법을 평가하기 위해 우리는 복잡한 다 관절 운동, 불안정하고 풍부한 접촉 역학 및 보행 행동을 포함하는 다양한 도전적인 신체 제어 문제를 만들었습니다.
이 중 많은 것들은 카트폴 스윙 업 문제뿐만 아니라 많은 새로운 영역과 같은 고전적인 문제입니다. 로봇 제어의 오랜 과제는 비디오 같은 원시 감각 입력으로부터 직접 행동 정책을 학습하는 것입니다. 따라서 우리는 고정 관측점 카메라를 시뮬레이터에 배치하고 저 차원 관측 (예 : 관절 각) 및 픽셀에서 직접 모든 작업을 시도했습니다.
Deep DPG (DDPG)라고하는 우리의 모델없는 접근 방식은 동일한 하이퍼 매개 변수 및 네트워크 구조를 사용하여 저 차원 관측 (예 : 직교 좌표 또는 관절 각)을 사용하여 모든 작업에 대한 경쟁력있는 정책을 학습 할 수 있습니다. 대부분의 경우 픽셀에서 직접 올바른 정책을 학습하고 하이퍼 매개 변수 및 네트워크 구조를 계속 유지할 수 있습니다 1

이 접근법의 핵심 기능은 간단합니다. 즉, "움직이는 부분"이 거의없는 배우 - 평론 아키텍처와 학습 알고리즘 만 있으면되므로 더 어려운 문제와 대규모 네트워크를 쉽게 구현하고 확장 할 수 있습니다. 물리적 인 통제 문제에 대해서는 우리
결과는 기본 시뮬레이트 된 동역학 및 그 파생어 (보충 정보 참조)에 완전히 액세스 할 수있는 플래너 (Tassa 외, 2012)에 의해 계산 된 기준선으로 나타납니다. 흥미롭게도 DDPG는 경우에 따라 플래너의 성능을 초과하는 정책을 찾을 수 있습니다 (플래너는 항상 기본 저 차원 상태 공간을 계획합니다).

One of the primary goals of the field of artificial intelligence is to solve complex tasks from unprocessed, high-dimensional, sensory input. Recently, significant progress has been made by combining advances in deep learning for sensory processing (Krizhevsky et al., 2012) with reinforcement learning, resulting in the “Deep Q Network” (DQN) algorithm (Mnih et al., 2015) that is capable of human level performance on many Atari video games using unprocessed pixels for input. To do so, deep neural network function approximators were used to estimate the action-value function.
However, while DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly
applied to continuous domains since it relies on a finding the action that maximizes the action-value function, which in the continuous valued case requires an iterative optimization process at every step.
An obvious approach to adapting deep reinforcement learning methods such as DQN to continuous domains is to to simply discretize the action space. However, this has many limitations, most notably the curse of dimensionality: the number of actions increases exponentially with the number of degrees of freedom. For example, a 7 degree of freedom system (as in the human arm) with the coarsest discretization ai ∈ {−k, 0, k} for each joint leads to an action space with dimensionality:
3
7 = 2187. The situation is even worse for tasks that require fine control of actions as they require a correspondingly finer grained discretization, leading to an explosion of the number of discrete actions. Such large action spaces are difficult to explore efficiently, and thus successfully training DQN-like networks in this context is likely intractable. Additionally, naive discretization of action spaces needlessly throws away information about the structure of the action domain, which may be essential for solving many problems.
In this work we present a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces. Our work is based on the deterministic policy gradient (DPG) algorithm (Silver et al., 2014) (itself similar to NFQCA (Hafner & Riedmiller, 2011), and similar ideas can be found in (Prokhorov et al., 1997)). However, as we show below, a naive application of this actor-critic method with neural function approximators
is unstable for challenging problems.
Here we combine the actor-critic approach with insights from the recent success of Deep Q Network (DQN) (Mnih et al., 2013; 2015). Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable. DQN is able to learn value functions using such function approximators in a stable and robust way due to two innovations: 1. the network is trained off-policy with samples from a replay buffer to minimize correlations between samples; 2. the network is trained with a target Q network to give consistent targets during temporal difference backups. In this work we make use of the same ideas, along with batch normalization (Ioffe & Szegedy, 2015), a recent advance in deep learning.
In order to evaluate our method we constructed a variety of challenging physical control problems that involve complex multi-joint movements, unstable and rich contact dynamics, and gait behavior.
Among these are classic problems such as the cartpole swing-up problem, as well as many new domains. A long-standing challenge of robotic control is to learn an action policy directly from raw sensory input such as video. Accordingly, we place a fixed viewpoint camera in the simulator and attempted all tasks using both low-dimensional observations (e.g. joint angles) and directly from pixels.
Our model-free approach which we call Deep DPG (DDPG) can learn competitive policies for all of our tasks using low-dimensional observations (e.g. cartesian coordinates or joint angles) using the same hyper-parameters and network structure. In many cases, we are also able to learn good policies directly from pixels, again keeping hyperparameters and network structure constant 1
.
A key feature of the approach is its simplicity: it requires only a straightforward actor-critic architecture and learning algorithm with very few “moving parts”, making it easy to implement and scale to more difficult problems and larger networks. For the physical control problems we compare our
results to a baseline computed by a planner (Tassa et al., 2012) that has full access to the underlying simulated dynamics and its derivatives (see supplementary information). Interestingly, DDPG can sometimes find policies that exceed the performance of the planner, in some cases even when learning from pixels (the planner always plans over the underlying low-dimensional state space).

## 2 BACKGROUND

이산 타임 스텝에서 환경 E와 상호 작용하는 에이전트로 구성된 표준 보강 학습 설정을 고려합니다. 에이전트는 각 타임 스텝에서 관찰 xt를 받고, 조치를 취하고 스칼라 보상 rt를 받는다. 여기에서 고려되는 모든 환경에서 액션은 IRN에서 실제 값입니다. 일반적으로 환경은 부분적으로 관찰되어 전체
관찰의 역사, 상태 쌍을 기술하기 위해 동작 쌍 st = (x1, a1, ..., at-1, xt)가 요구 될 수있다. 여기서 우리는 환경이 완전히 관찰되어 st = xt라고 가정했다.
에이전트의 행동은 상태 π : S → P (A)에 대한 확률 분포로 상태를 매핑하는 정책 π에 의해 정의됩니다. 환경 E도 확률적일 수 있습니다. 우리는 그것을 Markov로 모델링합니다.
상태 공간 S, 행동 공간 A = IRN, 초기 상태 분포 p (s1), 전이 역학 p (st + 1 | st, at) 및 보상 함수 r (st, at)
주에서의 수익은 할인 된 미래 보상의 합으로 정의됩니다. Rt =
태평양 표준시
i = t
γ
(i-t)
r (si
, ai)
할인 인자 γ ∈ [0, 1]로 계산된다. 반환 값은 선택한 동작에 따라 달라 지므로 정책 π에 따라 달라지며 확률적일 수 있습니다. 강화 학습의 목표는 시작 분포 J = Eri, si ~ E, ai ~ π [R1]로부터 예상 수익을 최대화하는 정책을 학습하는 것입니다. 우리는 정책 π에 대한 할인 된 주 방문 분포를 ρ
π.
액션 가치 함수는 많은 보강 학습 알고리즘에 사용됩니다. 이것은 state at에서 조치를 취한 후 정책 π에 따라 기대 수익을 설명합니다.


보강 학습에서의 많은 접근법은 Bellman 방정식으로 알려진 재귀 관계를 사용합니다.
큐
π
(st, at) = Ert, st + 1 - E

r (st, at) + γEat + 1 - π [Q
π
(st + 1, at + 1)]
(2)
목표 정책이 결정 론적이라면 μ : S ← A 함수로 설명하고 내부 기대를 피할 수 있습니다.
큐
μ
(st, at) = Ert, st + 1 - E [r (st, at) + γQμ
(st + 1, μ (st + 1))] (3)
기대는 환경에만 달려 있습니다. 이것은 Qμ off 정책을 배우는 것이 가능하다는 것을 의미하며,
다른 확률 적 행동 방침 (β)으로부터 생성 된 전이를 사용하여.
일반적으로 사용되는 off-policy 알고리즘 인 Q-learning (Watkins & Dayan, 1992)은 욕심쟁이 정책 μ (s) = arg maxa Q (s, a)를 사용합니다. θ Q에 의해 매개 변수화 된 함수 근사를 고려하면 손실을 최소화하여 최적화됩니다.
L (θ
Q) = Est ρβ, at ~ β, rt ~ E
h
Q (st, at | θ
Q) - yt
? 2
나는
(4)
어디에
yt = r (st, at) + γQ (st + 1, μ (st + 1) | θ
큐). (5)
yt는 또한 θ에 의존하지만
Q, 이것은 일반적으로 무시됩니다.
이론적 인 성능 보장이 불가능하고 실제적으로 학습이 불안정 해지는 경향이 있으므로 학습 가치 또는 행동 가치 함수에 대한 커다란 비선형 함수 근사를 사용하는 것이 과거에는 종종 회피되었습니다. 최근에 (Mnih et al., 2013; 2015) 대형 신경망을 함수 근사자로 효과적으로 활용하기 위해 Q-learning 알고리즘을 채택했다. 그들의 알고리즘은 픽셀에서 Atari 게임을하는 법을 배울 수있었습니다. Q- 학습을 확장하기 위해 그들은 재생 버퍼의 사용과 yt를 계산하기위한 별도의 목표 네트워크라는 두 가지 주요 변경 사항을 도입했습니다. 우리는 DDPG의 맥락에서 이들을 채택하고 다음 절에서 그 구현을 설명한다.

We consider a standard reinforcement learning setup consisting of an agent interacting with an environment E in discrete timesteps. At each timestep t the agent receives an observation xt, takes an action at and receives a scalar reward rt. In all the environments considered here the actions are real-valued at ∈ IRN . In general, the environment may be partially observed so that the entire
history of the observation, action pairs st = (x1, a1, ..., at−1, xt) may be required to describe the state. Here, we assumed the environment is fully-observed so st = xt.
An agent’s behavior is defined by a policy, π, which maps states to a probability distribution over the actions π : S → P(A). The environment, E, may also be stochastic. We model it as a Markov
decision process with a state space S, action space A = IRN , an initial state distribution p(s1), transition dynamics p(st+1|st, at), and reward function r(st, at).
The return from a state is defined as the sum of discounted future reward Rt =
PT
i=t
γ
(i−t)
r(si
, ai)
with a discounting factor γ ∈ [0, 1]. Note that the return depends on the actions chosen, and therefore on the policy π, and may be stochastic. The goal in reinforcement learning is to learn a policy which maximizes the expected return from the start distribution J = Eri,si∼E,ai∼π [R1]. We denote the discounted state visitation distribution for a policy π as ρ
π.
The action-value function is used in many reinforcement learning algorithms. It describes the expected return after taking an action at in state st and thereafter following policy π:

Many approaches in reinforcement learning make use of the recursive relationship known as the
Bellman equation:
Qπ(st, at) = Ert,st+1∼E

r(st, at) + γ Eat+1∼π [Q
π
(st+1, at+1)]
(2)
If the target policy is deterministic we can describe it as a function µ : S ← A and avoid the inner
expectation:
Q
µ
(st, at) = Ert,st+1∼E [r(st, at) + γQµ
(st+1, µ(st+1))] (3)
The expectation depends only on the environment. This means that it is possible to learn Qµ offpolicy,
using transitions which are generated from a different stochastic behavior policy β.
Q-learning (Watkins & Dayan, 1992), a commonly used off-policy algorithm, uses the greedy policy
µ(s) = arg maxa Q(s, a). We consider function approximators parameterized by θ
Q, which we
optimize by minimizing the loss:
L(θ
Q) = Est∼ρβ,at∼β,rt∼E
h
Q(st, at|θ
Q) − yt
2
i
(4)
where
yt = r(st, at) + γQ(st+1, µ(st+1)|θ
Q). (5)
While yt is also dependent on θ
Q, this is typically ignored.
The use of large, non-linear function approximators for learning value or action-value functions has
often been avoided in the past since theoretical performance guarantees are impossible, and practically
learning tends to be unstable. Recently, (Mnih et al., 2013; 2015) adapted the Q-learning
algorithm in order to make effective use of large neural networks as function approximators. Their
algorithm was able to learn to play Atari games from pixels. In order to scale Q-learning they introduced
two major changes: the use of a replay buffer, and a separate target network for calculating
yt. We employ these in the context of DDPG and explain their implementation in the next section