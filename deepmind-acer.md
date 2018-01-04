# Sample Efficient Actor-Critic With Experience Replay

## Abstract

이 논문은 안정적인 표본 효율을 지니고 있으며, 각각의 57 게임에서 아타리 게임들과 몇 가지 Continuous Control Problems를 포함하여 어려운 환경에서 놀랄만큼 잘 수행되는 Experience Replay을 가진 Actor-Critic Deep Reinforcement Learning 에이전트를 제시합니다. 이를 달성하기 위해이 논문에서는 bias 보정, stochastic dueling network architectures 및 새로운 trust region policy optimization method을 사용하여 truncated importance sampling과 같은 몇 가지 혁신을 소개합니다.

```text
This paper presents an actor-critic deep reinforcement learning agent with ex- perience replay that is stable, sample efficient, and performs remarkably well on challenging environments, including the discrete 57-game Atari domain and several continuous control problems. To achieve this, the paper introduces several inno- vations, including truncated importance sampling with bias correction, stochastic dueling network architectures, and a new trust region policy optimization method.
```

## Introduction

에이전트가 인지 능력의 큰 레퍼토리를 배우도록 훈련받을 수있는 현실적인 시뮬레이션 환경은 인공 지능의 최근 돌파구의 핵심에있습니다 (Bellemare et al., 2013, Mnih et al., 2015, Schulman et al., 2015a, Narasimhan et al., 2015; Mnih et al., 2016; Brockman et al., 2016; Oh et al., 2016). 보다 풍부하고 현실적인 환경에서 에이전트의 기능이 향상되고 향상되었습니다. 불행하게도 이러한 진보로 인해 시뮬레이션 비용이 크게 증가했습니다. 특히 에이전트가 환경에 대해 작업 할 때마다 값 비싼 시뮬레이션 단계가 수행됩니다. 따라서 시뮬레이션 비용을 줄이기 위해서는 시뮬레이션 단계 (즉, 환경 샘플)의 수를 줄여야합니다. 에이전트가 실제 환경에 배포 될 때 샘플 효율성에 대한 이러한 요구가 더욱 강력 해졌습니다.

Experience replay (Lin, 1992)는 Deep Q-learning (Mnih et al., 2015; Schaul et al., 2016; Wang 등, 2016; Narasimhan et al., 2015)에서 인기를 얻었는데, 샘플 상관 관계를 줄이는 기술로 Replay는 샘플 효율을 향상시키는 데 실제로 가치있는 도구이며 우리의 실험에서 볼 수 있듯이 최첨단 Deep Q-learning 방법 (Schaul et al., 2016; Wang et al., 2016) 이 시점에서 아타리에 대한 가장 효율적인 샘플 기법을 큰 차이로 설명합니다. 그러나 두 가지 중요한 한계가 있기 때문에 Deep Q-learning보다 더 잘 수행해야합니다. 첫째, optimal policy의 deterministic 성질은 적 도메인에서의 사용을 제한한다. 둘째, Q 함수와 관련하여 욕심 많은 행동을 찾는 것은 큰 행동 공간에 비용이 많이 든다.

Policy gradient 방법은 인공 지능과 로봇 공학에서 중요한 진보의 핵심에 있었다 (Silver et al., 2014; Lillicrap 외 2015; Silver et al., 2016; Levine et al., 2015; Mnih et al., 2016 Schulman et al., 2015a, Heess et al., 2015). 이러한 방법 중 다수는 연속 도메인 또는 바둑과 같은 매우 구체적인 작업으로 제한됩니다. Mnih et al.의 on-policy asynchronous advantage actor critic (A3C)와 같이 continuous 및 discrete domain 모두에 적용 가능한 기존 변형. (2016) 은 샘플을 사용하는 데 비효율적이다.

연속적이고 이산적인 행동 공간 모두에 적용되는 안정적이고 표본 효율적인 배우 비평 방법의 설계는 오랜 기간 강화 된 보강 학습 (RL)의 장애물이었습니다. 우리는이 보고서가 이러한 도전 과제를 성공적으로 수행 한 첫 번째 사례라고 생각합니다. 좀 더 구체적으로, 우리는 아타리 (Aari)에서 우선 순위가 지정된 재생과 최첨단 Q 네트워크의 최첨단 성능에 거의 부합하는 경험 재생 (ACER)을 가진 배우 평론가를 소개하고 Atari와 Atari 모두에서 샘플 효율성 측면에서 실질적으로 A3C를 능가합니다 연속 제어 도메인.

ACER는 딥 뉴럴 네트워크, variance reduction method, the off-policy Retrace algorithm (Munos et al., 2016) 및 parallel training of RL agents (Mnih et al., 2016)의 최근 발전을 이용합니다. 그러나이 연구의 성공은이 논문에서 제시된 혁신, 즉 bias correction, stochastic dueling network architectures 및 efficient trust region policy optimization으로 생략 된 중요한 샘플링에 달려 있습니다.

이론적 인면에서이 논문은 바이어스 보정 기법을 통해 제안 된 절삭 중요성 샘플링에서 Retrace 연산자를 다시 작성할 수 있음을 입증합니다.

```text
Realistic simulated environments, where agents can be trained to learn a large repertoire of cognitive skills, are at the core of recent breakthroughs in AI (Bellemare et al., 2013; Mnih et al., 2015; Schulman et al., 2015a; Narasimhan et al., 2015; Mnih et al., 2016; Brockman et al., 2016; Oh et al., 2016). With richer realistic environments, the capabilities of our agents have increased and improved. Unfortunately, these advances have been accompanied by a substantial increase in the cost of simulation. In particular, every time an agent acts upon the environment, an expensive simulation step is conducted. Thus to reduce the cost of simulation, we need to reduce the number of simulation steps (i.e. samples of the environment). This need for sample efficiency is even more compelling when agents are deployed in the real world.

Experience replay (Lin, 1992) has gained popularity in deep Q-learning (Mnih et al., 2015; Schaul et al., 2016; Wang et al., 2016; Narasimhan et al., 2015), where it is often motivated as a technique for reducing sample correlation. Replay is actually a valuable tool for improving sample efficiency and, as we will see in our experiments, state-of-the-art deep Q-learning methods (Schaul et al., 2016; Wang et al., 2016) have been up to this point the most sample efficient techniques on Atari by a significant margin. However, we need to do better than deep Q-learning, because it has two important limitations. First, the deterministic nature of the optimal policy limits its use in adversarial domains. Second, finding the greedy action with respect to the Q function is costly for large action spaces.

Policy gradient methods have been at the heart of significant advances in AI and robotics (Silver et al., 2014; Lillicrap et al., 2015; Silver et al., 2016; Levine et al., 2015; Mnih et al., 2016; Schulman et al., 2015a; Heess et al., 2015). Many of these methods are restricted to continuous domains or to very specific tasks such as playing Go. The existing variants applicable to both continuous and discrete domains, such as the on-policy asynchronous advantage actor critic (A3C) of Mnih et al. (2016), are sample inefficient.

The design of stable, sample efficient actor critic methods that apply to both continuous and discrete action spaces has been a long-standing hurdle of reinforcement learning (RL). We believe this paper is the first to address this challenge successfully at scale. More specifically, we introduce an actor critic with experience replay (ACER) that nearly matches the state-of-the-art performance of deep Q-networks with prioritized replay on Atari, and substantially outperforms A3C in terms of sample efficiency on both Atari and continuous control domains.

ACER capitalizes on recent advances in deep neural networks, variance reduction techniques, the off-policy Retrace algorithm (Munos et al., 2016) and parallel training of RL agents (Mnih et al., 2016). Yet, crucially, its success hinges on innovations advanced in this paper: truncated importance sampling with bias correction, stochastic dueling network architectures, and efficient trust region policy optimization.

On the theoretical front, the paper proves that the Retrace operator can be rewritten from our proposed truncated importance sampling with bias correction technique.
```

## 2 BACKGROUND AND PROBLEM SETUP

이산 시간 단계에 걸쳐 환경과 상호 작용하는 에이전트를 고려하십시오. 에이전트는 시간 단계 t에서 $nx$ 차원 상태 벡터 $x_t Rnx$를 관찰하고, 정책 $π (a | xt)$에 따라 행동을 선택하고 환경에 의해 생성 된 보상 신호 rt ∈ R을 관찰한다. 우리는 ${1, 2,. . . , Na}$, 제 5 절에서 ∈ A ⊆ Rna에 대한 연속적인 연 관들로 구성된다.

에이전트의 목표는 기대 수익률 $Rt = i0 γirt + i$를 극대화하는 것입니다. 할인 계수 $γ  [0,1]$은 즉각적이고 미래의 보상의 중요성을 상쇄시킨다. 정책 π를 따르는 에이전트의 경우 상태 작업 및 상태 값만 기능의 표준 정의를 사용합니다.
[Rt | xt, at]와 Vπ (xt) = Eat [Qt (xt, at) | xt]에서 Qt (xt, at) = Ext + 1 : ∞이다. 여기서, 기대는 관찰 된 환경 상태 xt와 생성 된 동작에 관한 것이다.
xt + 1 : ∞는 시간 t + 1에서 시작되는 상태 궤적을 나타낸다.
또한 우위 함수 Aπ (xt, at) = Qπ (xt, at) - Vπ (xt)를 정의 할 필요가있다.
그때의 상대적인 값은 [Aπ (xt, at)] = 0이다.
미분 가능 정책 πθ (at | xt)의 매개 변수 θ는 Schulman et al.의 표기법을 사용하는 정책 기울기 (Sutton et al., 2000)에 대한 할인 된 근사값을 사용하여 갱신 될 수있다.
(2015b)는 다음과 같이 정의됩니다.

Schulman et al. (xt, at), 할인 된 수익 Rt, 또는 시간차 잔여 rt + γV π (xt + 1)로 대체 할 수있다. V π (xt). 그러나 이러한 선택 사항은 다른 차이가 있습니다. 또한, 실제로 우리는 신경망으로 이러한 양을 근사화하여 추가 근사 오차 및 편향을 도입합니다. 일반적으로, Rt를 사용하는 정책 기울기 추정기는 더 높은 분산 및 낮은 바이어스를 가지지 만 함수 근사를 사용하는 평가 기는 높은 바이어스 및 더 낮은 분산을 가질 것이다. Rt를 현재 값 함수 근사와 결합하여 경계 편차를 유지하면서 바이어스를 최소화하면 ACER의 핵심 설계 원리 중 하나입니다.
바이어스와 분산을 트레이드 오프하기 위해, Mnih et al.의 비동기 우위 애호가 평론가 (A3C) (2016)은 단일 궤도 샘플을 사용하여 다음과 같은 기울기 근사값을 얻습니다.

A3C는 k-step returns와 function approximation을 결합하여 균형과 편향을 조정합니다. 우리
V π (xt)를 분산을 줄이기 위해 사용 된 정책 기울기 기준선으로 생각할 수있다. θv
다음 섹션에서는 ACER의 개별 액션 버전을 소개합니다. ACER은 Mnih et al.의 A3C 방법의 오프 정책 대응 물로 이해 될 수있다. (2016). 따라서 ACER는 효율적인 병렬 CPU 계산을 포함하여 A3C의 모든 엔지니어링 혁신을 기반으로합니다.

ACER는 단일 딥 신경망을 사용하여 정책 πθ (at | xt) 및 값 함수 V π (xt)를 추정합니다. (명확성과 보편성을 위해 두 개의 다른 기호를 사용하여
θv
정책과 가치 함수, θ와 θv, 그러나 이들 매개 변수의 대부분은 단일 신경에서 공유된다
네트워크입니다.) 우리의 신경 네트워크는 A3C에서 사용되는 네트워크를 기반으로하지만 몇 가지 수정 및 새로운 모듈을 도입 할 것입니다.

```text
Consider an agent interacting with its environment over discrete time steps. At time step t, the agent observes the nx-dimensional state vector xt ∈ X ⊆ Rnx , chooses an action at according to a policy π(a|xt) and observes a reward signal rt ∈ R produced by the environment. We will consider discrete actions at ∈ {1, 2, . . . , Na} in Sections 3 and 4, and continuous ac tions at ∈ A ⊆ Rna in Section 5.
The goal of the agent is to maximize the discounted return Rt = i≥0 γirt+i in expectation. The discount factor γ ∈ [0, 1) trades-off the importance of immediate and future rewards. For an agent following policy π, we use the standard definitions of the state-action and state only value functions:
Qπ(xt,at)=Ext+1:∞,at+1:∞ [Rt|xt,at] and Vπ(xt)=Eat [Qπ(xt,at)|xt]. Here, the expectations are with respect to the observed environment states xt and the actions generated
by the policy π, where xt+1:∞ denotes a state trajectory starting at time t + 1.
We also need to define the advantage function Aπ (xt , at ) = Qπ (xt , at ) − V π (xt ), which provides a
relativemeasureofvalueofeachactionsinceEat [Aπ(xt,at)]=0.
The parameters θ of the differentiable policy πθ(at|xt) can be updated using the discounted approxi- mation to the policy gradient (Sutton et al., 2000), which borrowing notation from Schulman et al.
(2015b), is defined as:

Following Proposition 1 of Schulman et al. (2015b), we can replace Aπ (xt , at ) in the above expression with the state-action value Qπ (xt , at ), the discounted return Rt , or the temporal difference residual rt + γV π(xt+1) − V π(xt), without introducing bias. These choices will however have different variance. Moreover, in practice we will approximate these quantities with neural networks thus introducing additional approximation errors and biases. Typically, the policy gradient estimator using Rt will have higher variance and lower bias whereas the estimators using function approximation will have higher bias and lower variance. Combining Rt with the current value function approximation to minimize bias while maintaining bounded variance is one of the central design principles behind ACER.
To trade-off bias and variance, the asynchronous advantage actor critic (A3C) of Mnih et al. (2016) uses a single trajectory sample to obtain the following gradient approximation:

A3C combines both k-step returns and function approximation to trade-off variance and bias. We
may think of V π (xt) as a policy gradient baseline used to reduce variance. θv
In the following section, we will introduce the discrete-action version of ACER. ACER may be understood as the off-policy counterpart of the A3C method of Mnih et al. (2016). As such, ACER builds on all the engineering innovations of A3C, including efficient parallel CPU computation.

ACER uses a single deep neural network to estimate the policy πθ(at|xt) and the value function V π (xt). (For clarity and generality, we are using two different symbols to denote the parameters of
θv
the policy and value function, θ and θv , but most of these parameters are shared in the single neural
network.) Our neural networks, though building on the networks used in A3C, will introduce several modifications and new modules.
```

## 3 DISCRETE ACTOR CRITIC WITH EXPERIENCE REPLAY

경험 재생을 통한 정책 외 학습은 배우 - 비평가의 표본 효율성을 향상시키는 확실한 전략으로 보입니다. 그러나 오프 정책 견적서의 분산과 안정성을 통제하는 것은 매우 어렵습니다. 중요도 표본 추출은 정책 외 학습을위한 가장 보편적 인 접근 방법 중 하나이다 (Meuleau et al., 2000; Jie & Abbeel, 2010; Levine & Koltun, 2013). 우리의 맥락에서, 그것은 다음과 같이 진행된다. 우리가 행동 정책 μ에 따라 샘플링 된 궤적 {x0, a0, r0, μ (· | x0), ..., xk, ak, rk, μ (· | xk) 우리의 경험을 기억하십시오. 그런 다음 중요도 가중치 정책 기울기는 다음과 같이 표시됩니다.

여기서 ρt = π (at | xt)는 중요도 가중치를 나타냅니다. 이 추정기는 편향되지 않지만,
 μ (at | xt)
잠재적으로 무한한 중요도 가중치가 많은 제품을 포함하기 때문에 매우 높은 차이가 난다. 중요도 가중치 제품이 폭발하지 않도록 Wawrzyn ski (2009)는이 제품을 잘라냅니다. 전체 궤적에 대해 절단 된 중요도 샘플링은 분산이 한정되어 있지만 상당한 편향을 겪을 수 있습니다.
최근 Degris et al. (2012)는 프로세스의 제한된 분포에 대해 한계 값 함수를 사용하여 다음과 같은 그라디언트 근사값을 산출함으로써이 문제를 공격했습니다.
gmarg = Ext ~ β, at ~ μ [ρt∇θ log πθ (at | xt) Qπ (xt, at)], (4)
여기서 Ext ~ β, at ~ μ [·]는 행동 방침 μ에 의한 제한 분포 β (x) = limt → ∞ P (xt = x | x0, μ)에 대한 기대치이다. 표기법을 간결하게하기 위해, 우리는 Ext ~ β, at ~ μ [·]를 Extat [•]로 대체 할 것이며 필요할 때 독자들에게 상기시켜 줄 것입니다.
방정식 (4)에 대한 두 가지 중요한 사실을 강조 표시해야합니다. 우선, 그것은 Qπ에 의존하지 않고 Qμ에 의존하지 않는다는 것에 주목하십시오. 따라서 Qπ를 추정 할 수 있어야합니다. 둘째로, 우리는 더 이상 중요도 가중치를 가지고 있지 않지만, 대신에 중요도 가중치 ρt 만 추정하면됩니다. 이 낮은 차원 공간에서 중요도 샘플링 (궤도와 반대되는 여백 이상)은 분산이 낮을 것으로 예상됩니다.
Degris et al. (2012)는 람다 리턴을 사용하여 방정식 (4)에서 Qπ를 추정한다 : Rtλ = rt + (1-λ) γV (xt + 1) + λγρt + 1Rtλ + 1. 이 추정기는 편향과 분산을 상쇄하기 위해 미리 λ를 선택하는 방법을 알고 있어야합니다. 더욱이 분산을 줄이기 위해 λ의 작은 값을 사용할 때 간혹 큰 중요도 가중치로 인해 여전히 불안정성이 발생할 수 있습니다.
다음 하위 섹션에서는 Munos et al.의 Retrace 알고리즘을 채택합니다. (2016) Qπ를 추정한다. 그 후, 우리는 Degris et al.의 정책 외 배우 평론가의 안정성을 향상시키기위한 중요도 가중치 절단 기법을 제안한다. (2012), 정책 최적화를위한 계산 효율적인 효율적인 신뢰 영역 기법을 소개한다. 지속적인 행동 공간을위한 ACER의 공식화는 5 장에서 진전 된 추가 혁신을 필요로 할 것이다.

```text
Off-policy learning with experience replay may appear to be an obvious strategy for improving the sample efficiency of actor-critics. However, controlling the variance and stability of off-policy estimators is notoriously hard. Importance sampling is one of the most popular approaches for off- policy learning (Meuleau et al., 2000; Jie & Abbeel, 2010; Levine & Koltun, 2013). In our context, it proceeds as follows. Suppose we retrieve a trajectory {x0 , a0 , r0 , μ(·|x0 ), · · · , xk , ak , rk , μ(·|xk )}, where the actions have been sampled according to the behavior policy μ, from our memory of experiences. Then, the importance weighted policy gradient is given by:

where ρt = π(at|xt) denotes the importance weight. This estimator is unbiased, but it suffers from
 μ(at |xt )
very high variance as it involves a product of many potentially unbounded importance weights. To prevent the product of importance weights from exploding, Wawrzyn ́ski (2009) truncates this product. Truncated importance sampling over entire trajectories, although bounded in variance, could suffer from significant bias.
Recently, Degris et al. (2012) attacked this problem by using marginal value functions over the limiting distribution of the process to yield the following approximation of the gradient:
gmarg = Ext∼β,at∼μ [ρt∇θ log πθ(at|xt)Qπ(xt, at)] , (4)
where Ext∼β,at∼μ[·] is the expectation with respect to the limiting distribution β(x) = limt→∞ P(xt = x|x0,μ) with behavior policy μ. To keep the notation succinct, we will replace Ext∼β,at∼μ[·] with Extat [·] and ensure we remind readers of this when necessary.
Two important facts about equation (4) must be highlighted. First, note that it depends on Qπ and not on Qμ, consequently we must be able to estimate Qπ. Second, we no longer have a product of importance weights, but instead only need to estimate the marginal importance weight ρt. Importance sampling in this lower dimensional space (over marginals as opposed to trajectories) is expected to exhibit lower variance.
Degris et al. (2012) estimate Qπ in equation (4) using lambda returns: Rtλ = rt +(1−λ)γV (xt+1)+ λγρt+1Rtλ+1. This estimator requires that we know how to choose λ ahead of time to trade off bias and variance. Moreover, when using small values of λ to reduce variance, occasional large importance weights can still cause instability.
In the following subsection, we adopt the Retrace algorithm of Munos et al. (2016) to estimate Qπ . Subsequently, we propose an importance weight truncation technique to improve the stability of the off-policy actor critic of Degris et al. (2012), and introduce a computationally efficient trust region scheme for policy optimization. The formulation of ACER for continuous action spaces will require further innovations that are advanced in Section 5.
```