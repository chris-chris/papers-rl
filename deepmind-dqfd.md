# DQFD

## Abstract

심층 강화 학습 (RL)은 어려운 의사 결정 문제에서 몇 가지 중요한 성공을 거두었습니다. 그러나 이러한 알고리즘은 일반적으로 합리적인 성능에 도달하기 전에 엄청난 양의 데이터가 필요합니다. 사실, 학습 중 성능은 극도로 떨어질 수 있습니다. 이는 시뮬레이터에서 허용 될 수 있지만, 에이전트가 실제 환경에서 학습해야하는 많은 실제 작업에 대한 심층 RL의 적용 가능성을 심각하게 제한합니다. 이 문서에서는 에이전트가 시스템의 이전 제어에서 데이터에 액세스 할 수있는 설정을 연구합니다. 우리는 작은 양의 데모 데이터를 활용하여 비교적 적은 양의 데모 데이터로부터 학습 프로세스를 대폭 가속화하고 학습하는 동안 데모 데이터의 필요한 비율을 자동으로 평가할 수있는 데모 (DQfD)의 딥 Q학습 알고리즘을 제시합니다 우선 순위가 매겨진 재생 메커니즘 덕분입니다. DQfD는 일시적인 차이점 업데이트와 Demonstrator의 행동에 대한 감독 분류를 결합하여 작동합니다. 우리는 DQfD가 42 개의 게임 중 41 개의 첫 번째 단계에서 더 좋은 점수로 시작하고 평균적으로 PDD DQN 8300 만 걸음을 따라 가면서 DQfD가 Prioritized Dueling Double Deep Q-Networks (PDD DQN)보다 더 나은 초기 성능을 보인다는 것을 보여줍니다. DQfD의 성능. DQfD는 42 개 게임 중 14 개 게임에서 제공되는 최고의 데모를 능가하는 것을 배웁니다. 또한 DQfD는 인간 Demonstration를 활용하여 11 개의 게임에 대한 최첨단 결과를 얻습니다. 마지막으로 DQfD가 데모 데이터를 DQN에 통합하기위한 세 가지 관련 알고리즘보다 우수한 성능을 보여줍니다.

Deep reinforcement learning (RL) has achieved several high profile successes in difficult decision-making problems. However, these algorithms typically require a huge amount of data before they reach reasonable performance. In fact, their performance during learning can be extremely poor. This may be acceptable for a simulator, but it severely limits the applicability of deep RL to many real-world tasks, where the agent must learn in the real environment. In this paper we study a setting where the agent may access data from previous control of the system. We present an algorithm, Deep Q-learning from Demonstrations (DQfD), that leverages small sets of demonstration data to massively accelerate the learning process even from relatively small amounts of demonstration data and is able to automatically assess the necessary ratio of demonstration data while learning thanks to a prioritized replay mechanism. DQfD works by combining temporal difference updates with supervised classification of the demonstrator’s actions. We show that DQfD has better initial performance than Prioritized Dueling Double Deep Q-Networks (PDD DQN) as it starts with better scores on the first million steps on 41 of 42 games and on average it takes PDD DQN 83 million steps to catch up to DQfD’s performance. DQfD learns to out-perform the best demonstration given in 14 of 42 games. In addition, DQfD leverages human demonstrations to achieve state-of-the-art results for 11 games. Finally, we show that DQfD performs better than three related algorithms for incorporating demonstration data into DQN.

## Introduction

지난 몇 년 동안, 순차적 인 의사 결정 문제와 통제를위한 학습 정책에서 수많은 성공이있었습니다. 주목할만한 예로는 일반적인 Atari 게임 (Mnih 외 2015)을위한 심층 모델없는 Q-learning, 로봇 모터 제어에 대한 종단 간 정책 검색 (Levineetal.2016), 포함 된 모델 예측 제어 ( Watter et al 2015), 검색과 결합 된 전략적 정책은 Go 게임에서 최고의 인간 전문가를 물리 쳤다 (Silver et al., 2016). 이러한 접근 방법의 성공에 중요한 부분은 최근 학습 내용을 확장 학습 및 심화 학습의 성과에 활용하는 것이 었습니다 (LeCun, Bengio, Hinton 2015). 도입 된 접근법 (Mnih et al., 2015)은 배치 RL을 사용하여이 데이터로부터 감독 방식으로 큰 길쌈 신경 네트워크를 훈련시키는 이전 경험 데이터 세트를 구축합니다. 현재의 경험보다는이 데이터 세트로부터 샘플링함으로써, 주 분배 편향에서 얻은 가치의 상관 관계가 완화되어 좋은 (많은 경우 슈퍼 로봇) 제어 정책을 유도합니다.

이러한 알고리즘을 데이터 센터, 자동 차량 (Hester and Stone 2013), 헬리콥터 (Abbeel 외 2007) 또는 추천 시스템 (Shani, Heckerman 및 Brafman 2005). 일반적으로 이러한 알고리즘은 시뮬레이션에서 매우 열악한 성능의 수백만 단계를 거쳐야 만 좋은 제어 정책을 학습합니다. 이 상황은 완벽하게 정확한 시뮬레이터가있는 경우 허용됩니다. 그러나 많은 실제 문제는 그러한 시뮬레이터와 함께 제공되지 않습니다. 대신, 이러한 상황에서 상담원은 실 제 영역에서 실제 행동에 대한 실제 결과를 학습해야하며, 이는 상담원이 학습 시작부터 양호한 온라인 성능을 요구합니다. 정확한 시뮬레이터를 찾기는 어렵지만 이러한 문제의 대부분은 이전 컨트롤러 (사람 또는 기계)에서 시스템이 작동하는 데이터를 합리적으로 잘 수행합니다. 이 작업에서 우리는이 데모 데이터를 사용하여 에이전트를 사전 학습하여 학습 시작 시점부터 잘 수행 할 수 있고 자체 생성 데이터로 계속해서 개선 할 수 있습니다. 이 프레임 워크에서 학습을 활성화하면 데모 데이터가 일반적이지만 정확한 시뮬레이터가 존재하지 않는 많은 실제 문제에 RL을 적용 할 수있는 가능성이 열립니다.

Over the past few years, there have been a number of successes in learning policies for sequential decision-making problems and control. Notable examples include deep model-free Q-learning for general Atari game-playing (Mnih et al. 2015), end-to-end policy search for control of robot motors (Levineetal.2016), model predictive control with embeddings (Watter et al. 2015), and strategic policies that combined with search led to defeating a top human expert at the game of Go (Silver et al. 2016). An important part of the success of these approaches has been to leverage the recent contributions to scalability and performance of deep learning (LeCun, Bengio, and Hinton 2015). The approach taken in (Mnih et al. 2015) builds a data set of previous experience using batch RL to train large convolutional neural networks in a supervised fashion from this data. By sampling from this data set rather than from current experience, the correlation in values from state distribution bias is mitigated, leading to good (in many cases, super-human) control policies.

It still remains difficult to apply these algorithms to real world settings such as data centers, autonomous vehicles (Hester and Stone 2013), helicopters (Abbeel et al. 2007), or recommendation systems (Shani, Heckerman, and Brafman 2005). Typically these algorithms learn good control policies only after many millions of steps of very poor performance in simulation. This situation is acceptable when there is a perfectly accurate simulator; however, many real world problems do not come with such a simulator. Instead, in these situations, the agent must learn in the real domain with real consequences for its actions, which requires that the agent have good online performance from the start of learning. While accurate simulators are difficult to find, most of these problems have data of the system operating under a previous controller (either human or machine) that performs reasonably well. In this work, we make use of this demonstration data to pre-train the agent so that it can perform well in the task from the start of learning, and then continue improving from its own self-generated data. Enabling learning in this framework opens up the possibility of applying RL to many real world problems where demonstration data is common but accurate simulators do not exist.

우리는 새로운 심층 강화 학습 알고리즘 인 DQfD (Deep Q-learning from Demonstrations)를 제안합니다. DQfD는 매우 적은 양의 데모 데이터를 활용하여 학습 속도를 대폭 향상시킵니다. DQfD는 처음에는 시차 (TD)와 감독 손실의 조합을 사용하여 데모 데이터에 전적으로 기울인다. 감시 손실은 알고리즘이 Demonstrator를 모방하는 것을 배울 수있게하며, TD 손실은 RL로 학습을 계속할 수있는 자체 일관성 값 함수를 학습하게합니다. 사전 교육 후에 에이전트는 학습 된 정책으로 도메인과 상호 작용하기 시작합니다. 에이전트는 데모와 자체 생성 데이터가 혼합되어 네트워크를 업데이트합니다. 실제로 데모와 자가 생성 데이터 간의 비율을 선택하는 것은 알고리즘의 성능을 향상시키는 데 중요합니다. 우리의 공헌 중 하나는이 비율을 자동으로 제어하기 위해 우선 순위가 매겨진 재생 메커니즘 (Schaul 외 2016)을 사용하는 것입니다. 제 만 단계 42 개 게임 (41)에, 그리고 (외 2,016 왕..; 반 하셀, Guez,은 2,016 Schaul 등 2016) DQfD는 우선 결투 더블 DQN (PDD DQN)를 이용하여 학습 순수한 보강 밖에서 수행 평균적으로 PDD DQN이 DQfD를 따라 잡으려면 8,300 만 단계가 필요합니다. 또한 DQfD는 42 게임 중 39 게임의 평균 점수에서 순수 모방 학습을 능가하며 42 게임 중 14 게임에서 가장 좋은 데모를 능가합니다. DQfD는 인간 Demonstration를 활용하여 42 개 게임 중 11 개에 최첨단 정책을 학습합니다. 마지막으로 DQfD가 데모 데이터를 DQN에 통합하기위한 세 가지 관련 알고리즘보다 우수한 성능을 보여줍니다.

We propose a new deep reinforcement learning algorithm, Deep Q-learning from Demonstrations (DQfD), which leverages even very small amounts of demonstration data to massively accelerate learning. DQfD initially pretrains solely on the demonstration data using a combination of temporal difference (TD) and supervised losses. The supervised loss enables the algorithm to learn to imitate the demonstrator while the TD loss enables it to learn a selfconsistent value function from which it can continue learning with RL. After pre-training, the agent starts interacting with the domain with its learned policy. The agent updates its network with a mix of demonstration and self-generated data. In practice, choosing the ratio between demonstration and self-generated data while learning is critical to improve the performance of the algorithm. One of our contributions is to use a prioritized replay mechanism (Schaul et al. 2016) to automatically control this ratio. DQfD out-performs pure reinforcement learning using Prioritized Dueling Double DQN (PDD DQN) (Schaul et al. 2016; van Hasselt, Guez, and Silver 2016; Wang et al. 2016) in 41 of 42 games on the first million steps, and on average it takes 83 million steps for PDD DQN to catch up to DQfD. In addition, DQfD out-performs pure imitation learning in mean score on 39 of 42 games and out-performs the best demonstration given in 14 of 42 games. DQfD leverages the human demonstrations to learn state-of-the-art policies on 11 of 42 games. Finally, we show that DQfD performs better than three related algorithms for incorporating demonstration data into DQN.

## Background

우리는이 연구를 위해 표준 Markov Decision Process (MDP) 법칙을 채택했다 (Sutton and Barto 1998). MDP는 상태 S의 집합, 행동 A의 집합, 보상 함수 R (s, a), 전이 함수 T (s, s)로 구성된 튜플 ⟨S, A, R, T, , a, s ') = P (s'| s, a), 그리고 할인 요인 γ. 에이전트는 각각의 상태 s ∈ S에서 ∈ A의 액션을 취한다.이 행동을 취하면 에이전트는 보상 R (s, a)를 받고 확률 분포 P (s '| s, a). 정책 π는 에이전트가 취할 각 상태에 대해 지정합니다. 에이전트의 목표는 에이전트의 수명 동안 예상되는 할인 된 총 보상을 최대화하는 동작으로 정책 π 매핑 상태를 찾는 것입니다. 주어진 상태 행위 쌍 (s, a)의 값 Qπ (s, a)는 정책 π에 뒤따를 때 (s, a)로부터 얻을 수있는 기대 미래 보상의 추정치이다. 최적 값 함수 Q * (s, a)는 모든 상태에서 최대 값을 제공하며 Bellman 방정식을 풀면 다음과 같이 결정됩니다.

그러므로 최적 정책 π는 π (s) = argmaxa∈A Q * (s, a)이다. DQN (Mnih et al., 2015)은 주어진 상태 입력 s에 대한 일련의 활동 값 Q (s, ·; θ)를 출력하는 심 신경 네트워크를 이용하여 값 함수 Q (s, a)를 근사화한다. 네트워크 매개 변수 이 작업을 수행하는 DQN에는 두 가지 주요 구성 요소가 있습니다. 첫째, 목표 Q 값이보다 안정적 이도록 정규 네트워크에서 τ 스텝마다 복사되는 별도의 목표 네트워크를 사용합니다. 둘째, 에이전트는 모든 경험을 재생 버퍼 Dreplay에 추가 한 다음 네트워크에서 업데이트를 수행하기 위해 균일하게 샘플링합니다.

이중 Q-learning 업데이트 (Hasselt, Guez 및 Silver 2016)는 현재 네트워크를 사용하여 다음 상태 값에 대한 argmax와 해당 작업 값에 대한 대상 네트워크를 계산합니다. 이중 DQN 손실은 다음과 같이 나타낼 수있다. 여기서, θ '는 매개 변수이고, Q (s, a, θ) target network의 amax = argmax Q (s, a; θ)이다. 이 두 변수에 사용 된 값 t + 1을 t + 1 함수로 분리하면 일반적인 Q-learning 업데이트로 생성되는 상향 바이어스가 감소합니다. 우선 순위 경험 재생 (Schaul 외 2016)은 DQN 에이전트를 수정하여 재생 버퍼에서 더 중요한 전환을 더 자주 샘플링합니다. 특정 전이 i를 표본 추출 할 확률은 우선 순위 P (i) = i α에 비례하며, 우선 순위 pi = | δi | + | δi는 k pk이다. Q * (s, a) = ER (s, a) + γ P (s '| s, a) max Q * (s', a ')입니다.

이 전이에 대해 계산 된 마지막 TD 오차 및 ǫ는 모든 전이가 일부 확률로 샘플링되도록하는 작은 양의 상수이다. 분포의 변화를 고려하기 위해 네트워크 업데이트는 중요도 샘플링 가중치 w = (1 · 1) β로 가중치가 부여됩니다. 여기서 NP는 재생 버퍼이며 β는 중요도가없는 중요도 샘플링 양을 제어합니다 β = 0 인 경우 샘플링하고 β = 1 인 경우 전체 중요도 샘플링을 수행합니다. β는 β0에서 1까지 선형으로 어닐링됩니다.

```text

We adopt the standard Markov Decision Process (MDP) formalism for this work (Sutton and Barto 1998). An MDP is defined by a tuple ⟨S,A,R,T,γ⟩, which consists of a set of states S, a set of actions A, a reward function R(s, a), a transition function T(s,a,s′) = P(s′|s,a), and a discount factor γ. In each state s ∈ S, the agent takes an action a ∈ A. Upon taking this action, the agent receives a reward R(s, a) and reaches a new state s′, determined from the probability distribution P(s′|s,a). A policy π specifies for each state which action the agent will take. The goal of the agent is to find the policy π mapping states to actions that maximizes the expected discounted total reward over the agent’s lifetime. The value Qπ (s, a) of a given state-action pair (s, a) is an estimate of the expected future reward that can be obtained from (s,a) when following policy π. The optimal value function Q∗(s, a) provides maximal values in all states and is determined by solving the Bellman equation:

The optimal policy π is then π(s) = argmaxa∈A Q∗(s, a). DQN (Mnih et al. 2015) approximates the value function Q(s, a) with a deep neural network that outputs a set of action values Q(s, ·; θ) for a given state input s, where θ are the parameters of the network. There are two key components of DQN that make this work. First, it uses a separate target network that is copied every τ steps from the regular network so that the target Q-values are more stable. Second, the agent adds all of its experiences to a replay buffer Dreplay , which is then sampled uniformly to perform updates on the network.

The double Q-learning up-date (van Hasselt, Guez, and Silver 2016) uses the current network to calculate the argmax over next state values and the target network for the value of that action. The double DQN loss is J (Q) =  R(s,a)+γQ(s ,amax;θ′)−Q(s,a;θ) 2, DQ t+1 t+1 where θ′ are the parameters of the target network, and amax = argmax Q(s , a; θ). Separating the value t+1 a t+1 functions used for these two variables reduces the upward bias that is created with regular Q-learning updates. Prioritized experience replay (Schaul et al. 2016) modifies the DQN agent to sample more important transitions from its replay buffer more frequently. The probability of sampling a particular transition i is proportional to its priority, P(i)=   i α ,wheretheprioritypi =|δi|+ǫ,andδi is k pk Q∗(s, a) = E  R(s, a) + γ   P (s′|s, a) max Q∗(s′, a′). 

the last TD error calculated for this transition and ǫ is a small positive constant to ensure all transitions are sampled with some probability. To account for the change in the distribution, updates to the network are weighted with importance sampling weights, w =(1 · 1 )β,whereN isthesizeof i NP(i) the replay buffer and β controls the amount of importance sampling with no importance sampling when β = 0 and full importance sampling when β = 1. β is annealed linearly from β0 to 1.
```

모방 학습은 주로 Demonstrator의 성과를 맞추는 것과 관련이 있습니다. 한 가지 인기있는 알고리즘 인 DAGGER (Ross, Gordon 및 Bagnell 2011)는 원래 상태 공간 외부의 전문가 정책을 폴링하여 새로운 정책을 반복적으로 생성하여 온라인 학습 관점에서 유효성 검사 데이터에 대해 후회하지 않습니다. DAGGER는 훈련 중에 상담원에게 추가적인 피드백을 제공 할 수 있도록 전문가의 도움을 필요로합니다. 또한, 모방과 보강 학습을 결합하지 않기 때문에 DQfD가 할 수있는 것처럼 전문가 이상의 수준으로 향상시키는 법을 배울 수는 없습니다.

Deep AggreVaTeD (Sun 외 2017)는 DAGGER를 확장하여 심층 신경 네트워크 및 연속 동작 공간에서 작동합니다. DAGGER처럼 항상 사용할 수있는 전문가가 필요 할뿐만 아니라 전문가는 작업 외에 가치 기능을 제공해야합니다. DAGGER와 마찬가지로 Deeply AgVaTeD는 모방 학습 만하고 전문가를 향상시키는 법을 배울 수 없습니다.

다른 대중적인 패러다임은 학습자가 정책을 선택하고 적의가 보상 기능을 선택하는 제로섬 게임을 설정하는 것입니다 (Syed and Schapire 2007, Syed, Bowling, Schapire 2008, Ho and Ermon 2016). 데모는 또한 고차원적이고 연속적인 로봇 제어 문제 (Finn, Levine 및 Abbeel 2016)에서 역 최적 제어를 위해 사용되었습니다. 그러나 이러한 접근법은 모방 학습을 수행하고 업무 보상에서 학습을 허용하지 않습니다.

최근 RL (Subramanian, Jr. 및 Thomaz 2016)의 어려운 Exploration 문제를 돕기위한 데모 데이터가 나와 있습니다. 이 결합 된 모조 및 RL 문제에 대한 최근의 관심이 또한있었습니다. 예를 들어, HAT 알고리즘은 인간 정책 (Taylor, Suay 및 Chernova 2011)에서 직접 지식을 전송합니다. 이 연구의 후속 조치는 RL 문제 (Brys 외 2015; Suay 외 2016)에서 보상을 형성하기 위해 전문가의 조언이나 시연을 어떻게 사용할 수 있는지를 보여 주었다.

```text
Imitation learning is primarily concerned with matching the performance of the demonstrator. One popular algorithm, DAGGER (Ross, Gordon, and Bagnell 2011), iteratively produces new policies based on polling the expert policy outside its original state space, showing that this leads to no-regret over validation data in the online learning sense. DAGGER requires the expert to be available during training to provide additional feedback to the agent. In addition, it does not combine imitation with reinforcement learning, meaning it can never learn to improve beyond the expert as DQfD can.

Deeply AggreVaTeD (Sun et al. 2017) extends DAGGER to work with deep neural networks and continuous action spaces. Not only does it require an always available expert like DAGGER does, the expert must provide a value function in addition to actions. Similar to DAGGER, Deeply AggreVaTeD only does imitation learning and cannot learn to improve upon the expert.
Another popular paradigm is to setup a zero-sum game where the learner chooses a policy and the adversary chooses a reward function (Syed and Schapire 2007; Syed, Bowling, and Schapire 2008; Ho and Ermon 2016). Demonstrations have also been used for inverse optimal control in high-dimensional, continuous robotic control problems (Finn, Levine, and Abbeel 2016). However, these approaches only do imitation learning and do not allow for learning from task rewards.

Recently, demonstration data has been shown to help in difficult exploration problems in RL (Subramanian, Jr., and Thomaz 2016). There has also been recent interest in this combined imitation and RL problem. For example, the HAT algorithm transfers knowledge directly from human policies (Taylor, Suay, and Chernova 2011). Follow-ups to this work showed how expert advice or demonstrations can be used to shape rewards in the RL problem (Brys et al. 2015; Suay et al. 2016).
```

다른 접근법은 경험을 샘플링하는 데 사용되는 정책을 형성하거나 (Cederborg 외 2015), 데모 Chemali 및 Lezaric 2015에서 정책 반복을 사용하는 것입니다.

우리 알고리즘은 Demonstrator가 사용하는 환경에서 보상을 받는 시나리오에서 작동합니다. 이 프레임 워크는 적절하게 (Piot, Geist, Pietquin 2014a)의 RLED (Reinforcement Learning with Expert Demonstrations)라고도하며 (Kim et al., 2013, Chemali and Lezaric 2015) 평가됩니다. 우리의 설정은 모형없는 설정에서 배치 알고리즘에서 TD와 분류 손실을 결합한다는 점에서 (Piot, Geist 및 Pietquin 2014a)와 유사합니다. 우리의 에이전트는 초기에 데모 데이터에 대한 사전 교육을 받았으며 자체 생성 데이터 배치가 시간이 지남에 따라 증가하고 깊은 Q 네트워크를 교육하기위한 경험 재생으로 사용된다는 점이 다릅니다. 또한, 각 미니 배치에서 데모 데이터의 양을 균형을 맞추기 위해 우선 순위가 지정된 재생 메커니즘이 사용됩니다. (Piot, Geist 및 Pietquin 2014b)는 감독 된 분류 손실에 TD 손실을 추가하면 보상이없는 경우에도 모방 학습이 향상된다는 흥미로운 결과가 나타납니다.

```text
A different approach is to shape the policy that is used to sample experience (Cederborg et al. 2015), or to use policy iteration from demonstrations Chemali and Lezaric 2015).

Our algorithm works in a scenario where rewards are given by the environment used by the demonstrator. This framework was appropriately called Reinforcement Learning with Expert Demonstrations (RLED) in (Piot, Geist, and Pietquin 2014a) and is also evaluated in (Kim et al. 2013; Chemali and Lezaric 2015). Our setup is similar to (Piot, Geist, and Pietquin 2014a) in that we combine TD and classification losses in a batch algorithm in a model-free setting; ours differs in that our agent is pre-trained on the demonstration data initially and the batch of self-generated data grows over time and is used as experience replay to train deep Q-networks. In addition, a prioritized replay mechanism is used to balance the amount of demonstration data in each mini-batch. (Piot, Geist, and Pietquin 2014b) present interesting results showing that adding a TD loss to the supervised classification loss improves imitation learning even when there are no rewards.
```

우리와 비슷한 동기가되는 또 다른 연구가 있습니다 (Schaal 1996). 이 작업은 로봇에 대한 실제 학습에 중점을두고 있으며 따라서 온라인 성능에도 관심이 있습니다. 우리의 작업과 마찬가지로 에이전트는 작업과 상호 작용하기 전에 데모 데이터로 에이전트를 사전 훈련합니다. 그러나 감독 된 학습을 사용하여 알고리즘을 사전 교육하지 않으며 사전 훈련이 장바구니에서 학습하는 데 도움이되는 사례를 하나만 찾을 수 있습니다.

원샷 모방 학습 (Duanetal.2017)에서 에이전트는 현재 상태뿐만 아니라 입력으로 전체 데모를 제공받습니다. 데모는 원하는 목표 상태를 지정하지만 다른 초기 조건에서 지정합니다. 상담원은 더 많은 데모에서 대상 액션으로 학습됩니다. 이 설정에서는 데모도 사용하지만 초기 조건 및 목표 상태가 다른 작업 배포가 필요하며 에이전트는 데모를 개선하지 못합니다.

AlphaGo (Silver et al., 2016)는 실제 작업과 상호 작용하기 전에 데모 데이터로부터 프리 트레이닝 작업에 대한 유사한 접근법을 사용합니다. AlphaGo는 감독 된 학습을 사용하여 전문가가 취한 행동을 예측하는 3 천만 개의 전문가 행동 데이터 집합에서 정책 네트워크를 먼저 교육합니다. 그런 다음이를 셀프 플레이 중에 계획 롤아웃과 결합 된 정책 그라디언트 업데이트를 적용하는 출발점으로 사용합니다. 여기서는 계획을 세울 수있는 모델이 없으므로 모델이없는 Q- 학습 사례에 중점을 둡니다.
Human Experience Replay (HER) (Hosu and Rebedea 2016)는 에이전트와 데모 데이터가 혼합 된 재생 버퍼에서 샘플링하는 알고리즘으로,이 방법과 유사합니다. 이익은 무작위 에이전트보다 약간 좋았으며 환경의 상태를 설정할 수있는 Human Checkpoint Replay라는 대안적인 접근 방식을 능가했습니다. 알고리즘은 두 데이터 세트에서 샘플링한다는 점에서 비슷하지만 에이전트를 사전 훈련 시키거나 감독 손실을 사용하지는 않습니다. 우리의 결과는 환경에 대한 완전한 액세스를 요구하지 않고도보다 다양한 게임에 비해 높은 점수를 보여줍니다. Replay Buffer Spiking (RBS) (Lipton et al., 2016)은 DQN 에이전트의 재생 버퍼가 데모 데이터로 초기화되는 것과 유사한 또 다른 접근법이지만 좋은 초기 성능을 위해 에이전트를 미리 훈련 시키거나 데모 데이터를 영구히 보관하지 않습니다.

우리와 가장 밀접하게 관련된 연구는 ADET (Accelerated DQN with Expert Trajectories) (Lakshminarayanan, Ozair 및 Bengio 2016)를 제시하는 워크숍 논문입니다. 또한 심층 Q- 학습 설정에서 TD와 분류 손실을 결합합니다. 그들은 훈련 된 DQN 에이전트를 사용하여 대부분의 게임에서 인간 데이터보다 나은 데모 데이터를 생성합니다. 또한 Demonstrator가 사용하는 정책을 도제 에이전트가 동일한 상태 입력 및 네트워크 아키텍처를 사용하므로 표현할 수 있습니다. 그들은 DQfD가 사용하는 큰 마진 손실보다는 크로스 엔트로피 분류 손실을 사용하며 환경과의 첫 번째 상호 작용에서 잘 수행되도록 에이전트를 사전 훈련하지 않습니다.

```text
Another work that is similarly motivated to ours is (Schaal 1996). This work is focused on real world learning on robots, and thus is also concerned with on-line performance. Similar to our work, they pre-train the agent with demonstration data before letting it interact with the task. However, they do not use supervised learning to pre-train their algorithm, and are only able to find one case where pre-training helps learning on Cart-Pole.
In one-shot imitation learning (Duanetal.2017), the agent is provided with an entire demonstration as input in addition to the current state. The demonstration specifies the goal state that is wanted, but from different initial conditions. The agent is trained with target actions from more demonstrations. This setup also uses demonstrations, but requires a distribution of tasks with different initial conditions and goal states, and the agent can never learn to improve upon the demonstrations.

AlphaGo (Silver et al. 2016) takes a similar approach to our work in pre-training from demonstration data before interacting with the real task. AlphaGo first trains a policy network from a dataset of 30 million expert actions, using supervised learning to predict the actions taken by experts. It then uses this as a starting point to apply policy gradient updates during self-play, combined with planning rollouts. Here, we do not have a model available for planning, so we focus on the model-free Q-learning case.

Human Experience Replay (HER) (Hosu and Rebedea 2016) is an algorithm in which the agent samples from a replay buffer that is mixed between agent and demonstration data, similar to our approach. Gains were only slightly better than a random agent, and were surpassed by their alternative approach, Human Checkpoint Replay, which requires the ability to set the state of the environment. While their algorithm is similar in that it samples from both datasets, it does not pre-train the agent or use a supervised loss. Our results show higher scores over a larger variety of games, without requiring full access to the environment. Replay Buffer Spiking (RBS) (Lipton et al. 2016) is another similar approach where the DQN agent’s replay buffer is initialized with demonstration data, but they do not pre-train the agent for good initial performance or keep the demonstration data permanently.

The work that most closely relates to ours is a workshop paper presenting Accelerated DQN with Expert Trajectories (ADET) (Lakshminarayanan, Ozair, and Bengio 2016). They are also combining TD and classification losses in a deep Q-learning setup. They use a trained DQN agent to generate their demonstration data, which on most games is better than human data. It also guarantees that the policy used by the demonstrator can be represented by the apprenticeship agent as they are both using the same state input and network architecture. They use a cross-entropy classification loss rather than the large margin loss DQfD uses and they do not pre-train the agent to perform well from its first interactions with the environment.
```

## Deep Q-Learning from Demonstrations

많은 실제 환경에서 보강 학습의 경우 이전 컨트롤러가 작동하는 시스템의 데이터에 액세스 할 수 있지만 시스템의 정확한 시뮬레이터에 액세스 할 수는 없습니다. 그러므로 에이전트는 실제 시스템을 실행하기 전에 데모 데이터에서 가능한 한 많이 배우기를 원합니다. 사전 교육 단계의 목표는 에이전트가 환경과 상호 작용을 시작하면이 상향 TD 업데이트와 함께 일 할 수 있도록 벨맨 방정식을 만족하는 함수의 값과 Demonstration을 모방하는 법을 배워야하는 것입니다. 이 미리 트레이닝 단계 동안, 제 샘플 시연 데이터로부터 미니 일괄와 위쪽 네 손실을 적용하여 네트워크를 기간 : 1 단계 이중 Q 학습 손실의 n 단계 이중 Q 학습 손실하는 슈퍼 큰 마진 분류 손실 및 네트워크 가중치 및 바이어스에 대한 L2 정규화 손실을 고려해야합니다. 감시 된 손실은 시연자의 행동 분류에 사용되며, Q학습 손실은 네트워크가 Bellman 방정식을 만족시키고 TD 학습을위한 출발점으로 사용될 수 있음을 보장합니다.
감독 된 손실은 사전 훈련이 어떤 영향을 미치기 위해 중요합니다. 데모 데이터는 반드시 상태 공간의 좁은 부분을 다루고 모든 가능한 행동을 취하지 않기 때문에 많은 상태 행동이 결코 취해지지 않았고 현실적인 가치에 근거 할 수있는 어떠한 데이터도 가지고 있지 않다. 우리는 다음 상태의 최대 값으로 만 Q-학습 업데이트로 네트워크를 사전 훈련을한다면, 네트워크는 이러한 접지 변수의 가장으로 업데이트 할 것이며,이 값을 전파 할 네트워크는 Q 기능을 전역 개 . 우리는 큰 마진 분류 손실 (Piot, Geist 및 Pietquin 2014a)을 추가합니다.

```text
In many real-world settings of reinforcement learning, we have access to data of the system being operated by its previous controller, but we do not have access to an accurate simulator of the system. Therefore, we want the agent to learn as much as possible from the demonstration data before running on the real system. The goal of the pre-training phase is to learn to imitate the demonstrator with a value function that satisfies the Bellman equation so that it can be updated with TD updates once the agent starts interacting with the environment. During this pre-training phase, the agent samples mini-batches from the demonstration data and updates the network by applying four losses: the 1-step double Q-learning loss, an n-step double Q-learning loss, a supervised large margin classification loss, and an L2 regularization loss on the network weights and biases. The supervised loss is used for classification of the demonstrator’s actions, while the Q-learning loss ensures that the network satisfies the Bellman equation and can be used as a starting point for TD learning.
The supervised loss is critical for the pre-training to have any effect. Since the demonstration data is necessarily covering a narrow part of the state space and not taking all possible actions, many state-actions have never been taken and have no data to ground them to realistic values. If we were to pre-train the network with only Q-learning updates towards the max value of the next state, the network would update towards the highest of these ungrounded variables and the network would propagate these values throughout the Q function. We add a large margin classification loss (Piot, Geist, and Pietquin 2014a):
```

여기서 aE는 전문가 Demonstrator가 상태에서 취한 행동이고 l (aE, a)는 a = aE 일 때 0이고 그렇지 않으면 양수인 마진 기능이다. 이 손실은 다른 행동의 가치를 적어도 Demonstrator의 행동 가치보다 낮은 마진으로 만든다. 이 손실을 추가하면 보이지 않는 행동의 가치를 합리적인 가치로 끌어 올리며 가치 기능에 의해 유도 된 욕심 많은 정책을 Demonstrator를 모방하게 만듭니다. 알고리즘이이 예비 손실만을 사용하여 사전 훈련 된 경우, 연속 상태와 Q네트워크 사이의 값을 제한하는 것은 TD와 함께 온라인 정책을 개선하는 데 필요한 Bellman 방정식을 충족시키지 못합니다 배우기.
n-step return (n = 10)을 추가하면 전문가의 궤도 값을 모든 이전 상태로 전파하여 사전 교육을 향상시킬 수 있습니다. n 단계 반환 값은 다음과 같습니다.

```text
where aE is the action the expert demonstrator took in state s and l(aE,a) is a margin function that is 0 when a = aE and positive otherwise. This loss forces the values of the other actions to be at least a margin lower than the value of the demonstrator’s action. Adding this loss grounds the values of the unseen actions to reasonable values, and makes the greedy policy induced by the value function imitate the demonstrator. If the algorithm pre-trained with only this supervised loss, there would be nothing constraining the values between consecutive states and the Q-network would not satisfy the Bellman equation, which is required to improve the policy on-line with TD learning.
Adding n-step returns (with n = 10) helps propagate the values of the expert’s trajectory to all the earlier states, leading to better pre-training. The n-step return is:
```

$$rt + γrt+1 + ... + γn−1rt+n−1 + maxaγnQ(st+n, a),$$

우리는 A3C (Mnih et al. 2016)와 유사하게 전방보기를 사용하여 계산합니다.
우리는 또한 네트워크의 가중치와 편향에 적용된 L2 정규화 손실을 추가하여 상대적으로 작은 데모 데이터 세트에 과도하게 끼워지는 것을 방지합니다. 네트워크를 업데이트하는 데 사용 된 전체 손실은 네 가지 손실 모두를 합한 것입니다.

```text
which we calculate using the forward view, similar to A3C (Mnih et al. 2016).
We also add an L2 regularization loss applied to the weights and biases of the network to help prevent it from over-fitting on the relatively small demonstration dataset. The overall loss used to update the network is a combination of all four losses:
```

$$J(Q) = JDQ(Q) + λ1Jn(Q) + λ2JE(Q) + λ3JL2(Q).$$

λ 매개 변수는 손실 사이의 가중치를 제어합니다. 섹션에서 이러한 손실 중 일부를 제거하는 방법을 검토합니다.
사전 교육 단계가 완료되면 에이전트가 시스템에서 작동하여 자체 생성 데이터를 수집하고이를 재생 버퍼 Dreplay에 추가합니다. 데이터가 찰 때까지 재생 버퍼에 데이터가 추가 된 후 에이전트는 해당 버퍼의 이전 데이터를 덮어 쓰기 시작합니다. 그러나 에이전트는 데모 데이터를 덮어 쓰지 않습니다. 비례 사전 표본 추출의 경우 에이전트와 데모 전환의 우선 순위에 다른 작은 양의 상수 ǫa와 ǫd가 추가되어 데모 대 에이전트 데이터의 상대적 샘플링을 제어합니다. 두 단계 모두에서 모든 손실이 악마 데이터에 적용되지만 감독 손실은 자체 생성 데이터 (λ2 = 0)에는 적용되지 않습니다.
전반적으로, DQfD (Deep Q-learning with Demonstration)는 PDD DQN과 6 가지 주요 방법이 다릅니다.

```text
The λ parameters control the weighting between the losses. We examine removing some of these losses in Section .
Once the pre-training phase is complete, the agent starts acting on the system, collecting self-generated data, and adding it to its replay buffer Dreplay. Data is added to the replay buffer until it is full, and then the agent starts overwriting old data in that buffer. However, the agent never over-writes the demonstration data. For proportional prioritized sampling, different small positive constants, ǫa and ǫd, are added to the priorities of the agent and demonstration transitions to control the relative sampling of demonstration versus agent data. All the losses are applied to the demonstration data in both phases, while the supervised loss is not applied to self-generated data (λ2 = 0).
Overall, Deep Q-learning from Demonstration (DQfD) differs from PDD DQN in six key ways:
```

- 데모 데이터 : DQf 데모
데이터는 재생 버퍼에 영구히 보관됩니다.
- 사전 교육 (Pre-training) : DQfD는 초기에 환경 데이터와의 상호 작용을 시작하기 전에 악마 데이터에 대해 전적으로 훈련한다.
범위.
- 감독 손실 : TD 손실 외에도,
Demonstrator의 행동 가치를 다른 행동 가치 (Piot, Geist 및 Pietquin 2014a)보다 높게 설정하는 Gin 감독 손실이 적용됩니다.
- L2 정규화 손실 :이 알고리즘은 데모 데이터에 대한 과도한 피팅을 방지하기 위해 네트워크 가중치에 L2 정규화 손실을 추가합니다.
- N-step TD 손실 : 에이전트는 1 단계 및 n 단계 반환의 조합을 통해 목표로 Q-network을 업데이트합니다.
- 데모 우선 순위 보너스 : 데몬스트레이션 전환의 우선 순위에는 샘플링되는 빈도를 높이기 위해 ǫd 보너스가 주어집니다.

```text
- Demonstration data: DQfDisgivenasetofdemonstration
data, which it retains in its replay buffer permanently.
- Pre-training: DQfD initially trains solely on the demonstration data before starting any interaction with the envi-
ronment.
- Supervised losses: In addition to TD losses, a large margin supervised loss is applied that pushes the value of the demonstrator’s actions above the other action val- ues (Piot, Geist, and Pietquin 2014a).
- L2 Regularization losses: The algorithm also adds L2 reg- ularization losses on the network weights to prevent over- fitting on the demonstration data.
- N-step TD losses: The agent updates its Q-network with targets from a mix of 1-step and n-step returns.
- Demonstration priority bonus: The priorities of demon- stration transitions are given a bonus of ǫd, to boost the frequency that they are sampled.
```

![alt text](images/dqfd1.png)

```python

```

## 실험 설정

우리는 아케이드 학습 환경 (ALE)에서 DQfD를 평가했다 (Bellemare et al., 2013). ALE는 DQN의 표준 벤치마킹이며 가장 우수한 학습 에이전트보다 여전히 뛰어난 성능을 나타내는 많은 게임을 포함하는 Atari 게임 세트입니다. 에이전트는 그레이 스케일로 변환 된 게임 화면의 다운 샘플링 된 84x84 이미지에서 Atari 게임을 재생하고 에이전트는이 프레임 중 4 개를 상태로 함께 스택합니다. 에이전트는 각 게임에 대해 18 가지 가능한 조치 중 하나를 출력해야합니다. 에이전트는 0.99의 할인 계수를 적용하고 모든 작업을 4 개의 Atari 프레임에 대해 반복합니다. 각 에피소드는 무작위로 시작 위치를 제공하기 위해 최대 30 번의 무단 조작으로 초기화됩니다. 보도 된 점수는 대리인이 내부적으로 보상을 나타내는 방식에 관계없이 Atari 게임의 점수입니다.
모든 실험에서 3 가지 알고리즘을 평가했으며 각 알고리즘은 평균 4 회에 걸쳐 시행되었습니다.

- 인간 데모가 포함 된 전체 DQfD 알고리즘
- 데모 데이터없이 PDD DQN 학습
- 데모 데이터로부터 모방을 감독하지 않고 모든 환경 상호 작용

우리는 6 개의 Atari 게임에서 모든 알고리즘에 대해 비공식적 인 매개 변수 튜닝을 수행 한 다음 전체 게임 세트에 대해 동일한 매개 변수를 사용했습니다. 알고리즘에 사용되는 매개 변수는 부록에 나와 있습니다. 우선 순위 지정 및 n 단계 반환 매개 변수에 대한 우리의 거친 검색은 DQfD 및 PDD DQN에 대해 동일한 최상의 매개 변수를 이끌어 냈습니다. PDD DQN은 시연 데이터, 사전 교육, 감독 손실 또는 정규화 손실이 없으므로 DQfD와 다릅니다. DQfD와 PDD DQN 비교를위한 더 나은 기준을 제공하기 위해 PDD DQN에서 n 단계 반환을 포함했습니다. 3 가지 알고리즘 모두 결투 상태 - 장점 컨벌루션 네트워크 아키텍처를 사용한다 (Wang et al., 2016).

감독 된 모방 비교를 위해, 우리는 DQfD에서 사용 된 동일한 네트워크 아키텍처와 L2 정규화를 사용하여 크로스 엔트로피 손실을 사용하여 시위자의 행동을 감독 분류했다. 모방 알고리즘은 TD 손실을 사용하지 않았습니다. 모방 학습은 사전 학습을 통해서만 학습되며 추가 상호 작용에서는 학습되지 않습니다.
우리는 42 개의 Atari 게임 중 무작위로 선택된 부분 집합에 대해 실험을 수행했습니다. 우리는 인간 플레이어가 각 게임을 3 번에서 12 번 재생하도록했습니다. 각 에피소드는 경기가 끝날 때까지 또는 20 분 동안 진행되었습니다. 게임을하는 동안 우리는 에이전트의 상태, 행동, 보상 및 종료를 기록했습니다. 인간 시위의 범위는 게임 당 5,574 건에서 75,472 건으로 다양합니다. DQfD는 AlphaGo (Silver et al., 2016)가 3 천만 개의 인간 전이로부터 배우고, DQN (Mnih et al. 2015)이 2 억 개 이상의 프레임에서 학습하기 때문에 다른 유사한 작업과 비교하여 매우 작은 데이터 세트에서 학습합니다. DQfD의 데모 데이터 세트가 작 으면 오버 피팅없이 좋은 표현을 배우는 것이 더 어려워집니다. 각 게임의 데모 점수는 부록에있는 표에 나와 있습니다. 우리의 인간 Demonstrator는 일부 게임 (예 : Priate vs Pitfall)에서 PDD DQN보다 훨씬 좋지만 많은 게임 (예 : Breakout, Pong)에서 PDD DQN보다 훨씬 나쁩니다.

우리는 인간 플레이어가 DQN보다 더 좋은 게임에서 DQN이 모든 보상을 1로 클리핑했기 때문에 DQN이 훈련을 받았다는 것을 알게되었습니다. 예를 들어 사립 탐정에서 DQN은 25,000 개의 ver- 인간의 시위자와 대리인에 의해 사용 된 보상 기능을보다 일관되게하기 위해, 우리는 잘려지지 않은 보상을 사용하고 로그 스케일을 사용하여 보상을 변환했다 : ragent = sign (r) · log (1 + | r |). 이 변환은 신경망이 배울 수있는 합리적인 척도를 유지하면서 개인 보상의 상대적 규모에 대한 중요한 정보를 전달합니다. 이러한 적응 보상은 우리의 실험에서 모든 알고리즘에 의해 내부적으로 사용됩니다. 결과는 아타리 문학에서 일반적으로 수행되는 것처럼 실제 게임 점수를 사용하여보고됩니다 (Mnih 외 2015).

We evaluated DQfD on the Arcade Learning Environment (ALE) (Bellemare et al. 2013). ALE is a set of Atari games that are a standard benchmark for DQN and contains many games on which humans still perform better than the best learning agents. The agent plays the Atari games from a down-sampled 84x84 image of the game screen that has been converted to greyscale, and the agent stacks four of these frames together as its state. The agent must output one of 18 possible actions for each game. The agent applies a discount factor of 0.99 and all of its actions are repeated for four Atari frames. Each episode is initialized with up to 30 no-op actions to provide random starting positions. The scores reported are the scores in the Atari game, regardless of how the agent is representing reward internally.
For all of our experiments, we evaluated three different algorithms, each averaged across four trials:
• Full DQfD algorithm with human demonstrations
• PDD DQN learning without any demonstration data
• Supervised imitation from demonstration data without
any environment interaction
We performed informal parameter tuning for all the algo- rithms on six Atari games and then used the same param- eters for the entire set of games. The parameters used for the algorithms are shown in the appendix. Our coarse search over prioritization and n-step return parameters led to the same best parameters for DQfD and PDD DQN. PDD DQN differs from DQfD because it does not have demonstra- tion data, pre-training, supervised losses, or regularization losses. We included n-step returns in PDD DQN to provide a better baseline for comparison between DQfD and PDD DQN. All three algorithms use the dueling state-advantage convolutional network architecture (Wang et al. 2016).
For the supervised imitation comparison, we performed supervised classification of the demonstrator’s actions using a cross-entropy loss, with the same network architecture and L2 regularization used by DQfD. The imitation algorithm did not use any TD loss. Imitation learning only learns from the pre-training and not from any additional interactions.
We ran experiments on a randomly selected subset of 42 Atari games. We had a human player play each game be- tween three and twelve times. Each episode was played ei- ther until the game terminated or for 20 minutes. During game play, we logged the agent’s state, actions, rewards, and terminations. The human demonstrations range from 5,574 to 75,472 transitions per game. DQfD learns from a very small dataset compared to other similar work, as Al- phaGo (Silver et al. 2016) learns from 30 million human transitions, and DQN (Mnih et al. 2015) learns from over 200 million frames. DQfD’s smaller demonstration dataset makes it more difficult to learn a good representation with- out over-fitting. The demonstration scores for each game are shown in a table in the Appendix. Our human demonstrator is much better than PDD DQN on some games (e.g. Pri- vate Eye, Pitfall), but much worse than PDD DQN on many games (e.g. Breakout, Pong).
We found that in many of the games where the human player is better than DQN, it was due to DQN being trained with all rewards clipped to 1. For example, in Private Eye, DQN has no reason to select actions that reward 25,000 ver- sus actions that reward 10. To make the reward function used by the human demonstrator and the agent more consis- tent, we used unclipped rewards and converted the rewards using a log scale: ragent = sign(r) · log(1 + |r|). This transformation keeps the rewards over a reasonable scale for the neural network to learn, while conveying important information about the relative scale of individual rewards. These adapted rewards are used internally by the all the al- gorithms in our experiments. Results are still reported using actual game scores as is typically done in the Atari litera- ture (Mnih et al. 2015).

먼저, Hero, Pitfall 및 Road Runner의 세 가지 게임에 대해 그림 1의 학습 곡선을 보여줍니다. 영웅에
그리고 Pitfall, 인간 시위는 DQfD를 가능하게한다.
이전에 출판 된 결과보다 높은 점수를 얻습니다. 두 게임의 동영상은 https://www.youtube.com/watch?v=JR6wmLaYuu4에서 확인할 수 있습니다. 영웅상에서 DQfD는 이전에 발표 된 결과뿐만 아니라 인간 시위보다 높은 점수를 얻습니다. Pitfall은 Atari 게임 중 가장 어려운 게임입니다.
매우 드문 긍정적 인 보상과 짙은 부정적인 보상. 이 게임에 대한 어떠한 긍정적 인 보상도 얻지 못했지만 DQfD의 최고 점수는 3 백만 단계 동안 평균 394.0입니다.

Road Runner에서 에이전트는 일반적으로 인간 플레이와 크게 다른 점수 악용으로 슈퍼 인간 정책을 학습합니다. 우리의 시위는 단지 인간이며 최대 점수는 20,200입니다. 로드 러너 (Road Runner)는 가장 작은 인간 시위 (단지 5,574 개의 전이)를 가진 게임입니다. 이러한 요인에도 불구하고, DQfD는 여전히 처음 3,600 만 단계에 대해 PDD DQN보다 높은 점수를 획득하고 그 이후에 PDD DQN의 성능과 일치합니다.
그림 1의 우측 서브 플롯은 데모 데 이터가 샘플링 된 방법과 균일 한 샘플링으로 샘플링되는 정도의 비율을 보여줍니다. Pitfall 및 Montezuma 's Revenge와 같은 가장 어려운 게임의 경우 데모 데이터가 시간이 지남에 따라 더 빈번하게 샘플링됩니다. 대부분의 다른 게임의 경우 비율은 각 게임마다 다른 거의 일정한 수준으로 수렴됩니다.
실제 업무에서 상담원은 첫 번째 조치에서 잘 수행되어야하며 신속하게 학습해야합니다. DQfD는 42 게임 중 41 번의 첫 번째 단계에서 PDD DQN보다 뛰어났습니다. 또한, 31 개의 게임에서 DQfD는 순전히 모방 학습보다 높은 성능으로 시작합니다. TD 손실의 추가는 요원이 악마 데이터를 더 잘 일반화하는 데 도움이됩니다. 평균적으로 PDD DQN은 DQfD의 성능을 8300 만 단계까지 초과하지 않으며 평균 점수에서 결코 능가하지 않습니다.
초기 성능 향상 외에도 DQfD는 가장 어려운 Atari 게임에 대해 더 나은 정책을 학습하기 위해 인간 시위를 활용할 수 있습니다. DQN, Double DQN, DQN, Dueling DQN, PopArt, DQN + CTS, DQN + PixelCNN (Mnih et al., 2015; van Hasselt, Guez, Silver 2016, Schaul et al 2016, Wang et al 2016, van Hasselt 외 2016, Ostrovskietal.2017). 우리는 DQfD 점수를 위해 4 종 이상의 평균 3 백만 창을 뽑았습니다. DQfD는 표 1에 나와있는 42 개 게임 중 11 개 게임에서 11 개 알고리즘보다 더 나은 점수를 얻습니다. A3C (Mnih 등 2016) 또는 Reactor (Gruslys 외 2017)와는 비교할 수 없습니다. 우리는 UNREAL (Jaderberg 외 2016)과 비교하지 않고 게임 당 최고 하이퍼 파라미터를 선택합니다. 이 사실에도 불구하고, DQfD는 여전히 10 개의 게임에서 최고의 UNREAL 결과를 능가합니다. 카운트 기반 탐색을 사용하는 DQN (Ostrovski 외 2017)은 가장 어려운 탐사 게임에서 최고의 결과를 얻기 위해 설계되고 달성됩니다. 6 개의 희소 한 보상에, 단단한 탐험 게임 둘 다 달린 산법, DQfD는 6 개의 게임의 4에 더 나은 정책을 배운다.
DQfD는 42 게임 중 29 게임에서 가장 최악의 데모 에피소드를 능가하며 Amidar, Atlantis, Boxing, Breakout, Crazy Climber, De-fender 게임에서 가장 좋은 데모 에피소드보다 더 나은 게임을 배웁니다. , Enduro, 낚시 더비, 영웅, James Bond, 쿵푸 마스터, Pong,로드 러너, Up N Down. 비교해 보면 순수 모방 학습은 모든 게임에서 논증 자의 수행보다 나쁩니다.
그림 2는 DQfD가 0-1로 설정된 DQfD와 DQfD가 최첨단 결과 인 Montezuma 's Revenge와 Q-Bert를 달성 한 두 게임에서의 비교를 보여줍니다. 예상대로,

First, we show learning curves in Figure 1 for three games: Hero, Pitfall, and Road Runner. On Hero
and Pitfall, the human demonstrations enable DQfD
to achieve a score higher than any previously pub- lished result. Videos for both games are available at https://www.youtube.com/watch?v=JR6wmLaYuu4. On Hero, DQfD achieves a higher score than any of the human demonstrations as well as any previously published result. Pitfall may be the most difficult Atari game, as it has
very sparse positive rewards and dense negative rewards. No previous approach achieved any positive rewards on this game, while DQfD’s best score on this game averaged over a 3 million step period is 394.0.

On Road Runner, agents typically learn super-human policies with a score exploit that differs greatly from hu- man play. Our demonstrations are only human and have a maximum score of 20,200. Road Runner is the game with the smallest set of human demonstrations (only 5,574 tran- sitions). Despite these factors, DQfD still achieves a higher score than PDD DQN for the first 36 million steps and matches PDD DQN’s performance after that.
The right subplot in Figure 1 shows the ratio of how of- ten the demonstration data was sampled versus how much it would be sampled with uniform sampling. For the most difficult games like Pitfall and Montezuma’s Revenge, the demonstration data is sampled more frequently over time. For most other games, the ratio converges to a near constant level, which differs for each game.
In real world tasks, the agent must perform well from its very first action and must learn quickly. DQfD performed better than PDD DQN on the first million steps on 41 of 42 games. In addition, on 31 games, DQfD starts out with higher performance than pure imitation learning, as the ad- dition of the TD loss helps the agent generalize the demon- stration data better. On average, PDD DQN does not surpass the performance of DQfD until 83 million steps into the task and never surpasses it in mean scores.
In addition to boosting initial performance, DQfD is able to leverage the human demonstrations to learn better policies on the most difficult Atari games. We compared DQfD’s scores over 200 million steps with that of other deep reinforcement learning approaches: DQN, Double DQN, Prioritized DQN, Dueling DQN, PopArt, DQN+CTS, and DQN+PixelCNN (Mnih et al. 2015; van Hasselt, Guez, and Silver 2016; Schaul et al. 2016; Wang et al. 2016; van Hasselt et al. 2016; Ostrovskietal.2017). We took the best 3 million step window averaged over 4 seeds for the DQfD scores. DQfD achieves better scores than these algorithms on 11 of 42 games, shown in Table 1. Note that we do not compare with A3C (Mnih et al. 2016) or Reactor (Gruslys et al. 2017) as the only published results are for human starts, and we do not compare with UNREAL (Jaderberg et al. 2016) as they select the best hyper-parameters per game. Despite this fact, DQfD still out-performs the best UNREAL results on 10 games. DQN with count-based explo- ration (Ostrovski et al. 2017) is designed for and achieves the best results on the most difficult exploration games. On the six sparse reward, hard exploration games both algorithms were run on, DQfD learns better policies on four of six games.
DQfD out-performs the worst demonstration episode it was given on in 29 of 42 games and it learns to play bet- ter than the best demonstration episode in 14 of the games: Amidar, Atlantis, Boxing, Breakout, Crazy Climber, De- fender, Enduro, Fishing Derby, Hero, James Bond, Kung Fu Master, Pong, Road Runner, and Up N Down. In compari- son, pure imitation learning is worse than the demonstrator’s performance in every game.
Figure 2 shows comparisons of DQfD with λ1 and λ2 set to 0, on two games where DQfD achieved state-of-the- art results: Montezuma’s Revenge and Q-Bert. 

예상대로 관리자가없는 손실없이 예비 교육을 수행하면 네트워크가 비 접지 Q- 학습 목표에 대해 교육을 받고 결과적으로 에이전트가 훨씬 낮은 성능으로 시작하여 개선 속도가 느려집니다. n-step TD 손실을 제거하는 것은 제한된 데모 데이터 세트로부터 학습하는 데 크게 도움이되므로 초기 성능에 거의 큰 영향을 미칩니다.

그림 2의 오른쪽 그림은 DQN에서 데모 데이터를 활용하기 위해 DQfD와 세 가지 관련 알고리즘을 비교합니다.

Replay Buffer Spiking (RBS) (Lipton 외 2016) Human Experience Replay (HER) (Hosu 및 Rebedea 2016)
• 전문가 궤적 (ADET)을 이용한 가속화 된 DQN (Lakshminarayanan, Ozair 및 Bengio 2016)
RBS는 처음에는 데모 데이터로 가득 찬 재생 버퍼가있는 PDD DQN입니다. HER는 데모 데이터를 보관하고 데모 및 에이전트 데이터를 각 미니 배치에 혼합합니다. ADET는 본질적으로 DQfD이며, 큰 마진 감독 손실이 교차 엔트로피 손실로 대체되었습니다. 결과는 두 가지 게임 모두에서 세 가지 접근 방식 모두가 DQfD보다 나 빠졌다는 것을 보여줍니다. DQfD와 ADET 모두 다른 두 알고리즘보다 훨씬 우수한 성능을 발휘하기 때문에 감독 손실을 갖는 것이 좋은 성능을 발휘하는 데 중요합니다. 모든 알고리즘은 DQfD와 동일한 데모 데이터를 사용합니다. 가능한 한 비교를 강하게 만들기 위해 우선 순위가 매겨진 재생 메커니즘과 이러한 모든 알고리즘에서 n 단계 반환을 포함했습니다.

As expected, pre-training without any supervised loss results in a network trained towards ungrounded Q-learning targets and the agent starts with much lower performance and is slower to im- prove. Removing the n-step TD loss has nearly as large an impact on initial performance, as the n-step TD loss greatly helps in learning from the limited demonstration dataset.

The right subplots in Figure 2 compare DQfD with three related algorithms for leveraging demonstration data in DQN:

Replay Buffer Spiking (RBS) (Lipton et al. 2016) Human Experience Replay (HER) (Hosu and Rebedea 2016)
• Accelerated DQN with Expert Trajectories (ADET) (Lakshminarayanan, Ozair, and Bengio 2016)
RBS is simply PDD DQN with the replay buffer initially full of demonstration data. HER keeps the demonstration data and mixes demonstration and agent data in each mini-batch. ADET is essentially DQfD with the large margin supervised loss replaced with a cross-entropy loss. The results show that all three of these approaches are worse than DQfD in both games. Having a supervised loss is critical to good perfor- mance, as both DQfD and ADET perform much better than the other two algorithms. All the algorithms use the exact same demonstration data used for DQfD. We included the prioritized replay mechanism and the n-step returns in all of these algorithms to make them as strong a comparison as possible.