# LSTM ITERATION NETWORKS: AN EXPLORATION OF DIFFERENTIABLE PATH FINDING

Lisa Lee∗ & Emilio Parisotto∗ & Devendra Singh Chaplot & Ruslan Salakhutdinov
Machine Learning Department
Carnegie Mellon University
Pittsburgh, PA 15213, USA
{lslee,eparisot,chaplot,rsalakhu}@cs.cmu.edu

## Abstract

우리의 동기는 계산량을 크게 증가시키지 않으면서 대규모 환경으로 Value Iteration을 확장하고 spatial invariance 과 unstable optimization와 같은 VIN (Value Iteration Networks) 고유의 문제를 수정하는 것입니다. 우리는 VIN과 그 단점을 개선 한 확장 된 VIN까지도 경험적으로 최적화하기가 어렵고 무작위적인 씨앗에 대한 민감성과 교육 과정에서의 불안정성을 보여줍니다. 또한 과거의 차별화 된 경로 계획 모듈에 활용 된 귀납적 편향 요소가 필요한지 여부를 조사하고 아키텍처가 엄격히 길 찾기 알고리즘과 유사하다는 요구 사항이 충족되지 않는다는 것을 입증합니다. 우리는 LSTM-Iteration Network라는 새로운 경로 계획 아키텍처를 설계함으로써 성공률, 교육 안정성 및 임의 시드에 대한 민감도와 같은 메트릭에서 VIN보다 우수한 성능을 달성합니다.

Our motivation is to scale value iteration to larger environments without a huge increase in computational demand, and fix the problems inherent to Value Iteration Networks (VIN) such as spatial invariance and unstable optimization. We show that VINs, and even extended VINs which improve some of their shortcomings, are empirically difficult to optimize, exhibiting instability during training and sensitivity to random seeds. Furthermore, we explore whether the inductive biases utilized in past differentiable path planning modules are even necessary, and demonstrate that the requirement that the architectures strictly resemble pathfinding algorithms does not hold. We do this by designing a new path planning architecture called the LSTM-Iteration Network, which achieves better performance than VINs in metrics such as success rate, training stability, and sensitivity to random seeds.

## 1 INTRODUCTION

다양한 강화 학습 영역에서 발생하는 일반적인 유형의 하위 작업은 경로 찾기입니다.
일부 시작 상태에서 부 목적에 도달하는 가장 짧은 일련의 동작을 찾습니다. 그 편재성 때문에 중요한 응용 프로그램 인 최근 작업 (Tamar et al., 2017)은 차별화 된 경로 탐색을 설계했습니다
모듈. 이 값 - 반복 네트워크 (VIN)는 값 반복 (VI)의 적용을 모방합니다.
미리 정의 된 MDP 매개 변수없이 2D 그리드 세계. VIN은
2D 미로와 3D 풍경에서 최적 경로 근처를 계산하는 전이 모델 P (s
0
| s, a)
선험적으로 제공되지 않았으며 배워야했습니다.
그들의 성공에도 불구하고, VIN에는 몇 가지 단점이 있습니다. 첫 번째는 전이 모델이 회선에 의해 매개 변수화 되었기 때문에 사실상 공간 변화가 없다는 사실입니다. 즉, 전환 모델은지도의 모든 그리드 단위에 대해 동일하므로 많은 애플리케이션에서 의미가 없습니다. 두 번째 단점은 길이 k의 최적 경로를 찾기 위해 k 반복의 순서로 요구할 수 있다는 것입니다. 즉, 잠재적으로 많은 수의 재귀 반복을 수행하여 목표 경로를 추정해야합니다. VIN은 에이전트가 모든 실제 환경 시간 단계에서 수행하는 내부 프로세스로 간주되기 때문에 많은 수의 반복을 수행하면 상담원의 계산 요구에 큰 부담을 줄 수 있습니다.
이 논문에서는 VIN 확장이 주요한 단점을 해결할 수 있는지를 경험적으로 평가한다. 즉, (1) 각 공간 위치의 필터 가중치가 맵 디자인의 인접 셀로부터 예측되는 "Hyper-VIN"을 만들기 위해 공간을 가중치로 풀다. , 및 (2) 정보의 흐름을 공간적으로 증가시키기 위해 커널 크기를 증가시킴으로써 최적의 경로를 발견하는 데 필요한 반복 횟수를 줄인다. Hyper-VIN은 일반적으로 VIN의 성능과 동등하지만,
값 반복에 따라 구조화되지 않은 아키텍처보다 나쁨. 또한 커널 크기가 커지면 VIN의 안정성이 떨어지며 원래 사용 된 표준 (3,3) 커널과 비교하여 성능이 크게 떨어지는 경우가 있음을 보여줍니다.
또한 VIN은 종종 훈련의 불안정성과 무작위 종자 감도에 시달리는 것을 보여줍니다. 이러한 최적화의 어려움으로 인해 우리는 VIN을 반복적 컨벌루션 네트워크로 재구성합니다. 이는 재래식 컨볼 루션 + 최대 풀링 반복 VIN 업데이트를 LSTM 업데이트와 같은 잘 알려진 반복 운영자로 대체 할 수있게합니다 (Hochreiter & Schmidhuber, 1997 ). 이러한 "LSTMIN"(Long Short-Term Memory Iteration Network)은 VIN이 값 반복을 수행하도록 유도 바이어스를 완화하는보다 일반적인 모델입니다. LSTMIN은 적어도 VIN보다 성능이 뛰어나고 성능이 뛰어나며 하이퍼 파라미터 감도가 적습니다
및 훈련 불안정성. 우리는 또한 3D VizDoom (Kempka et al., 2016) 환경에 대한 부록과 아래의 VIN 모듈을 대체 할 다운 스트림 작업에 대한 경험적 결과를 제시합니다.
QMDP-net (Karkus et al., 2017)은 LSTMIN을 사용했다.

A common type of sub-task that arises in various reinforcement learning domains is path finding:
finding a shortest set of actions to reach a subgoal from some starting state. Due to its ubiquity in important applications, recent work (Tamar et al., 2017) has designed a differentiable path-finding modules. These Value-Iteration-Networks (VINs) mimic the application of Value Iteration (VI) on a 2D grid world, but without pre-specified MDP parameters. VINs were shown to be capable of computing near optimal paths in 2D mazes and 3D landscapes where the transition model P(s
0
|s, a)
was not provided a priori and had to be learned.
Despite their successes, VINs have several shortcomings. The first is the fact that because the transition model is parameterized by a convolution, it is effectively spatially invariant. This means that the transition model is the same for every grid unit in the map, which does not make sense for many applications. The second shortcoming is that it can require on the order of k iterations to find an optimal path of length k. This means that a potentially large number of recursive iterations must be performed to get an estimate of the path to goal. Since the VIN is meant to be thought of as an inner process that the agent performs at every real environment time step, performing large numbers of iterations can cause a large strain on the agents computational demands.
In this paper, we empirically evaluate whether extending VIN can solve its major shortcoming, by (1) untying the weights spatially to create a “Hyper-VIN”, where the filter weights of each spatial position are predicted from neighboring cells in the map design, and (2) by increasing the kernel size in order to increase the flow of information spatially and thereby require fewer iterations to find optimal paths. We show that Hyper-VINs, generally on par with the performance of VINs, do
worse than architectures which are not structured according to value iteration. We also demonstrate that larger kernel sizes can decrease the stability of VIN, often degrading performance significantly compared to the standard (3,3) kernel used originally.
Additionally, we demonstrate that VIN is often plagued by training instability and random seed sensitivity. Owing to these optimization difficulties, we re-frame VIN as a recurrent-convolutional network, which enables us to replace the unconventional convolution+max-pooling recurrent VIN update with well-established recurrent operators such as the LSTM update (Hochreiter & Schmidhuber, 1997) . These “LSTMIN” (Long Short-Term Memory Iteration Network) are a more general model that relaxes the inductive bias that forces VINs to perform value-iteration. The LSTMIN is shown to perform at least as well or better than VIN and exhibits less hyperparameter sensitivity
and training instability. We also present empirical results in the appendix on 3D VizDoom (Kempka et al., 2016) environments, and a downstream task where we replace the VIN module within a
QMDP-net (Karkus et al., 2017) with an LSTMIN.

## 2 METHOD

아키텍처에서 Value-Iteration을 근사해야합니까? 한 가지 질문은 VIN에서 제공하는 귀납적 편향 요소가 필요한지 여부입니다. 대안적인보다 일반적인 아키텍처를 사용하는 것이 VIN보다 훨씬 더 잘 작동 할 수 있습니까? 각 반복에서 모든 공간 위치 $$(i', j')$$에서 재발 성 상태 $$(V_{i',j'}(t))$$를 갱신하면서 컨벌루션 - 재발 네트워크의 관점에서 VIN을 볼 수있다.

Is it necessary to approximate Value-Iteration in the architecture? One question to ask is whether the inductive biases provided by the VIN are even necessary: is it possible that using alternative, more general architectures might work significantly better than those of the VIN? We can view the VIN within the perspective of a convolutional-recurrent network, updating a recurrent state $$(V_{i',j'}(t))$$ at every spatial position $$(i', j')$$ at each iteration:

$$\bar{V} = max_a \biggl( \sum_{i,j}W^\bar{a}_{R,i,j} \bar{R}_{i' - i,j' - j} + W^\bar{a}_{V,i,j}\bar{V}^{(t-1)}_{i'-i, j'-j} \biggl) = \omega \big( W^\bar{a}_R\bar{R}_{[i',j',3]} + W^\bar{a}_V \bar{V}^{(t-1)}_{[i', j', 3]} \big) $$

위의 반복적 인 VIN 업데이트를 표준 반복 네트워크의 많은 문제점을 완화하는 잘 정의 된 LSTM 업데이트 (Hochreiter & Schmidhuber, 1997)로 쉽게 대체 할 수 있지만 입력 및 반복의 컨볼 루션 특성을 유지할 수 있습니다 무게 매트릭스 :

이 업데이트를 사용하는 길쌈 LSTM (Shi et al., 2015) 경로 계획 모듈을 "LSTM 반복 네트워크"(LSTMIN)라고 부릅니다.
Hyper-VIN : 원래의 Value Iteration Network의 문제는 회선을 사용하여 모델을 나타 내기위한 것이 었습니다. 이로 인해 모델이 효과적으로 공간적으로 변하지 않게되었습니다. 즉, VIN은 진정한 모델의 가치 반복과 동일한 방식으로 mazeworld를 진정으로 해결할 수 없습니다. 결과적으로 VIN은 주 공간에 대한 비선형 성을 처리 할 수있는 해결 방법을 학습합니다. 즉, 모든 벽 위치에 거대한 부정적인 보상을 할당합니다. 이것은 그림 7에 나와 있습니다. 벽과 비 벽 사이의 큰 보상 구배는 모델이 벽을 "방문하는"정책을 생성하는 것을 방해합니다. 이는 실제 모델에서는 불가능합니다. 또한 공간 회선 모델은 모든 미로에 대해 고정되어 있으며 불변합니다. 2D 환경의 각 MDP는 미로 디자인을 기반으로 다른 전환 커널을 필요로하므로 의미가 없습니다.
이 논문에서 우리는 먼저 공간 컨볼 루션의 가중치를 풀어서 미로 디자인에서 직접 풀리지 않은 회선 가중치를 예측함으로써이 문제를 완화하려고합니다. 우리는이 변형을 Hyper-VIN이라고 부르며 Hypernetworks의 명명 규칙을 채택했으며 다른 네트워크에서 예측 한 가중치가있는 네트워크를 사용하는 메커니즘도 사용했습니다 (Ha et al., 2017).
Hyper-VIN을 구현하기 위해 환경의 각 위치 (i, j)에 대해 입력 맵 디자인에서 길쌈 웨이트 행렬을 예측합니다. Hyper-VIN 업데이트 방정식은 다음과 같습니다.

커널 크기 VIN의 또 다른 문제점은 값 반복이 잠재적으로 환경에서 가장 긴 경로의 길이만큼 반복 횟수를 요구한다는 것입니다. 즉, VIN은 최적의 경로를 찾기 위해 상당한 깊이가 필요할 수도 있습니다. 또한 다양한 모델의 커널 크기 F를 늘려서 반복 횟수를 줄일 수 있는지 테스트했습니다.

where the notation X[i
0
,j0
,F ] means to take the image patch centered at position (i
0
, j0
) and kernel
size F. We can easily replace the recurrent VIN update above with the well-established LSTM update (Hochreiter & Schmidhuber, 1997), whose gated update alleviates many of the problems with standard recurrent networks, but we can still maintain the convolutional properties of the input and recurrent weight matrix:

We call the convolutional LSTM (Shi et al., 2015) path planning modules which use this update “LSTM Iteration Networks” (LSTMINs).
Hyper-VIN: An issue with the original Value Iteration Network was the use of convolutions to represent the model. This caused the model to effectively be spatially invariant, meaning VINs are incapable of truly solving mazeworld in the same way as value iteration on the true model. The result is that VINs learn a work-around that enables them to deal with non-linearities over the state space: it assigns a huge negative reward to every wall position. This is shown in Figure 7. The large reward gradient between walls and non-walls discourages the model from producing policies that “visit” walls, which would be impossible under the true model. Additionally, the spatial convolution model was fixed and invariant for all mazes, which does not make sense as each MDP in the 2D environments require a different transition kernel based on the maze design.
In this paper we try to alleviate this issue by, first, untying the weights of the spatial convolution and, second, predicting the untied convolution weights directly from the maze design. We call this variant the Hyper-VIN, adopting the naming convention from Hypernetworks, where they also used the mechanism of using a network with weights predicted from another network (Ha et al., 2017).
To implement the Hyper-VIN, for each position (i, j) in the environment we predict a convolutional weight matrix from the input map design. The Hyper-VIN update equation then becomes:


Kernel Size Another issue with the VIN was that Value Iteration potentially requires the number of iterations to be at least as large as the length of the longest path in the environment. This means that VINs might also require significant depth in order to find optimal paths. We additionally tested whether we could reduce the number of iterations by increasing the kernel size F of the various
models.

## 3 EXPERIMENTS

% Optimimal은 모델에 의해 추정 된 정책 하에서 예측 된 경로가 최적 길이를 갖는 전체 상태의 백분율이고 % Success는 모델에 의해 추정 된 정책 하에서 예측 경로를 갖는 전체 상태의 백분율입니다 목표 상태에 도달하십시오.
성능 : 15 × 15 2D 미로에 대한 VIN, Hyper-VIN 및 LSTMIN의 결과는 표 1에 요약되어 있습니다. 가장 좋은 결과는 학습 속도 α ∈ {0.001, 0.005, 0.01}, K ∈ {5, 10, 15, 20}, F ∈ {3, 5, 7, 9, 11}이다. LSTMIN, VIN 및 Hyper-VIN을 비교할 수 있도록 LSTMIN에 대해 150, VIN 및 Hyper-VIN에 대해 600의 숨겨진 차원을 사용했습니다. 이는 4 개의 게이트로 인해 LSTMIN에 포함 된 매개 변수가 약 4 배 증가했기 때문입니다 그것은 계산합니다.
NEWS mazes의 경우 LSTMIN이 VIN보다 월등히 뛰어나고 분산이 훨씬 낮고 대부분의 임의 씨앗이 비슷한 성능 값으로 수렴한다는 것을 알 수 있습니다. 이것은 LSTMIN이 최적화하기가 훨씬 쉬우 며 초기 랜덤 시드에 덜 민감하다는 것을 의미합니다. 또한 LSTMIN이 이러한 작업을 위해 명시 적으로 설계되지 않은 업데이트 방정식에도 불구하고 매우 효과적인 경로 계획 모듈임을 보여주는 거의 완벽한 성능을 얻습니다. 차동 드라이브 미로의 경우 VIN은 LSTMIN과 비슷한 거의 완벽한 결과를 얻습니다. 가까운 최적 임에도 불구하고
VIN 모델이 LSTMIN보다 지속적으로 수렴하는 속도가 느린 그림 2의 학습 곡선에서 알 수 있듯이 VIN 모델은 훈련하기가 더 어렵습니다. NEWS 미로에서 Hyper-VIN은 때로는이 최적의 솔루션을 복구하지 못하거나 결과를 얻지 못하는 경우가 있습니다.
가설 클래스 내의 실제 MDP 매개 변수에 대한 값 반복. 이 점과 무작위 종자보다 높은 결과의 차이를 고려할 때 SGD를 사용한 Hyper-VIN 교육은 상당한 어려움을 낳습니다.
반복 횟수 및 커널 크기 분석 : 변수 모델에 대한 반복 횟수 K 및 커널 크기 F의 영향을 평가했습니다. 그림 1은 F와 K의 다른 값에 대한 15 × 15 2D 차동 드라이브 미로에서의 VIN 및 LSTMIN의 최적 결과를 보여줍니다.이 그림에서 볼 수 있습니다
큰 F를 갖는 VIN의 최적화는 LSTMIN보다 훨씬 더 불안정하며 성능이 크게 변동하는 것을 관찰 할 수 있습니다. 이러한 불안정성에도 불구하고 최대 성능은 일반적으로 LSTMIN에 가깝고, 조기 정지를 사용하므로 성능을 복구 할 수 있습니다. 차동 드라이브 미로에 대한 유사한 결과가 그림 2에 나와 있습니다.이 그림에서 볼 수 있듯이 커널 크기를 늘리면 일반적으로 LSTMIN의 성능이 향상되고 5 회 반복만으로도 거의 완벽한 결과를 얻을 수 있습니다. 반면에 VIN의 커널 크기를 늘리면 반드시 성능에 도움이되는 것은 아니며 K가 작을수록 커널 크기가 작을수록 성능이 향상되는 경우가 많습니다. 큰 F를 사용하는 VIN은 불안정 해지고 성능이 크게 향상됩니다 .
Hyper-VIN v.s. VI : Hyper-VIN에 대해 질문 할 수있는 질문은 실제 알고리즘이 모델 클래스에 있기 때문에 모방하도록 설계된 실제 알고리즘보다 성능이 뛰어나다는 것입니다. 이것은 그러한 모듈들이 실제로 컴퓨팅하고 있는지에 대한 몇 가지 증거를 제공 할 것이다.
또는 그들이 반복적 인 네트워크처럼 행동했으며 덜 해석 할 수있는 내부 표현을 계산했는지 여부.
표 1은 비교적 직선적 인 모델을 가진 15 × 15 2D 미로에 대한 결과를 보여 주며, 벽이 그 방향에서 바로 인접한 위치를 차단하는지 각 위치에 대해서만 요구합니다. Hyper-VIN이 값 반복보다 나쁜 결과를 얻고 무작위 종자보다 큰 차이를 보임으로써 Hyper-VIN이 SGD를 사용하여 최적화하는 것을 상당히 어렵게한다는 것을 알 수 있습니다.

We use two metrics to compare the models: %Optimal is the percentage of total states whose predicted paths under the policy estimated by the model has optimal length, and %Success is the percentage of total states whose predicted paths under the policy estimated by the model reach the goal
state.
Performance: The results for VIN, Hyper-VIN, and LSTMIN on 15×15 2D mazes are summarized in Table 1. The best results were from hyperparameters obtained from a sweep over learning rate α ∈ {0.001, 0.005, 0.01}, K ∈ {5, 10, 15, 20}, and F ∈ {3, 5, 7, 9, 11}. In order to make comparison fair between LSTMIN, VIN and Hyper-VIN, we utilized a hidden dimension of 150 for LSTMIN and 600 for VIN and Hyper-VIN, owing to the approximately 4× increase in parameters that LSTMIN contains due to the 4 gates it computes.
For the NEWS mazes, we can see that LSTMIN significantly outperforms VIN and has much lower variance, with most random seeds converging to similar performance values. This suggests that LSTMIN is much easier to optimize, and is less sensitive to initial random seed. It also obtains near perfect performance, showing that LSTMINs are extremely effective path planning modules, despite the update equations not being explicitly designed for such a task. For the Differential Drive mazes, VIN achieves similar near-perfect results as the LSTMIN. Despite the near optimal
performance, the VIN models are more difficult to train, as evidenced by the learning curves in Figure 2, where the VIN is consistently slower to converge than the LSTMIN. On the NEWS mazes, Hyper-VIN sometimes fails to recover this optimal solution or even results close to it, despite having
value iteration on the true MDP parameters within its hypothesis class. Considering this and the high variance of its results over random seeds suggests that training Hyper-VIN with SGD is significantly challenging.
Analysis of iteration count and kernel size: We further evaluated the effect of iteration count K and kernel size F on the variable models. Figure 1 shows %Optimal results of VIN and LSTMIN on 15×15 2D Differential-Drive Mazes for different values of F and K. We can see from this figure
that optimization of the VIN with large F is significantly more unstable than LSTMIN, and we could observe performance oscillating significantly. Despite this instability, maximum performance is typically near LSTMIN, and since we use early stopping we can recover this performance. Similar results on the Differential Drive mazes are presented in Figure 2. We can see from this figure that increasing kernel size generally improves the performance of LSTMIN, and it achieves near perfect results even with only 5 iterations. On the other hand, increasing the kernel size of VIN does not necessarily help performance, and we see that smaller kernel sizes often have greater performance especially with increasing K. VIN with large F also seems to become more unstable and performance oscillates significantly over training epochs.
Hyper-VIN v.s. VI: A question that can be asked about Hyper-VIN is if they perform as well (or better) than the actual algorithms they were designed to mimic because the true algorithm is within the model class. This would provide some evidence whether such modules were actually computing
the value or whether they acted simply like recurrent networks and computed a less interpretable internal representation.
Table 1 shows results on 15×15 2D mazes, which have relatively straightforward models, only requiring for each position to see if a wall blocks the directly adjacent position in that direction. We can see that Hyper-VIN achieves results often worse than Value Iteration and with large variance over random seeds, demonstrating that the Hyper-VIN is significantly difficult to optimize using
SGD.