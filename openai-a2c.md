# OpenAI-A2C & ACKTR

ACKTR \( "actor"라고 발음 함\) - Kronecker 인수 트러스트 영역을 사용하는 Actor Critic - 토론토 대학교와 뉴욕 대학교의 연구원이 개발했으며, OpenAI는 Baselines 구현을 발표하기 위해 협력했습니다.저자[는](https://arxiv.org/abs/1708.05144)ACKTR을[사용](https://arxiv.org/abs/1708.05144)하여 시뮬레이트 된 로봇 \(픽셀을 입력으로, 연속 동작 공간\) 및 Atari 에이전트 \(입력 및 불연속 동작 공간으로 픽셀 사용\)에 대한 제어 정책을 학습합니다.

ACKTR은[Actor - Critic 방법](https://arxiv.org/abs/1602.01783), 보다 일관된 개선을위한 [신뢰 영역 최적화](https://arxiv.org/abs/1502.05477), [분산 아키텍쳐](https://jimmylba.github.io/papers/nsync.pdf), [크로네커](https://arxiv.org/abs/1503.05671), [인수 분해](https://arxiv.org/abs/1602.01407)를 통해 샘플 효율과 확장 성을 향상시키는세 가지 기술을 결합합니다.

# 샘플 및 계산 효율성 {#sampleandcomputationalefficiency}

기계 학습 알고리즘의 경우 샘플 복잡성과 계산 복잡성과 같은 두 가지 비용을 고려해야합니다.샘플 복잡도는 상담원과 그 환경 간의 상호 작용 시간 수를 의미하며 계산 복잡성은 수행해야하는 수치 작업의 양을 나타냅니다.

ACKTR은 A2C와 같은 1 차 방법보다 더 좋은 표본 복잡성을 갖습니다. 그 이유는 그라디언트 방향 \(또는 ADAM 에서처럼 재조정 버전\)이 아닌_자연스러운 그래디언트_방향에서 한 걸음 걸리기 때문입니다.자연스러운 그라디언트는 KL- 발산을 사용하여 측정 된 네트워크 출력 분포의 단위 변화 당 최대의 \(즉각적인\) 개선을 달성하는 매개 변수 공간에서의 방향을 제공합니다.KL 분산을 제한함으로써 새로운 정책이 이전 정책과 근본적으로 다르게 동작하지 않도록함으로써 성능 저하를 초래할 수 있습니다.

계산상의 복잡성에 관해서는, ACKTR에 의해 사용 된 KFAC 업데이트는 업데이트 단계마다 표준 그래디언트 업데이트보다 10-25 % 더 비쌉니다.이것은 TRPO \(즉, Hessian-free 최적화\)와 같은 방법과 대조를 이룹니다.이 방법은보다 고가의 공액 - 기울기 계산이 필요합니다.

다음 비디오에서는 게임 Q-Bert를 해결하기 위해 ACKTR로 훈련 된 요원과 A2C에서 훈련 한 요원 간의 다양한 타임 스텝에서 비교를 볼 수 있습니다.ACKTR 에이전트는 A2C로 훈련 된 에이전트보다 높은 점수를 얻습니다.

| A2C | ACKTR |
| :--- | :--- |


\* ACKTR로 훈련 된 요원 \(오른쪽\)은 A2C \(왼쪽\)와 같은 다른 알고리즘으로 훈련 된 요원보다 짧은 시간 내에 더 높은 점수를 얻습니다. \*

# 기준선 및 벤치 마크 {#baselineandbenchmarks}

이 릴리스에는 A2A 릴리스뿐만 아니라 ACKTR의 OpenAI 기본 릴리스가 포함됩니다.

우리는 또한다양한 작업에서 A2C,[PPO](https://arxiv.org/abs/1707.06347)및[ACER](https://arxiv.org/abs/1611.01224)에 대한ACKTR을 평가하는[벤치 마크](https://github.com/openai/baselines-results)를발표하고 있습니다.다음 플롯에서 다른 알고리즘과 비교 한 49 건의 Atari 게임에 대한 ACKTR의 성능을 보여줍니다 : A2C, PPO, ACER.ACKTR의 하이퍼 파라미터는 하나의 게임 인 Breakout에서만 ACKTR의 저자에 의해 조정되었습니다.  
![](https://blog.openai.com/content/images/2017/08/Pasted-image-at-2017_08_18-09_01-AM.png)

ACKTR 성능은 각 배치의 정보로부터 그래디언트 추정치를 도출 할뿐만 아니라 정보를 매개 변수 공간의 로컬 곡률을 근사화하는 데 사용하기 때문에 배치 크기와도 잘 맞 춥니 다.이 기능은 대용량 배치 크기가 사용되는 대규모 분산 교육에 특히 유리합니다.  
![](https://blog.openai.com/content/images/2017/08/WX20170817-220206@2x-3.png)

# A2C 및 A3C {#a2canda3c}

비동기 Advantage Actor Critic \(비동기 Advantage Actor Critic\) 방법 \(A3C\)은이[보고서](https://arxiv.org/abs/1602.01783)가 발표 된이래 매우 큰 영향을 미쳤습니다.이 알고리즘은 몇 가지 핵심 아이디어를 결합합니다.

* 고정 길이의 경험 세그먼트 \(예 : 20 timesteps\)에서 작동하고 이러한 세그먼트를 사용하여 반품 및 이점 기능의 평가자를 계산하는 업데이트 체계.
* 정책과 가치 기능 사이의 계층을 공유하는 아키텍처.
* 비동기 업데이트.

이 논문을 읽은 후에 인공 지능 연구원은 비동기가 성능 향상으로 이어 졌는지 궁금해했다. \(예를 들어 "추가 된 잡음은 정규화 또는 탐색을 제공 할 것인가?"\) 아니면 CPU 기반 이행.

연구원은 비동기 구현의 대안으로, 모든 액터에 대해 평균을 내고 업데이트를 수행하기 전에 각 액터가 경험 세그먼트를 끝내기를 기다리는 동기식 결정 론적 구현을 ​​작성할 수 있음을 발견했습니다.이 방법의 한 가지 이점은 대형 배치 크기에서 가장 잘 수행되는 GPU를보다 효과적으로 사용할 수 있다는 것입니다.이 알고리즘은 자연스럽게 A2C라고 불리며 우대 Actor-Critic의 약자입니다.\(이 용어는[여러 논문](https://arxiv.org/abs/1611.05763)에서 사용되었습니다.\)

동기식 A2C 구현은 비동기식 구현보다 뛰어납니다. 비동기식으로 발생하는 잡음이 성능상의 이점을 제공한다는 어떠한 증거도 보지 못했습니다.이 A2C 구현은 단일 GPU 시스템을 사용하는 경우 A3C보다 비용 효율적이며 더 큰 정책을 사용하는 경우 CPU 전용 A3C 구현보다 빠릅니다.

우리는 A2C를 사용하여 Atari 벤치 마크에서 피드 포워드 통신 및 LSTM을 교육하기위한베이스 라인에 코드를 포함 시켰습니다.
