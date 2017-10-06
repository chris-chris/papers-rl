[\#deepmind](http://openresearch.ai/tags/deepmind)와 블리자드가 함께 pysc2라는 스타크래프트2 환경을 제공하면서 발표한 논문입니다.

* pysc2라는 스타크래프트2 python 환경 제공
* 기본적인 RL 모델링으로 미니게임, 풀게임 등에서 성능의 Baseline 제시
* 배틀넷에서 사람들이 경기한 데이터셋 공개
* 사람들의 실제 경기 데이터셋으로 pretrain, 혹은 supervised learning 했을 때의 학습에 대한 분석 제공

### Introduction

RL에서 스타크래프트2라는 새로운 환경을 연 것은 아래와 같은 도전적인 환경이라고 판단했다고 합니다.

* Multi Agent 문제임. 한 게임에 여러 참여자가 N vs N 형태로 참여하거나, 한 참여자가 컨트롤해야하는 유닛이 수백, 수천개.
* Imperfect Information Game. 즉, 기존의 바둑이나 아타리와 같이, 현재 게임 상태에 대해 전부 접근할 수 있는 것이 아님.
* 액션 스페이스가 매우 큼
* 게임이 수천 프레임 이상 지속됨. 액션이 이후 사건에 지속적으로 영향을 미침

즉 이러한 환경에서는 여러가지 관점들, Perception, Memory, Attention, Sequence Prediction, Modelling Uncertainty 등이 혼재되어 있다고 봅니다.

### Environments



[![](http://openresearch.ai/uploads/default/optimized/1X/5d23bc90a078dfe01b3f6cb1fb0ddae36e82a82b_1_690x430.jpg "image")image.jpg891x556 77.9 KB](http://openresearch.ai/uploads/default/original/1X/5d23bc90a078dfe01b3f6cb1fb0ddae36e82a82b.jpg)



제공하는 SC2LE라는 환경은 위와 같은 구조로 되어 있습니다.

* 리눅스 바이너리
* API
* PySC2

특히 리눅스 버전은 headless build로, 화면없이 실행할 수 있고, 랜덤시드를 고정할 수 있도록 제공합니다.

블리자드 스코어를 이용할 수 있도록 제공하는데 이는 게임이 종료되면 유닛 생성, 자원 채취에 따라 매겨지는 점수이나 이를 게임 중간 중간 엑세스 하도록 했습니다. RL 학습에 스코어로 쓸 수 있겠죠.



[![](http://openresearch.ai/uploads/default/optimized/1X/5b3ca608ba9722849df597c69c784e7071f36ea1_1_690x426.jpg "image")image.jpg872x539 146 KB](http://openresearch.ai/uploads/default/original/1X/5b3ca608ba9722849df597c69c784e7071f36ea1.jpg)



제공되는 환경에서는 ‘아직은’ RGB Screen에는 접근할 수 없고, 대신 위 그림과 같이 13개의 피쳐맵과, 사람이 해독가능한 형태로 이미지를 제공합니다. 이렇게 simplify 하고 나면 64x64이상의 크기면, 충분하다고 합니다.

### Actions



[![](http://openresearch.ai/uploads/default/optimized/1X/16373ef35e3d740e2568d5de4f6d2b3726c52691_1_690x378.jpg "image")image.jpg847x465 71.5 KB](http://openresearch.ai/uploads/default/original/1X/16373ef35e3d740e2568d5de4f6d2b3726c52691.jpg)



제공되는 액션은 사용자가 combination으로 하는 액션들을 적절히 섞었고, 현재 취할 수 있는 액션의 종류 역시도 제공합니다. 즉 실행 불가능한 액션, 예를 들어 마린을 선택하지 않았는데 움직이라고 명렁을 내리는 따위는 미리 막을 수 있습니다.

### Mini-games

7개의 미니게임을 탑재하여 제공합니다. RL학습의 베이스가 될 수 있을 것 같습니다.

### Baselines

A3C로 몇가지 네트워크 아키텍쳐를 이용해 베이스라인을 만들었습니다.  
A3C에 대한 설명은**\[여기\]**\([\(A3C\) Asynchronous Methods for Deep Reinforcement Learning](http://openresearch.ai/t/a3c-asynchronous-methods-for-deep-reinforcement-learning/25)\)을 참조하세요.



[![](http://openresearch.ai/uploads/default/optimized/1X/45ce38235110871ef9ed725990cfd42f4dd6c829_1_690x324.jpg "image")image.jpg902x424 62.5 KB](http://openresearch.ai/uploads/default/original/1X/45ce38235110871ef9ed725990cfd42f4dd6c829.jpg)



3가지 네트워크 아키텍쳐를 아래와 같습니다.

* Atari-net
  * 여러 Feature가 Convolutional Layer를 지난 후 Feature Vector로 Concate된 후 Value, Policy 를 출력하는 형태
* FullyConv
  * Atari-net에서 FC를 제거해 Feature Vector에 Spatial한 정보를 유지한 채로 Value, Policy를 출력하는 형태
* FullyConvLSTM
  * FullyConv의 마지막에 LSTM Cell을 추가하여 메모리 역할을 할 수 있도록 보조함



[![](http://openresearch.ai/uploads/default/optimized/1X/b25eba0464136234b36d762d94e221c5a67fc756_1_690x335.jpg "image")image.jpg873x425 112 KB](http://openresearch.ai/uploads/default/original/1X/b25eba0464136234b36d762d94e221c5a67fc756.jpg)



1vs1 풀게임에 대해서는 위와 같은 성능을 보여줬고, 왼쪽의 그래프 전부가 음수값으로 무조건 지는 것으로 converge되었으므로, 이 정도면 거의 학습이 안되었다고 볼 수 있겠습니다.



[![](http://openresearch.ai/uploads/default/optimized/1X/61e16bf6b70d0a9e1d5659c940acd62035b9252e_1_339x499.jpg "image")image.jpg662x974 181 KB](http://openresearch.ai/uploads/default/original/1X/61e16bf6b70d0a9e1d5659c940acd62035b9252e.jpg)



미니게임에 대해서는 조금 다른데, 특정 게임들에 대해 휴먼레벨 수준으로 잘 학습된 것을 볼 수 있습니다.

### Supervised Learning from Replays

수백만 건 이상의 리플레이를 제공할 예정이라고 합니다. 이 리플레이는 아래와 같은 연구에 쓰일 수 있습니다.

* 벤치마크 데이터
* Long term Correlation을 모델링
* Partial Observability를 극복하기 위한 방법 연구
* 에이전트와 사람의 결과 비교 등

또, pretrain된 모델이 더 좋은 성능을 낼 것이라고 예상한다고 밝혔습니다.

