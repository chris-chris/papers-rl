#Model-Free Prediction

## Introduction

### This lecture
Model-free prediction
Estimate the value function of an unknown MDP

예측만에 집중한다. 어떻게 행동을 예측할 것인가? 
알수 없는 MDP의 가치 함수를 측정한다

### Next lecture
Model-free control
알수 없는 MDP의 가치 함수를 최적화한다

## Monte-Carlo Learning

MC 방법은 에피소드 경험으로부터 직접적으로 학습한다
MC는 model-free이다. : MDP 전환이나 보상에 대한 지식이 없다
MC는 완전히 끝난 에피소드로부터 배운다. no bootstrapping : 추정하여 평가하지 않는다
MC는 단순한 아이디어를 사용한다. 가치는 평균이다
Caveat: MC는 에피소드 단위의 MDP에만 적용 가능하다. 모든 에피소드는 종료되어야만 한다.


## Monte-Carlo Policy Evaluation

- 목표: 정책 $$\pi$$의 에피소드 경험으로부터 $$v_\pi$$를 학습시킨다

- 다시 한번, 보상은 총 할인된 보상이다.

- 다시 한번, value function은 예상되는 가치이다.

- MC 정책 평가는 예측 보상(expected return)이 아닌, 경험적 평균(empirical mean)을 사용한다.


## First-Visit MC 정책 평가

- 상태 s를 평가하기 위해선
- 첫번째로만 t 시점에서 상태 s가 방문되었을 때
- 해당 상태를 방문한 횟수를 증가시킨다
$$N(s)++$$
- 총 보상을 증가시킨다
$$S(s) <- S(s) + G_t$$
- 가치 함수는 이렇게 구해진다
$$V(s) = S(s) / N(s)$$
- 큰 수의 법칙에 따라
$$V(s) -> v_\pi(s) as N(s) -> \infty$$


## Every-Visit MC 정책 평가

- 상태 s를 평가하기 위해선
- 매번 t 시점에서 상태 s가 방문되었을 때
- 해당 상태를 방문한 횟수를 증가시킨다
$$N(s)++$$
- 총 보상을 증가시킨다
$$S(s) <- S(s) + G_t$$
- 가치 함수는 이렇게 구해진다
$$V(s) = S(s) / N(s)$$
- 큰 수의 법칙에 따라
$$V(s) -> v_\pi(s) as N(s) -> \infty$$

## 점진적인 평균(Incremental Mean)


## 점진적인 몬테카를로 업데이트(Incremental Monte-Carlo Updates)

모든 상태 $$S_t$$에 대해서 보상 $$G_t$$

$$N(S_t) <- N(S_t) + 1$$
$$V(S_t) <- V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$$

non-stationary problems에 있어서 

## Temporal-Difference Learning

- TD methods learn directly from episodes of experience
- TD 방법은 model-free이다. 
- TD는 불완전한 에피소드로부터 배운다. 추정을 통해서
- 추정을 통해 업데이트를 한다.

## MC와 TD

- 목표: 정책 $$\pi$$ 아래에서 나온 경험을 통해 $$v_\pi$$를 훈련시킨다.
- 점진적인 every-visit Monte-Carlo
	- V(S_t)를 실제 보상 G_t를 통해 업데이트한다.

- 가장 단순한 TD 학습 알고리즘: TD(0)
	- $$V(S_t)$$를 추정된 다음 가치 TD-Target $$R_{t+1} + \gamma V(S_{t+1})$$ 으로 업데이트 시킨다.

	- $$R_{t+1} + \gamma V(S_{t+1})$$ : TD target
	- $$\delta_t = R_{t+1} +  \gamma V(S_{t+1}) - V(S_t)$$: TD error

### 