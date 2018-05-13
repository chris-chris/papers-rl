이번 포스팅에선, 확률의 기초 중의 기초를 소개하도록 하겠습니다. 
기초 중의 기초인만큼, 아예 암기하는 것을 추천드립니다.

기말고사가 끝나고 머릿 속에서 사라진 수학 기초를 저랑 같이 공부해보시죠-

# 이산 확률 변수(Discrete Random Variable)


## 이산형 분포(Discrete Distribution)

이번 포스팅에서 다룰 수학 기초는 이산형 분포(Discrete Distribution)입니다.

### 1. 이항 분포(Binomial Distribution)

이항 분포는 우리가 고등학교 때 수학 시간에 배웠던 내용입니다. n번의 독립적 시행에서 각 시행이 확률 p를 가질 때의 이산 확률 분포입니다. 이항분포는 베르누이 시행이라고 불리기도 한다. n=1일 때 이항 분포는 베르누이 분포이다.

#### 이항 계수(Binomial Coefficient)

$$\frac{n!}{k!(n-k)!} = \binom{n}{k}$$

#### 이항 분포(Binomial Distribution)

$$Bin(k|n,\theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$$

$$mean = \theta, var = n\theta(1-\theta)$$

### 2. 베르누이 분포(Bernoulli Distribution)

$$Ber(x|\theta) = $$

### 3. 다항 분포(Multinomial Distribution)

#### 다항 계수(Multinomial Coefficient)



$$Mu(x|n,\theta) = \binom{n}{x_1 ... x_K}\prod_{j=1}^K\theta_j^{x_j}$$

### 4. 멀티누이 분포(Multinoulli Distribution)

$$Mul(x|1,\theta) = \prod^K_{j=1}\theta_j^{\mathbb{1}(x_j=1)}$$

$$Mul(x|1,\theta) = Cat(x|\theta)$$

### 5. 푸아송 분포(Poisson Distribution)

$$Poi(x,\lambda) = e^{-\lambda} \frac{\lambda^{x}}{x!}$$

### 6. 경험적 분포(Empirical Distribution)

$$P_{emp}(A) = \frac{1}{N}\sum^N_{i=1}\delta_{x_i}(A)$$


