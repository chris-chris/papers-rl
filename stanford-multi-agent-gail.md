# MULTI-AGENT GENERATIVE ADVERSARIAL IMITATION LEARNING

Jiaming Song, Hongyu Ren, Dorsa Sadigh & Stefano Ermon
Computer Science Department
Stanford University

## ABSTRACT

We propose a new framework for multi-agent imitation learning for general Markov games, where we build upon a generalized notion of inverse reinforcement learning.
We introduce a practical multi-agent actor-critic algorithm with good empirical performance. Our method can be used to imitate complex behaviors in highdimensional environments with multiple cooperative or competitive agents.

## 1 MARKOV GAMES AND IMITATION LEARNING

We consider an extension of Markov decision process (MDPs) called Markov (stochastic) games (Littman, 1994; Lowe et al., 2017). A Markov game (MG) for N agents is defined by N sets of states {Si} N i=1 and N sets of actions {Ai}N i=1 for each agent respectively. We let S = Qi Si
represent the set of states. The function T : S × A1 × · · · × AN → P(S) describes the transition between states, where P(S) denotes the set of probability distributions over S. Each agent i receives a reward given by a function ri : S × A1 × · · · × AN → R, and aims to maximize its own total expected return Ri = PH t=0 γ t ri,t, where γ is the discount factor and H is the time horizon, by selecting actions through a stochastic policy πi : Si × Ai → [0, 1].
Our goal in imitation learning is to learn policies that behave similar to collected expert demonstrations under the assumption that we don’t have access to the reward values. The expert demonstrations form a dataset of state-action pairs D = {(sj , aj )}M j=1, which are collected by sampling s0 ∼ η(s), at = πE(at|st), st+1 ∼ T(st+1|st, at). We assume all experts operate in the same environment, and that once we obtain demonstrations D, we cannot ask for further expert interactions with the environment (unlike in (Ross et al., 2011) or (Hadfield-Menell et al., 2016)).

## 2 MULTI-AGENT GENERATIVE ADVERSARIAL IMITATION LEARNING

In this work, we consider a distributed setting for imitation learning, where we leverage ideas from (Ho & Ermon, 2016) for a novel algorithm for Multi-Agent Generative Adversarial Imitation Learning (MAGAIL). For each agent i, we have a discriminator (denoted as Dωi ) mapping state action-pairs to scores. The discriminators are optimized to discriminate expert demonstrations from behaviors produced by πi . Implicitly, Dωi plays the role of a reward function for the generator, which in turn attempts to train the agent to maximize its reward thus fooling the discriminator. Hence we optimize the following objective for each agent i:



This process is similar to coordinate descent in the sense that each agent updates their policies independently. We update πθ through multi-agent reinforcement learning. It is possible to encode our inductive bias about the reward structure through the discriminator learning process. We consider three types of priors for common multi-agent settings.


Fully Cooperative. The easiest case is to assume that the agents are fully cooperative, i.e. they share the same reward function. One could argue this corresponds to the GAIL case, where the RL procedure is operated on multiple agents, so conclusions in (Ho & Ermon, 2016) would still hold.

Decentralized Rationality. We make no assumptions over the correlation between the rewards, yet we assume experts are acting rationally under a reward function that depends only on their own observations and actions. This setup corresponds to having one discriminator for each agent which
discriminates the trajectories as observed by agent i.
Zero Sum. Assume there are two agents that receive opposite rewards, so r1 = −r2. An adversarial training procedure can be designed using the following fact:

where V (π1, π2) = Eπ1,π2 [r1(s, a)] is the expected outcome for agent 1 when the agents choose policies π1 and π2 respectively. The discriminator could maximize the reward for trajectories in (πE1 , π2) and minimize the reward for trajectories in (π2, πE1).

## 3 MULTI-AGENT ACTOR-CRITIC WITH KRONECKER FACTORS

Once we obtain rewards from the discriminator, we wish to use an algorithm for multi-agent RL that has good sample efficiency in practice, which then can result in effective updates at each step. We design our MARL algorithm based on Actor-Critic with Kronecker-factored Trust Region (ACKTR, Wu et al. (2017)), a state-of-the-art actor-critic method in deep RL. 
Our algorithm, which we refer to as Multi-agent Actor-Critic with Kronecker-factors (MACK), uses the framework of centralized training with decentralized execution (Foerster et al., 2016; Lowe et al., 2017); policies are trained with additional information to reduce variance (only in training). We let the advantage function of every agent be a function of all agents’ observations and actions:



where Vπiφi(ok, a−i) is an estimated value function for i, when other agents take actions a−i. This value function treats other agents as a part of the environment, whereas (o−i, a−i) serves as additional information for variance reduction. The policy gradient for agent i is written as:

We use K-FAC to update both θ and φ, but do not use trust regions to schedule the learning rate, since we find a linearly decaying learning rate schedule to have similar performance.


## 4 EXPERIMENTS

We consider the two-dimensional particle environment proposed in (Lowe et al., 2017), which consists of N agents and L landmarks. Agents may take physical actions and communication actions that get broadcasted to other agents. We consider two cooperative settings (all agents attempt to maximize a shared reward) and two that are competitive (agents have conflicting goals). All these environments have an underlying true reward that allows us to estimate the performance of our agents.
Cooperative Communication: two agents must cooperate to reach one of three colored landmarks.
We consider an asymmetric observation setting: One agent (“speaker”) knows the goal but cannot move, so it must convey the message to the other agent (“listener”) that can move but does not observe the goal.
Cooperative Navigation: three agents must cooperate through physical actions to reach three landmarks; ideally, each agent should cover a single landmark.
Keep-Away: two agents have contradictory goals, where agent 1 tries to reach one of the two targeted landmarks, while agent 2 (the adversary) tries to keep agent 1 from reaching its target. The adversary does not observe the target, so it can only act based on agent 1’s actions.
Predator-Prey: three slower cooperating adversaries must chase the faster agent; the adversaries are rewarded by touching the faster agent while that agent is penalized.

For the cooperative tasks, we use an analytic expression defining the expert policy. For the competitive tasks, we use MACK to train expert policies based on the true underlying rewards. We then use the expert policies to simulate trajectories D, which can then be used as demonstrations for
imitation learning, where we assume both the underlying rewards and the expert policy are unknown.
Following (Li et al., 2017), we pretrain MAGAIL using behavior cloning as initialization to reduce sample complexity.

## 4.1 COOPERATIVE TASKS

We evaluate performance in cooperative tasks via the average expected reward obtained by all the agents in an episode. In this environment, the starting state is randomly initialized, so generalization is crucial. We consider 100 to 400 episodes as expert demonstration, and display the performance of cooperative MAGAIL, decentralized MAGAIL and behavior cloning (BC) in Figure 1.
Naturally, the performance of BC and MAGAIL increases with more expert demonstrations. MAGAIL performs consistently better than BC in all the settings. Moreover, the cooperative MAGAIL performs slightly better than decentralized MAGAIL due to the better prior, but decentralized MAGAIL still manages to achieve reasonable performance.

### 4.1.1 COMPETITIVE TASKS

We consider all three types of Multi-Agent GAIL (cooperative, decentralized, zero-sum) and BC in both competitive tasks. Since there are two opposing sides, it is hard to measure performance directly.
Therefore, we compare the rewards by letting (agents trained by) BC play against (adversaries trained by) other methods, and vice versa. From Figure 2, decentralized and zero-sum MAGAIL often perform better than centralized MAGAIL and BC, which suggests that the selection of the suitable prior is important for good empirical performance.

## 5 CONCLUSION

We propose a model-free imitation learning algorithm for multi-agent settings that leverages recent advances in adversarial training; the algorithm is agnostic to the competitive/noncompetitive structure of the environment, but allows for incorporating a prior when it is available. Experimental results demonstrate that it is able to imitate complex behaviors in high-dimensional environments with both cooperative and adversarial interactions.

