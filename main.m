%% Banditology
% We explore various n-armed bandit algorithms presented in chapter 2 of
% Sutton and Barto's book _Reinforcement Learning: An Introduction._
%
% The n-armed bandit is a basic problem in reinforcement learning which
% posits the _exploration-exploitation_ tradeoff. We have a slot
% machine with n levers. Pulling a lever generates a stochastic quantity
% of reward. Suppose that some levers generate, on average, more reward
% than others. The goal is to generate the greatest cumulative reward.
%
% In these examples, the player has no prior knowledge of the levers. As
% the player makes moves, the player will remember how much reward each
% lever generates. To succeed, the player must balance
% _exploitation_---using her current knowledge of the levers to make the
% best short-term choice, with _exploration_---deliberately picking a lever
% other than the local maximum to check that she is not missing out on
% potentially greater reward.
%
% This project compares various algorithms to solve the
% exploration/exploitation tradeoff.
%

%% Make bandits.
clear
% Make 2000 bandits with variance 1 and whose means are sampled from a
% standard normal distribution, as in the book.
N = 2000;
T = 1000;
A = 10;

global banditmeans banditoptima
% We have N bandits, each with A actions. Rows are bandits; columns are
% actions.
banditmeans = randn(N, A);
% To evaluate performance, we compare the choices our algorithm made to the
% optimal choice. Given a bandit (row), we define the optimal choice to be
% the action (column index) with the highest mean reward.
[~, banditoptima] = max(banditmeans, [], 2);


%% Try each algorithm in turn.

% The $\epsilon$-greedy algorithm is the simplest method to balance
% exploration and exploitation. It selects the greedy choice (choice with
% highest _value_) with probability $1 - \epsilon$ and it selects a
% random uniform choice with probability $\epsilon$. The _value_ of an
% action $a$ is the sample mean of rewards given by $a$. By the law of
% large numbers, the value converges to the mean reward of $a$.
%
% Other than asymptotic convergence of action values, $\epsilon$-greedy has
% few theoretical guarantees. Nevertheless, it is insanely simple and easy
% to tune, and yields great results in the Gaussian case.
eg_01 = make_epsilon_greedy(0.1);

% A lower $\epsilon$ favors more exploitation to exploration. On the
% graphs, notice that while the percent optimal curve for $\epsilon = 0.01$
% is significantly lower than that of for $\epsilon = 0.1$, the reward
% curves are much closer. Thus, while it takes longer for the $\epsilon =
% 0.01$ player to find the global optimum, the $\epsilon = 0.01$ makes
% better use of the local optima that it finds.
eg_001 = make_epsilon_greedy(0.01);

% A drawback of the $\epsilon$-greedy algorithms is that every bad move is
% drawn with probability $\epsilon / A$. This may not be desirable if bad
% moves are very bad. The softmax algorithm maintains sample mean values
% like the $\epsilon$-greedy algorithm, but it always chooses an action
% from the _Gibbs distribution_, where the probability of choosing $a$ is
%
% $$\dfrac{\exp(Q(a) / \tau)}{\sum_{b=1}^A \exp(Q(b) / \tau)}$$
% 
% Thus, the probability of an action scales exponentially with its value.
% The parameter $\tau$ represents temperature. As $\tau \rightarrow
% \infty$, softmax approaches uniform selection, while as $\tau \rightarrow
% 0$, softmax approaches greedy selection.
%
% In practice, $\tau$ is much less intuitive than $\epsilon$ and is
% consequently hard to tune. The author had to try several times to get a
% reasonable result.
smax = make_softmax(0.3);

% Reinforcement comparison...
rc_01_01 = make_reinforcement_compare(0.1, 0.1);

% Pursuit...
purs = make_pursuit(0.01);

%% General comparison of algorithms.
compare_bandits(T, [], '\epsilon = 0.1', eg_01, ...
                       '\epsilon = 0.01', eg_001, ...
                       'softmax \tau = 0.3', smax, ...
                       'comparison \alpha = 0.1, \beta = 0.1', rc_01_01, ...
                       'pursuit \beta = 0.01', purs);

%% Compare parameter settings for reinforcement comparison
% In reinforcement pursuit, $\alpha$ controls the update rate of the action
% values while $\beta$ controls the update rate of the reference actions.
% From the previous graphs, we get good results by setting them equal. What
% if we set them asymmetrically?
rc_001_001 = make_reinforcement_compare(0.01, 0.01);
rc_01_001 = make_reinforcement_compare(0.1, 0.01);
rc_001_01 = make_reinforcement_compare(0.01, 0.1);

compare_bandits(4000, 'Reinforcement Comparison Parameters', ...
                          '\alpha = 0.01, \beta = 0.01', rc_001_001, ...
                          '\alpha = 0.1, \beta = 0.01', rc_01_001, ...
                          '\alpha = 0.001, \beta = 0.1', rc_001_01);

% Reducing $\alpha$ slightly hurt our results, as we are not able to update
% our knowledge of action values as quickly. But the loss is not drastic.
%
% However, reducing $\beta$% significantly decreases convergence speed,
% regardless of the value of $\alpha$. In this case, reference rewards are
% initialized to zero (a "realistic" as opposed to an optimistic setting).
% Low $\beta$ means a slowly-increasing reference reward. Thus, suboptimal
% actions will continue to exceed the reference reward for a long time and
% will continue to be selected. A low $\beta$ encourages exploration.
%
% Here, we set T = 4000 to fully observe the effect of additional
% exploration. Eventually, the curves for $\beta = 0.01$ exceed that of
% $\beta = 0.1$, showing that the additional exploration does eventually
% pay off.

%% Exercise 2.11: compare the adjusted probabilities vs. not
T = 1000;
rc_01_01_minus5   = make_reinforcement_compare(0.1, 0.1, -10);
rcapurs_01_minus5 = make_reinforcement_compare_adjusted_prob(0.1, 0.1, -10);
[f1, m1] = eval_bandit(rc_01_01_minus5, T);
[f2, m2] = eval_bandit(rcapurs_01_minus5, T);

x = 1:T
figure
plot(x,f1,x,f2);
legend('unadjusted', 'adjusted');
title('Percent optimal action --- reference reward = -10');

figure
plot(x,m1,x,m2);
legend('unadjusted', 'adjusted');
title('Mean reward --- reference reward = -10');

%% Optimistic version of above
rc_01_01_plus5   = make_reinforcement_compare(0.1, 0.1, 5);
rcapurs_01_plus5 = make_reinforcement_compare_adjusted_prob(0.1, 0.1, 5);
[f1, m1] = eval_bandit(rc_01_01_plus5, T);
[f2, m2] = eval_bandit(rcapurs_01_plus5, T);

figure
plot(x,f1,x,f2);
legend('unadjusted', 'adjusted');
title('Percent optimal action --- reference reward = 5');

figure
plot(x,m1,x,m2);
legend('unadjusted', 'adjusted');
title('Mean reward --- reference reward = 5');