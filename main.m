%% Banditology
% We explore various n-armed bandit algorithms presented in chapter 2 of
% Sutton and Barto's book _Reinforcement Learning: An Introduction._
% The n-armed bandit is a basic problem in reinforcement learning which
% posits the _exploration-exploitation_ tradeoff. Here, we have a slot
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
banditmeans = randn(N, A);
[~, banditoptima] = max(banditmeans, [], 2);


%% Try each algorithm in turn.
eg_01 = make_epsilon_greedy(0.1);
[fo_eg_01, mr_eg_01] = eval_bandit(eg_01, T);
eg_001 = make_epsilon_greedy(0.01);
[fo_eg_001, mr_eg_001] = eval_bandit(eg_001, T);
smax_01 = make_softmax(0.1);
[fo_smax_01, mr_smax_01] = eval_bandit(smax_01, T);
rc_01_01 = make_reinforcement_compare(0.1, 0.1);
[fo_rc_01_01, mr_rc_01_01] = eval_bandit(rc_01_01, T);

%% Graph general comparison of algorithms.
x = 1:T;
figure
plot(x,fo_eg_01,x,fo_eg_001,x,fo_smax_01,x,fo_rc_01_01);
legend('\epsilon = 0.1', '\epsilon = 0.01', 'softmax \tau = 0.1', 'comparison \alpha = 0,1, \beta = 0.1');
title('Percent optimal action');

figure
plot(x,mr_eg_01,x,mr_eg_001,x,mr_smax_01,x,mr_rc_01_01);
legend('\epsilon = 0.1', '\epsilon = 0.01', 'softmax \tau = 0.1', 'comparison \alpha = 0,1, \beta = 0.1');
title('Mean reward');

%% Exercise 2.11: compare the adjusted probabilities vs. not
rc_01_01_minus5   = make_reinforcement_compare(0.1, 0.1, -10);
rcap_01_01_minus5 = make_reinforcement_compare_adjusted_prob(0.1, 0.1, -10);
[f1, m1] = eval_bandit(rc_01_01_minus5, T);
[f2, m2] = eval_bandit(rcap_01_01_minus5, T);

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
rcap_01_01_plus5 = make_reinforcement_compare_adjusted_prob(0.1, 0.1, 5);
[f1, m1] = eval_bandit(rc_01_01_plus5, T);
[f2, m2] = eval_bandit(rcap_01_01_plus5, T);

figure
plot(x,f1,x,f2);
legend('unadjusted', 'adjusted');
title('Percent optimal action --- reference reward = 5');

figure
plot(x,m1,x,m2);
legend('unadjusted', 'adjusted');
title('Mean reward --- reference reward = 5');