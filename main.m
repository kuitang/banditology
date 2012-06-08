% Tries various bandit algorithms, and plots %-optimal action results.

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

%% Graphs!!!
x = 1:T;
figure
plot(x,fo_eg_01,x,fo_eg_001,x,fo_smax_01);
legend('\epsilon = 0.1', '\epsilon = 0.01', 'softmax \tau = 0.1');
title('Percent optimal action');

figure
plot(x,mr_eg_01,x,mr_eg_001,x,mr_smax_01);
legend('\epsilon = 0.1', '\epsilon = 0.01', 'softmax \tau = 0.1');
title('Mean reward');
