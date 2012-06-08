% Tries various bandit algorithms, and plots %-optimal action results.

%% Make bandits.
% Make 2000 bandits with variance 1 and whose means are sampled from a
% standard normal distribution, as in the book.
N = 2000;
T = 1000;
A = 10;

global banditmeans
banditmeans = normrnd(0, 1, N, A);


%% Try each algorithm in turn.
choices = zeros(N, T);
for n = 1:N
    %bandit = bandits{n};    
    choices(n,:) = epsilon_greedy(n, A, 0.1, T);
end

[~, optima] = max(banditmeans, [], 2);
pct_optimal = zeros(1, T);
for t = 1:T
    pct_optimal(t) = sum(choices(:,t) == optima) / N;
end

plot(pct_optimal);