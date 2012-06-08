function [ fopt, mean_reward ] = eval_bandit( bandit_alg, T )
%EVAL_BANDIT Run bandit_alg on many bandits and return performance.
%   [ fopt, mean_reward ] = eval_bandit(T, bandit_alg) runs bandit_alg T
%   times on the global banditmeans and returns 1-by-T vectors for the
%   fraction of bandits for which optimal move was selected, and the
%   average rewards from all bandits at each time.

global banditmeans;
[N A] = size(banditmeans);

actions = zeros(N, T);
rewards = zeros(N, T);
for n = 1:N
    [ actions(n,:), rewards(n,:) ] = run_bandit(bandit_alg, n, T);
end

fopt = frac_optimal(actions);
mean_reward = mean(rewards, 1);
