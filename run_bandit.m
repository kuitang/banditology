function [ actions, rewards ] = run_bandit( bandit_alg, n, T, values )
%RUN_BANDIT Runs banditalg for T timesteps, recording choices and rewards.
%   [ actions, rewards ] = run_bandit(bandit_alg, n, T, values=0) runs
%   bandit algorithm on bandit n (a function handle) for T plays.
%
%   [ actions, rewards ] = bandit_alg(A, T, actions, rewards, values) is a
%   function that runs T bandit plays. bandit_alg should just take care of
%   one timestep but for speed, we implement the entire inner loop in
%   bandit_alg.

global banditmeans
[N A] = size(banditmeans);

if nargin < 4
    values = zeros(1, A);
end

actions = zeros(1, T);
rewards = zeros(1, T);
[ actions, rewards, ~ ] = bandit_alg(n, A, T, actions, rewards, values);
