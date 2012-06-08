function [ ] = compare_bandits( T, title_suffix, varargin )
%COMPARE_BANDITS Run several bandits and plot their performance 
%   compare_bandits(T, label1, bandit1, label2, bandit2, ...) runs the
%   bandits for T timesteps.

if mod(length(varargin), 2) ~= 0
    error('Must have even input arguments')
end

N = length(varargin) / 2;
fracs_optimal = cell(1, N);
mean_rewards  = cell(1, N);
labels        = cell(1, N);

for n = 1:N
    labels{n} = varargin{2*n - 1};
    disp(['Running ' labels{n}])
    tic;
    [fracs_optimal{n}, mean_rewards{n}] = eval_bandit(varargin{2*n}, T);
    disp(['Took ' num2str(toc) ' seconds.'])
end

if ~isempty(title_suffix)
    title_suffix = ['---' title_suffix];
end

plot_many(labels, fracs_optimal);
title(['Fraction optimal' title_suffix]);

plot_many(labels, mean_rewards);
title(['Mean rewards' title_suffix]);

end

