function [ bandits ] = make_gaussian_bandits( N, A, meanmean, meansd, sd )
%MAKE_BANDITS Construct N Gaussian bandits whose means are sampled from
%   a (meanmena, meansd) Gaussian distribution and standard deviations are
%   sd. Returns a cell array of bandit functions, which take an action
%   from 1..A and return a reward.

banditmeans = normrnd(meanmean, meansd, N, A);
bandits = cell(1, N);
for n = 1:N
    bandits{n} = @(a) double(normrnd(banditmeans(n, a), sd));
end

end

