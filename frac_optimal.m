function [ fopt ] = frac_optimal( actions )
%FRAC_OPTIMAL Summarizes optimal choices over time.
%   fopt = frac_optimal(choices) takes a N-by-T matrix of choices and
%   returns a 1-by-T vector where each entry contains the fraction of the N
%   bandits on which we made the optimal choice.

global banditmeans banditoptima

[N T] = size(actions);

fopt = zeros(1, T);
for t = 1:T
    fopt(t) = sum(actions(:,t) == banditoptima) / N;
end

end