function [ bandit_alg ] = make_softmax( temperature )
%MAKE_SOFTMAX Make algorithm to sample action from Gibbs distribution.

global banditmeans

    function [ actions, rewards, values ] = ba(n, ~, T, actions, rewards, values)
        for t = 1:T
            exps = exp(values ./ temperature);
            % Unnormalized inverse cdf sampling
            cdf  = cumsum(exps);
            a = find(cdf > cdf(end)*rand, 1);
            
            r = randn + banditmeans(n,a);
            values(a) = values(a) + (r - values(a)) / t;
            actions(t) = a;
            rewards(t) = r;
        end
    end

bandit_alg = @ba;

end

