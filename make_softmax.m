function [ bandit_alg ] = make_softmax( temperature )
%MAKE_SOFTMAX Make algorithm to sample action from Gibbs distribution.

global banditmeans

    function [ actions, rewards, values ] = ba(n, A, T, actions, rewards, values)
        counts = zeros(1, A);
        for t = 1:T
            exps = exp(values ./ temperature);
            % Unnormalized inverse cdf sampling
            cdf  = cumsum(exps);
            a = find(cdf > cdf(end)*rand, 1);
            counts(a) = counts(a) + 1;
            
            r = randn + banditmeans(n,a);
            values(a) = values(a) + (r - values(a)) / counts(a);
            actions(t) = a;
            rewards(t) = r;
        end
    end

bandit_alg = @ba;

end

