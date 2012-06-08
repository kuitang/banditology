function [ bandit_alg ] = make_reinforcement_compare_adjusted_prob( alpha, beta, ref )
%MAKE_REINFORCEMENT_COMPARE_ADJUSTED_PROB Add (1 - p(a)) to value updates.

global banditmeans

if nargin < 3
    ref = 0;
end

    function [ actions, rewards, values ] = ba(n, ~, T, actions, rewards, values)
        ref_reward = ref;
        for t = 1:T
            exps = exp(values);
            % Unnormalized inverse cdf sampling
            cdf  = cumsum(exps);
            a = find(cdf > cdf(end)*rand, 1);
            pa = exps(a) / cdf(end);
            
            r = randn + banditmeans(n,a);
            values(a) = values(a) + beta*(r - ref_reward + (1 - pa));
            ref_reward = ref_reward + alpha*(r - ref_reward);
            
            actions(t) = a;
            rewards(t) = r;
        end
    end

bandit_alg = @ba;    

end
