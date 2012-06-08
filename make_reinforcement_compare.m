function [ bandit_alg ] = make_reinforcement_compare( alpha, beta, ref )
%MAKE_REINFORCEMENT_COMPARE Compare action reward vs. reference reward.

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
            
            r = randn + banditmeans(n,a);
            values(a) = values(a) + beta*(r - ref_reward);
            ref_reward = ref_reward + alpha*(r - ref_reward);
            
            actions(t) = a;
            rewards(t) = r;
        end
    end

bandit_alg = @ba;    

end
