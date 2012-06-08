function [ bandit_alg ] = make_epsilon_greedy( epsilon )
%MAKE_EPSILON_GREEDY Make an epsilon_greedy algorithm for bandit n.
%   The bandit_alg will select the action with highest value with
%   probability 1 - epsilon and will select an action uniformly randomly
%   with probability epsilon. After each choice, the value for that choice
%   is updated to the mean of the experienced rewards.

global banditmeans

    function [ actions, rewards, values ] = ba(n, A, T, actions, rewards, values)
        for t = 1:T
            if rand < epsilon
                a = randi(A);
            else
                [~, a] = max(values);
            end
            
            r = randn + banditmeans(n,a);
            values(a) = values(a) + (r - values(a)) / t; % on-line mean update
            actions(t) = a;
            rewards(t) = r;
        end
    end

bandit_alg = @ba;

end
