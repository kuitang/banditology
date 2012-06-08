function [ choices ] = epsilon_greedy( n, A, epsilon, T, values )
%EPSILON_GREEDY Perform greedy action with probability 1 - eps.
%   choices = epsilon_greedy(bandit, epsilon, T, values=0) performs T bandit
%   plays on bandit n by selecting the action with highest sample average
%   value with probability 1 - epsilon and initial values supplied by
%   values. Returns a 1-by-N vector of the choices at each play.

global banditmeans

if nargin < 5
    values = zeros(1, A); % default
end

choices = zeros(1, T);
for t = 1:T
    if rand < epsilon
        % Pick an action uniformly randomly
        a = randi(A);        
    else
        % Pick the greedy choice
        [~, a] = max(values);        
    end
    
    r = randn + banditmeans(n,a);
    
    %r = bandit(a);
    values(a) = values(a) + 1/t * (r - values(a)); % on-line mean update
    choices(t) = a;
end

end
