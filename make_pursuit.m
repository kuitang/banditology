function [ bandit_alg ] = make_pursuit( beta )
%MAKE_PURSUIT Selection probabilities follow greedy choice.

global banditmeans

    function [ actions, rewards, values ] = ba(n, A, T, actions, rewards, values)
        pdf = ones(1, A) / A;
        counts = zeros(1, A);
        for t = 1:T
            cdf = cumsum(pdf);
            a = find(cdf > rand, 1);
            counts(a) = counts(a) + 1;
            
            r = randn + banditmeans(n,a);
            
            values(a) = values(a) + (r - values(a)) / counts(a);            
            actions(t) = a;
            rewards(t) = r;
            
            % Update the pdf for the next iteration
            [~, amax] = max(values);            
            pdf(amax) = pdf(amax) + beta*(1 - pdf(amax));
            pdf(pdf ~= amax) = pdf(pdf ~= amax) + beta.*(-pdf(pdf ~= amax));
            % The update rules don't guarantee normalization
            pdf = pdf ./ sum(pdf);
        end
    end

bandit_alg = @ba;
        
end
