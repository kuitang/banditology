function [ ] = plot_many( labels, vecs )
%PLOT_MANY Plot several labelled vectors on one figure.
%   plot_many(label1, v1, label2, v2, ...). All vs must be of the same
%   length.

if length(labels) ~= length(vecs)
    error('Labels and vecs must have same length.');
end

N = length(labels);
plot_vecs = cell(1, 2*N);

L = length(vecs{1});
x = 1:L;

for n = 2:2:2*N
    plot_vecs{n-1} = x;
    plot_vecs{n} = vecs{n/2};    
end

figure
plot(plot_vecs{:});
legend(labels{:});
