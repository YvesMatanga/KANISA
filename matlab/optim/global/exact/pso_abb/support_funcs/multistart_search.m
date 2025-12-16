function [x_opt,f_opt] = multistart_search(f,xl,xu,options)
%UNTITLED Summary of this function goes here
if nargin < 4
    popSize = 30;
    post_opts = optimoptions(@fmincon,'Algorithm','sqp','Display','off');
else
    popSize = options.popSize;
    post_opts = options.post_opts;
end

%generate population
nx = length(xl);
Xks = rand(nx,1).*(xu-xl) + xl;%generate random points
BUB=+inf;
gk = [];
for i=1:popSize
    if f(Xks) < BUB
        gk = Xks;
        BUB = f(Xks);
    end
end
%--------------------
[x_opt,f_opt] = fmincon(f,gk,[],[],[],[],xl,xu,[],post_opts);
end