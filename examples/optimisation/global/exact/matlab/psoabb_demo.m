nx = length(xl);
%% PSO parameters
pso_options.maxIter = 40;
pso_options.popSize = 30;
pso_options.wmax = 0.9;
pso_options.wmin = 0.4;
pso_options.cr = 2;
pso_options.sr = 2;
pso_options.maxStallIter = 20;
pso_options.maxScoutIter = 20;
pso_options.geps = 10^(-3);
pso_options.EnStopHeurestic = 1;
pso_options.EnPostOptimisation = 1;
pso_options.EnElitism = 0;
pso_options.eliteQ = 0.3;
pso_options.x0 = [];
pso_options.Nstep = 2;
pso_options.pop = [];
pso_options.grad_en = 0;
pso_options.post_opts =  optimoptions(@fmincon,'Algorithm','sqp','Display','off');
%% options
options.eps = 10^(-3);
options.intprune = 1;%enable interval analysis pruning
options.mntprune = 1;%enable monotonoy pruning
options.ncvxprune = 1;%enable nonconvexity prune
options.gVal = [];%obj_value;%set global optimum value
options.ubSolverOptions = optimoptions(@fmincon,'Algorithm','sqp','Display','off');
options.lbSolverOptions = optimoptions(@fmincon,'Algorithm','sqp','Display','off');
options.maxIter = Inf;%set maximum branching
options.search = 'bfs';%set best first search strategy
options.branching = 'nearSq';
options.displayOn = 1;
options.alpha_filtering = 0;%disable alpha filtering
options.ub_options = pso_options;
%% Test

ticIn = clock;
[xopt_pso,fopt_pso,algCost_pso] = pso_solver(obj_func,xl,xu,pso_options)
ticOut = clock;
teps_psoabb = etime(ticOut,ticIn)

% ticIn = clock;
% [xopt_pso_abb,fopt_pso_abb,algCost_pso_abb] = ABB_PSO_SOLVE(obj_func,xl,xu,options)
% ticOut = clock;

ticIn = clock;
[xopt_abb,fopt_abb,algCost_abb] = ABB_SOLVE(obj_func,xl,xu,options)
ticOut = clock;

clc
algCost_pso
algCost_pso_abb
algCost_abb

%%
function f = rosenbrock2D(x)
% ROSENBROCK2D  Two-dimensional Rosenbrock function
%
%   f = rosenbrock2D(x1, x2)

    f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
end
