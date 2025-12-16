%% New Solver Test
clc
clear all
close all
% in case of cleanup error
% addpath(fullfile(matlabroot,'toolbox','matlab','general'))  
%% Requirements
fprintf('------------------------------------------------------\n')
fprintf('Dependency: [1] MATLAB (2019b)                        \n')
fprintf('            [2] INTLAB library (typical version  V14) \n')
fprintf('------------------------------------------------------\n')
fprintf('Installation:\n')
fprintf('[1] Acquire the INTLAB Library from https://www.tuhh.de/ti3/intlab/\n')
fprintf('[2] Set Path to the library folder in your MATALB\n')
fprintf('[3] run the command startintlab on MATLAB to enable the interval analysis library\n')
fprintf('[4] Proceed to running the code...\n')
fprintf('------------------------------------------------------\n')

%% Test Function
obj_func = @(x)schaffer2_func(x);
xl = [-65;-65];
xu = [65;65];
%% PSO parameters
pso_options.maxIter = 40;%number of iterations
pso_options.popSize = 30;%population size
pso_options.wmax = 0.9;
pso_options.wmin = 0.4;
pso_options.cr = 2;
pso_options.sr = 2;
pso_options.maxStallIter = 20;
pso_options.geps = 10^(-3);%improvement value significance
pso_options.EnStopHeurestic = 1;%Enable stalling stop heuristic
pso_options.EnPostOptimisation = 1;%enable solution fine-tuning by SQP
pso_options.Nstep = 2;%|v_max| = (Iu-Il)/Nstep
pso_options.pop = [];%no initial population pre-load
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
%% RUN the Solver

[xopt_pso_abb,fopt_pso_abb,algCost_pso_abb] = ABB_PSO_SOLVE(obj_func,xl,xu,options)

%% Test function
function [y] = schaffer2_func(x)
x1 = x(1);
x2 = x(2);
fact1 = (sin(x1^2-x2^2))^2 - 0.5;
fact2 = (1 + 0.001*(x1^2+x2^2))^2;
y = 0.5 + fact1/fact2;
end