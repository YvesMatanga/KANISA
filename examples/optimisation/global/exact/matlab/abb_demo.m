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
%% options
options.eps = 10^(-3);
options.intprune = 1;%enable interval analysis pruning
options.mntprune = 1;%enable monotonoy pruning
options.ncvxprune = 1;%enable nonconvexity
options.gVal = [];%set global optimum value
options.ubSolverOptions = optimoptions(@fmincon,'Algorithm','sqp','Display','off');%specify upper bound solver
options.lbSolverOptions = optimoptions(@fmincon,'Algorithm','sqp','Display','off');%specify lower bound solver
options.maxIter = Inf;%set maximum branching
options.search = 'bfs';%set best first search strategy
options.branching = 'nearSq';%branch the hybercube on the longest segment
options.displayOn = 1;%enable verbosity
options.alpha_filtering = 0;%disable alpha filtering
%% RUN the Solver

[xopt,fopt,perf_options] = ABB_SOLVE(obj_func,xl,xu,options)

%% Test function
function [y] = schaffer2_func(x)
x1 = x(1);
x2 = x(2);

fact1 = (sin(x1^2-x2^2))^2 - 0.5;
fact2 = (1 + 0.001*(x1^2+x2^2))^2;

y = 0.5 + fact1/fact2;
end
