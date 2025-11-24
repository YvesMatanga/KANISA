%% New Solver Test
clc
clear all
close all

%% Reuirements
fprintf('------------------------------------------------------\n')
fprintf('Dependency: INTLAB library (typical version  V14)...\n')
fprintf('------------------------------------------------------\n')
fprintf('Installation:\n')
fprintf('[1] Acquire the INTLAB Library from https://www.tuhh.de/ti3/intlab/\n')
fprintf('[2] Set Path to the library folder in your MATALB\n')
fprintf('[3] run the command startintlab on MATLAB to enable the interval analysis library\n')
fprintf('[4] Proceed to running the code...\n')
fprintf('------------------------------------------------------\n')

%%
test_function_list
format long
%% Test Functions
ls = [20 2 3 5 10 11 12 6 7 9 23 30 31 33 34 35 36];

fn = 20;% Select your test function

%display function details
bounds = function_list{fn}.bounds;
function_strut = function_list{fn}

%% Function Configuration
xl = bounds(:,1); 
xu = bounds(:,2);
obj_func = function_strut.obj_func;
obj_value = [];%function_strut.obj_value;
obj_point = function_strut.obj_point;
fprintf('-----------------------------------------\n')
%% options
options.eps = 10^(-3);
options.intprune = 1;%enable interval analysis pruning
options.mntprune = 1;%enable monotonoy pruning
options.ncvxprune = 1;%enable nonconvexity
options.gVal = obj_value;%set global optimum value
options.ubSolverOptions = optimoptions(@fmincon,'Algorithm','sqp','Display','off');%specify upper bound solver
options.lbSolverOptions = optimoptions(@fmincon,'Algorithm','sqp','Display','off');%specify lower bound solver
options.maxIter = Inf;%set maximum branching
options.search = 'bfs';%set best first search strategy
options.branching = 'nearSq';
options.displayOn = 1;%enable verbosity
options.alpha_filtering = 0;%disable alpha filtering
%% RUN the Solver

[xopt,fopt,perf_options] = ABB_SOLVE(obj_func,xl,xu,options)


%% Display Information
xl=xl;
xu=xu;

if length(xl) == 2 %display if the function is in 2D
    figure
    %xl = bounds(:,1);
    %xu = bounds(:,2);
    d = (max(xu)-min(xl))/100;
    [x1,x2] = meshgrid(xl(1):d:xu(1),xl(2):d:xu(2));
    y = obj_func2d(obj_func,x1,x2);
    mesh(x1,x2,y)%,'FaceAlpha',0.5)
    hold on 
    grid off
    plot3(xopt(1),xopt(2),fopt,'b*','MarkerSize',36)
end