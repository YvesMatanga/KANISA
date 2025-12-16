clc
clear all
close all

%% Test Function 
obj_func = @(x)schaffer2_func(x);
xl = [-65;-65];
xu = [65;65];
%%
pso_options.maxIter = 100;%set maximum search iteration
pso_options.displayOn = 1;%display iteration results
pso_options.popSize = 30;
pso_options.wmax = 0.9;
pso_options.wmin = 0.4;
pso_options.cr = 2;
pso_options.sr = 2;
pso_options.maxStallIter = Inf;%set maximum stalling iteration
pso_options.geps = 10^(-3);%improvement tolerance
pso_options.EnStopHeurestic = 0;%enable heuristic stopping criteria
pso_options.EnPostOptimisation = 0;%enable post optimisation
pso_options.Nstep = 2;%|v_max| = (Iu-Il)/Nstep
pso_options.pop = halton_sequence_gen(xl,xu,pso_options.popSize);%set to empty [] for random init
pso_options.post_opts =  optimoptions(@fmincon,'Algorithm','sqp','Display','off');
%% Solver
[x_opt,f_opt,alg_data] = PSO_SOLVE(obj_func,xl,xu,pso_options)
%% Test function
function [y] = schaffer2_func(x)
x1 = x(1);
x2 = x(2);
fact1 = (sin(x1^2-x2^2))^2 - 0.5;
fact2 = (1 + 0.001*(x1^2+x2^2))^2;
y = 0.5 + fact1/fact2;
end