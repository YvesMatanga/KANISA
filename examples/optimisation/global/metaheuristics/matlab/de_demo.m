clc
clear all
close all

%% Test Function 
obj_func = @(x)schaffer2_func(x);
xl = [-65;-65];
xu = [65;65];
%% Solver settings
de_options.maxIter =100;%maximum generation count
de_options.popSize = 30;%population size
de_options.EnStopHeurestic = 0;%enable heuristic stop
de_options.maxStallIter = Inf;%maximum stalling iteration
de_options.geps = 10^(-3);%tolerance improvement
de_options.F = 0.9;
de_options.cr = 0.4;%set cross over rate
de_options.cross_over_option = 'default';
de_options.EnPostOptimisation = 0;%enable post optimisation
de_options.post_opts =  optimoptions(@fmincon,'Algorithm','sqp','Display','off');
de_options.displayOn = 1;%enable iteration result display
%% Solver
[x_opt,f_opt,alg_data] = DE_SOLVE(obj_func,xl,xu,de_options)
%% Test function
function [y] = schaffer2_func(x)
x1 = x(1);
x2 = x(2);
fact1 = (sin(x1^2-x2^2))^2 - 0.5;
fact2 = (1 + 0.001*(x1^2+x2^2))^2;
y = 0.5 + fact1/fact2;
end