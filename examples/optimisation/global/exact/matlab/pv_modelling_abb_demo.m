clc
clear all
close all
format long
addpath('/Applications/MATLAB_R2019b.app/toolbox/matlab/general')
%% Data Collection Paper
% @article{ebrahimi2019parameters,
%   title={Parameters identification of PV solar cells and modules using flexible particle swarm optimization algorithm},
%   author={Ebrahimi, S Mohammadreza and Salahshour, Esmaeil and Malekzadeh, Milad and Gordillo, Francisco},
%   journal={Energy},
%   volume={179},
%   pages={358--372},
%   year={2019},
%   publisher={Elsevier}
% }
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
%%
xl = [0;0;00;0;1];
xu = [1;1;0.5;100;2];
Vm = [-0.2057;-0.1291;-0.0588;0.0057;0.0646;0.1185;0.1678;0.2132;0.2545;0.2924;0.3269;0.3585;0.3873;0.4137;0.4373;0.4590;0.4784;0.4960;0.5119;0.5265;0.5398;0.5521;0.5633;0.5736;0.5833;0.59];
Im = [0.764;0.762;0.7605;0.7605;0.76;0.759;0.757;0.757;0.7555;0.754;0.7505;0.7465;0.7385;0.728;0.7065;0.6755;0.632;0.573;0.499;0.413;0.3165;0.212;0.1035;-0.01;-0.123;-0.21];
T = 33;
%%
obj_func = @(x)p_obj_func(x,T,Vm,Im);
%% Algorithm Settings
%%
sqp_options.optim_opts = optimoptions(@fmincon,'Algorithm','sqp','Display','off');

xopt_sqp_bag = {};
xopt_pso_c_bag = {};
xopt_ga_c_bag = {};
xopt_de_c_bag = {};
xopt_snpso_c_bag = {};

fopt_sqp_av = [];
fopt_pso_c_av = [];
fopt_ga_c_av = [];
fopt_de_c_av = [];
fopt_snpso_c_av = [];

Nr = 10;
for i=1:Nr
tic
[xopt_sqp,fopt_sqp] = sqp_solver(obj_func,xl,xu,sqp_options)
toc
xopt_sqp_bag{end+1} = {xopt_sqp,fopt_sqp};
fopt_sqp_av(end+1) = fopt_sqp;
%%
end


%%
% clc
% xopt_sqp
% fopt_sqp
% xopt_pso_c
% fopt_pso_c
% xopt_snpso_c
% xopt_snpso_c
clc
fopt_sqp_av_c = mean(fopt_sqp_av)
std_sqp_c = std(fopt_sqp_av)
%%
ticIn = clock;
[xopt_pso_abb,fopt_pso_abb,algCost_pso_abb] = ABB_SOLVE(obj_func,xl,xu,options)
ticOut = clock;
%%
If = obj_func(infsup(xl,xu))
%%
function Ic = PV_model1_func(Iph,Io,Rs,Rsh,n,Il,Vl,T)
  k = 1.381*10^(-23);
  q = 1.6*10^(-19);
  Tk = T+273;
  Ic = Iph - Io*(exp(q*(Vl+Il*Rs)/(n*k*Tk))-1)- (Vl+Il*Rs)/Rsh;
end


function J = p_obj_func(x,T,Vm,Im)
  N = length(Im);
  J = 0;
  for i=1:N
      J = J + (Im(i) - PV_model1_func(x(1),x(2),x(3),x(4),x(5),Im(i),Vm(i),T)).^2;
  end
  %J = sqrt(J);
end

function [xopt,fopt,options_out] = sqp_solver(f,xl,xu,options)
%SQP_SOLVER Summary of this function goes here
options_out = options;
fprintf('SQP Solver start...\n')
tIn = clock();
Nx = length(xl);
x0 = rand(Nx,1).*(xu-xl) + xl;
[xopt,fopt,exitflag,output] = fmincon(f,x0,[],[],[],[],xl,xu,[],options.optim_opts) 
tOut = clock();
options_out.teps = etime(tOut,tIn);
options_out.iter = output.iterations;
fprintf('SQP Solver stop...\n')
end
