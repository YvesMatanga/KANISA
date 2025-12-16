
%% Test Function 
obj_func = @(x)camel6_func(x);
xl = [-4;-4];
xu = [4;4];
%%
sn_pso_options.maxIter = 200;
sn_pso_options.popSize = 30;
sn_pso_options.wmax = 0.9;
sn_pso_options.wmin = 0.4;
sn_pso_options.w = 0.729;
sn_pso_options.cr = 2;
sn_pso_options.sr = 2;
sn_pso_options.reMaxStallIter = 10;%after x stalling iterations restart the swarm in a new niche
sn_pso_options.EnStopHeurestic = 0;%disable 
sn_pso_options.geps = 10^(-3);%global solution improvement tolerance
sn_pso_options.maxStallIter = Inf;%disable global solution maximum stalling iteration
sn_pso_options.maxSwarmHeads = 6;%set maximum swarm heads
sn_pso_options.EnPostOptimisation = 1;%enable post optimisation
sn_pso_options.Nstep = 2;%|v_max| = (Iu-Il)/Nstep
sn_pso_options.pop = halton_sequence_gen(xl,xu,sn_pso_options.popSize);%uniform distribution,set [] to use random init
sn_pso_options.post_opts =  optimoptions(@fmincon,'Algorithm','sqp','Display','off');
sn_pso_options.displayOn = 1;%display verbosity
%% Solver

[xopt_snpso_c,fopt_snpso_c] = SN_PSO_SOLVE(obj_func,xl,xu,sn_pso_options)

%% Test function
function y = camel6_func(x)
 y = (4-2.1*x(1)^2+(x(1)^4)/3)*x(1)^2+x(1)*x(2)+...
     (4*x(2)^2-4)*x(2)^2;
end