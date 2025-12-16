
%% Test Function 
obj_func = @(x)camel6_func(x);
xl = [-4;-4];
xu = [4;4];
%%
sn_de_options.maxIter = 200;%maximum search budget
sn_de_options.popSize = 30;
sn_de_options.EnStopHeurestic = 0;%disable global solution stalling check
sn_de_options.geps = 10^(-3);%global solution improvement tolerance
sn_de_options.maxStallIter = Inf;%
sn_de_options.F = 0.9;
sn_de_options.maxClusters = 5;%number of clusters
sn_de_options.reMaxStallIter = 10;%%after x stalling iterations restart the swarm in a new niche
sn_de_options.cr = 0.4;%set cross over rate
sn_de_options.cross_over_option = 'default';
sn_de_options.EnPostOptimisation = 1;%enable post optimisation after niche fine-tuning
sn_de_options.displayOn = 1;%enable verbosity
sn_de_options.post_opts =  optimoptions(@fmincon,'Algorithm','sqp','Display','off');
%% Solver
[xopt_snde_c,fopt_snde_c] = SN_DE_SOLVE(obj_func,xl,xu,sn_de_options)

%% Test function
function y = camel6_func(x)
 y = (4-2.1*x(1)^2+(x(1)^4)/3)*x(1)^2+x(1)*x(2)+...
     (4*x(2)^2-4)*x(2)^2;
end