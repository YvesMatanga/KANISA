function [xopt,fopt,options] = pso_solver(f,xl,xu,options)
%% This is an implemntation of the PSO solver
%options flag = 0 -- maximum iteration reached
%options flag = 1 -- stopped by heurestic
fprintf('PSO start...\n')
tStart = clock();
if nargin < 4
    maxIter = 40;
    N = 30;
    wmax = 0.9;%initertial weights
    wmin = 0.4;
    cr = 2;
    sr = 2;
    maxStallIter = 10;
    geps = 10^(-3);
    EnStopHeurestic  = 0;
    EnPostOptimisation = 0;
    pop = [];
    post_opts = [];
else
    maxIter = options.maxIter;
    N = options.popSize;
    wmax = options.wmax;
    wmin = options.wmin;
    cr = options.cr;
    sr = options.sr;
    maxStallIter = options.maxStallIter;
    geps = options.geps;
    EnStopHeurestic = options.EnStopHeurestic;
    EnPostOptimisation = options.EnPostOptimisation;
    pop = options.pop;
    post_opts = options.post_opts;
end
%
Nstep = 2;
nx = length(xl);
Vmax = (xu-xl)/Nstep;
BUB = +Inf;%set best upper bound
options.optCurve = zeros(maxIter+1,nx+1);

%% Initialise the swarm
Fe = [];
Xe = [];
Np = 0;

Xks = zeros(nx,N+Np);
Vks = zeros(nx,N+Np);
Pks = zeros(nx,N+Np);
gk = zeros(nx,1);
fpk = zeros(N+Np,1);

gFuncPrev = 0;
maxStalli = 0;
gId = -1;
Vg = [];

for i=1:N   
   if isempty(pop)
        Xks(:,i) = (xu-xl).*rand(nx,1)+xl; %value bounded in xl < x < xu 
    else
        Xks(:,i) = pop(:,i).*(xu-xl)+xl;
   end 
    
    %Xks(:,i) = (xu-xl).*rand(nx,1)+xl; %value bounded in xl < x < xu  
%     end    
    Vks(:,i) = 2*Vmax.*rand(nx,1)-Vmax;%speed bound in -vmax < v < vmax
    Pks(:,i) = Xks(:,i);%initialise best pk per particle
    fpk(i) = f(Xks(:,i));
    %computer best gk
    if  fpk(i) < BUB
        gk = Xks(:,i);
        BUB = fpk(i);%f(Xks(:,i));
        gId = i;
        Vg = Vks(:,i);
    end
end

% Associate elites
for i=N+1:(Np+N)
    Xks(:,i) = Xe(i-N,:)';
    Pks(:,i) = Xks(:,i);
    Vks(:,i) = 2*Vmax.*rand(nx,1)-Vmax;%speed bound in -vmax < v < vmax
    fpk(i) = Fe(i-N);
    if fpk(i) < BUB
        gk = Xks(:,i);
        BUB = fpk(i);%f(Xks(:,i));
        gId = i;
        Vg = Vks(:,i);
    end
end

options.optCurve(1,1) = BUB;
options.optCurve(1,2:end) = gk';
%% PSO iterations
iter = 0;

while iter < maxIter
   %set adaptive inertial weight
   w = wmax - (wmax-wmin)*iter/maxIter;
   Tgk = gk;
   for i=1:(N+Np)       
       %set Vk+1
       Vkps = w*Vks(:,i)+cr*rand(nx,1).*(Pks(:,i)-Xks(:,i))+...
             sr*rand(nx,1).*(Tgk-Xks(:,i));
       %set maximum velocity
       Vkps = min(Vkps,Vmax);
       Vkps = max(Vkps,-Vmax); 
       %update new particle
       Xkps = Xks(:,i)+ Vkps;   
       %set limit of new particle
       Xkps = min(Xkps,xu);
       Xkps = max(Xkps,xl);     
       %get fkp
       fxkp = f(Xkps);
       %fpk(i) = f(Pks(:,i));
       %computer best pk
       if fxkp < fpk(i)
          Pks(:,i) = Xkps;
          fpk(i) = fxkp;
       end
       %compute best gk
       %fpk(i) = f(Pks(:,i));
       if fpk(i) < BUB
          BUB = fpk(i);
          gk = Pks(:,i);         
       end      
       %Update Xk and Vk
       Xks(:,i) = Xkps;
       Vks(:,i) = Vkps;
   end
   
   % heurestic stop
   if EnStopHeurestic == 1
       if abs(BUB-gFuncPrev) < geps
         maxStalli = maxStalli+1;
       else
         maxStalli = max(maxStalli-1,0);
       end
         gFuncPrev = BUB; 
       if maxStalli > maxStallIter
           %disp('Maximum stall reached')
           options.flag = 1;
         break;
       else
           options.flag = 0;
       end    
   else
       options.flag = 0;
   end
   %Move to next ieration
   iter = iter+1; 
   options.optCurve(iter+1,1) = BUB;
   options.optCurve(iter+1,2:end) = gk';
end
%set output

if EnPostOptimisation == 1
    %opts = optimoptions(@fmincon,'Algorithm','sqp','Display','off');
%     opts = [];
%     [gk,BUB] = patternsearch(f,gk,[],[],[],[],xl,xu,[],opts);
    
    %opts = [];
    [gk,BUB] = fmincon(f,gk,[],[],[],[],xl,xu,[],post_opts);
end
options.elites =  [fpk,Pks(:,1:N+Np)'];
xopt = gk;
fopt = BUB;
options.iter = iter;
tStop = clock();
options.teps = etime(tStop,tStart);
fprintf('PSO stop...\n')
end

function [fsort,xsort] = sort_points(fin,xin,N)
   sorted_flag = 1;
   while sorted_flag == 1
       sorted_flag = 0;
       for j=1:N-1
           if fin(j) > fin(j+1)
               tmpf = fin(j);
               tmpx = xin(j,:);
               
               fin(j) = fin(j+1);
               xin(j,:) = xin(j+1,:);
               fin(j+1) = tmpf;
               xin(j+1,:) = tmpx;
               sorted_flag = 1;
           end
       end
   end
   fsort = fin;
   xsort = xin;
end