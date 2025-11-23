function [xopt,fopt,options] = PSO_SOLVE(f,xl,xu,options)
%% This is an implemntation of the PSO solver
%options flag = 0 -- maximum iteration reached
%options flag = 1 -- stopped by heurestic
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
    displayOn = 0;
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
    displayOn = options.displayOn;
end
%
Nstep = 2;
nx = length(xl);
Vmax = (xu-xl)/Nstep;
options.fEvals = zeros(maxIter+1,nx+1);
fprintf('PSO start...\n')
%% Initialise the swarm
PNodes = {};
SwarmHeads = {};

gNode = PSO_NODE();
for i=1:N
    Node = PSO_NODE();
    Node.id = i;
    Node.xl = xl;
    Node.xlk = xl;
    Node.xu = xu;
    Node.xuk = xu;
    Node.swarm_id = 1;
    Node.vmin = -Vmax;
    Node.vmax = Vmax;   
    Node.vk = 2*Node.vmax.*rand(nx,1)-Node.vmax;    
    if isempty(pop)
        Node.xk = (xu-xl).*rand(nx,1)+xl;
    else
        Node.xk = pop(:,i);
    end
    Node.ub = f(Node.xk);%evaluate node function value
    if Node.ub < gNode.ub
        gNode = Node;%node becomes main swarm head
    end    
    Node.pk = Node.xk;   
    PNodes{end+1} = Node;%add particle node
end
SwarmHeads{end+1} = gNode;

gFuncPrev = 0;
maxStalli = 0;
options.fEvals(1,1) = gNode.ub;
options.fEvals(1,2:end) = gNode.xk;
swarm_colors = ['b','g','v','y','m','c'];
%%
if nx == 2 && displayOn == 1%if problem dimension is 2 and displayOn is 1
    d = (max(xu)-min(xl))/100;
    [x1,x2] = meshgrid(xl(1):d:xu(1),xl(2):d:xu(2));
    y = obj_func2d(f,x1,x2);
    figure(10)
     clf
     hold on
     contour(x1,x2,y,'ShowText','on')
     for ii=1:N
       Node = PNodes{ii};
        plot(Node.xk(1),Node.xk(2),'ro','LineWidth',2,'MarkerFaceColor','r')
     end
     %plot(lgk(1),lgk(2),'co','LineWidth',2,'MarkerFaceColor','c')
     plot(gNode.xk(1),gNode.xk(2),'go','LineWidth',2,'MarkerFaceColor','g')
     title(sprintf('DPSO: f(x^*) = %.5f - Iter %d',gNode.ub,0))
     axis([xl(1) xu(1) xl(2) xu(2)])   
end
%% PSO iterations
iter = 0;
reMaxStalli = 0;
rho = 10^(-3);
%computer radious of basin of convergence
rx = xu-xl;
sx = 1;
for ii=1:nx
   sx = sx*rx(ii);
end
rgk = (sx*0.01)^(1/nx);
%------------------------------------------
 
while iter < maxIter   
   %---
   Ns = length(SwarmHeads); 
   for s=1:Ns%go through every swarm
       %set adaptive inertial weight for the swarm
       sGNode = SwarmHeads{s};%get swarm head
       w = wmax-(wmax-wmin)*(iter-sGNode.swarm_sIter)/maxIter;      
       curGNodek = sGNode;
       for i=1:N
           Node = PNodes{i};%pick particle
           if Node.swarm_id == s %treat particle per swarm
               %store previous coordinate
               Node.xk1 = Node.xk;
               %get velocity for next iteration
               Node.vk = w*Node.vk + cr*rand(nx,1).*(Node.pk-Node.xk)+...
                   sr*rand(nx,1).*(sGNode.xk-Node.xk);
               Node = Node.vBound();%close out the bounds
               Node = Node.nextIter();%move to the next iter
               fpk = f(Node.xk);
               if fpk < Node.ub %update node experience
                   Node.ub = fpk;
                   Node.pk = Node.xk;
               end
               
               if Node.ub < curGNodek.ub %udpate gBest per swarm
                   curGNodek = Node;
               end
           end
           PNodes{i} = Node;%update Node info
       end    
       if curGNodek.ub < gNode.ub%update global swarm 
           gNode = curGNodek;
       end
       SwarmHeads{s} = curGNodek;%update swarm head
   end
   
   if displayOn == 1 && nx == 2
           figure(10)
           for s=1:Ns             
             for i=1:N
                Node = PNodes{i};
                plot([Node.xk(1);Node.xk1(1)],[Node.xk(2);Node.xk1(2)],'r--')
                plot(Node.xk(1),Node.xk(2),[swarm_colors(s),'*'],'LineWidth',1)
             end   
                sGNode = SwarmHeads{s};
                plot(sGNode.xk(1),sGNode.xk(2),[swarm_colors(s),'o'],'LineWidth',2,'MarkerFaceColor',swarm_colors(s))                
           end
           plot(gNode.xk(1),gNode.xk(2),'go','LineWidth',3,'MarkerFaceColor','g')
           title(sprintf('DPSO: f(x^*) = %.5f - Iter:%d - Stall Iter:%d - Nr: %d',gNode.ub,iter,reMaxStalli,1))
   end     
   %%
   % heurestic stop
   if EnStopHeurestic == 1
       if abs(gBUB-gFuncPrev) < geps
         maxStalli = maxStalli+1;
       else
         maxStalli = max(maxStalli-1,0);
       end
         gFuncPrev = gBUB; 
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
   options.fEvals(iter+1,1) = gNode.ub;
   options.fEvals(iter+1,2:end) = gNode.xk;
   
   if displayOn == 1
       fprintf('PSO Iteration count: %d of %d\n',iter,maxIter)
   end
end
%set output
if EnPostOptimisation == 1
    [gk,BUB] = fmincon(f,gNode.xk,[],[],[],[],xl,xu,[],post_opts);
    %[gk,BUB] = patternsearch(f,gNode.pk,[],[],[],[],xl,xu);
    gNode.xk = gk;
    gNode.pk = gk;
    gNode.ub = BUB;
end
options.elites =  [];
xopt = gNode.xk;
fopt = gNode.ub;
options.iter = iter;
tStop = clock();
options.teps = etime(tStop,tStart);
fprintf('PSO stop...\n');
end