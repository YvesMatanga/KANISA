function [xopt,fopt,options] = SN_PSO_SOLVE(fr,xl,xu,options)
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
    reMaxStallIter = 20;
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
    reMaxStallIter = options.reMaxStallIter;
end

%%
f = @(x)fr(x.*(xu-xl)+xl);
xli = xl;
xui = xu;
%%
nx = length(xl);
xu = ones(nx,1);
xl = zeros(nx,1);
%%
Nstep = 2;
Vmax = (xu-xl)/Nstep;
NumCompPart = ceil(min(7*nx/2,15));%Number of completers particles
maxSwarmHeads = options.maxSwarmHeads;
maxClusters = maxSwarmHeads;
options.optCurve = zeros(maxIter+1,nx+1);
% if displayOn == 1
    fprintf('SN - PSO start...\n')
% end
%% Computer Radius of Basin of Convergence
% rx = xu-xl;
% sx = 1;
% for ii=1:nx
%     sx = sx*rx(ii);
% end
% rgk = (sx*0.01)^(1/nx);
rgk = 0.5*computeBasin(xu,xl,maxClusters);
%% Initialise the swarm
SwarmHeads = {};
TabooCells = {};
[PNodes,gNode] = createSwarm(f,N,{},1,xl,xu,xl,xu,-Vmax,Vmax,TabooCells,rgk);%create initial swarm with
%N particles bounded by the problem space and with default velocity
SwarmHeads{end+1} = gNode;
gFuncPrev = 0;
maxStalli = 0;
options.optCurve(1,1) = gNode.ub;
options.optCurve(1,2:end) = gNode.xk;
swarm_colors = ['b','k','r','y','m','c'];
override = 1;
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
    hold off
end
%% PSO iterations
iter = 0;
reMaxStalli = 0;
rho = 10^(-3);
exitMainSwarm = 0;
while iter < maxIter && (exitMainSwarm ~= 1)
    Ns = length(SwarmHeads);
    %set adaptive inertial weight for the swarm
    sGNode = SwarmHeads{1};%get swarm head
    w = wmax-(wmax-wmin)*(iter-sGNode.swarm_sIter)/maxIter;
    curGNodek = sGNode;
    Np = length(PNodes);
    %fprintf('Number of particles:%d\n',Np);
    for i=1:Np
        Node = PNodes{i};%pick particle
        if Node.swarm_id ~= 1
            continue;
        end
        %store previous coordinate
        Node.xk1 = Node.xk;
        %get velocity for next iteration
        Node.vk = w*Node.vk + cr*rand(nx,1).*(Node.pk-Node.xk)+...
            sr*rand(nx,1).*(sGNode.xk-Node.xk);
        Node = Node.vBound();%close out the bounds
        Node = Node.nextIter();%move to the next iter
        %% Repulsion
        fpk = f(Node.xk);
        if Node.swarm_id ==1
            kr = 1:Ns-1;
            for kk= kr%2:Ns
                curHead = TabooCells{kk};
                dx = norm(Node.xk - curHead.cx);
                if dx <= rgk % the point in the tabu area
                    fpk = Inf;
                    break;
                end
            end
        end
        Node = Node.xBound();
        if fpk < Node.ub %update node experience
            Node.ub = fpk;
            Node.pk = Node.xk;
        end
        
        if Node.ub < curGNodek.ub %udpate gBest per swarm
            curGNodek = Node;
        end
        PNodes{i} = Node;%update Node info
    end
    fprintf('Iteration %d of %d\n',iter+1,maxIter);
    if curGNodek.ub < gNode.ub%update global swarm
        gNode = curGNodek;
    end
    SwarmHeads{1} = curGNodek;%update swarm head
    mGNode = SwarmHeads{1};
    mBUB = mGNode.ub;
    %% Restart Test
    if abs(mBUB-gFuncPrev) < geps
        reMaxStalli = reMaxStalli+1;
    else
        %reMaxStalli = max(reMaxStalli-5,0);
        reMaxStalli = max(reMaxStalli,0);
    end
    gFuncPrev = mBUB;
    if reMaxStalli > reMaxStallIter
        reMaxStalli = 0;%%optimisation restart reached
        Ns = length(SwarmHeads);
        if Ns < maxSwarmHeads+1%check if the maximum number of swarm heads is not reached
            tooClose = 0;%check if the new node is too close to the existing ones
            for ss=1:Ns-1
                sCurHead = TabooCells{ss};%get current swarm head
                if norm(sCurHead.cx-mGNode.pk) <= rgk
                    tooClose = 1;
                    %mGNode
                    if nx ==2 && displayOn == 1
                        plot(mGNode.pk(1),mGNode.pk(2),[swarm_colors(s),'o'],'LineWidth',1,'MarkerFaceColor','y')
                        disp('pause')
                        pause
                    end
                    break;
                end
            end
            if tooClose == 0 %%if it is not too close to any node
                fprintf('New Cluster found: %d\n',Ns);
                mGNode.cx = mGNode.pk;
                if (mGNode.ub < gNode.ub || Ns == 1) || override %% if it is lower than what has been current found
                    %fprintf('best gid:%d\n',mGNode.id)
                    if displayOn == 1 && nx ==2
                        figure(10) %clear figure
                        clf
                        hold on
                        contour(x1,x2,y,'ShowText','on')
                        axis([xl(1) xu(1) xl(2) xu(2)])
                        hold off
                    end
                    CloneNodes = cloneNodes(mGNode,NumCompPart,PNodes);%Clone Nc Particles close to mGNode
                    Ns = Ns + 1;%increment swarm list
                    xlk = mGNode.cx-rgk;
                    xuk = mGNode.cx+rgk;
                    vk = (xuk-xlk)/2;
                    [NewSwarm,cur_gNode] = createSwarm(f,N,CloneNodes,Ns,xl,xu,xlk,xuk,-vk,vk,TabooCells,rgk);%create new swarm
                    %cur_gNode
                    SwarmHeads{Ns} = cur_gNode;%update swarm head
                    PNodes = addToSwarmPool(PNodes,NewSwarm);%add new swarm to swarm pool
                    if cur_gNode.ub < gNode.ub%update gNode
                        gNode = cur_gNode;
                    end
                    %create a TabooCell
                    mGNode.swarm_sIter = iter;
                    TabooCells{end+1} = mGNode;%Taboo cell
                    if nx == 2 && displayOn == 1
                        for kk=1:Ns-1
                            curHead = TabooCells{kk};
                            viscircles([curHead.cx(1) curHead.cx(2)],rgk,'Color',swarm_colors(kk))
                        end
                    end
                    % Generate main swarm
                    PNodes = clearMainSwarm(PNodes);%clear main swarm
                    if Ns < maxSwarmHeads+1
                        [MainSwarm,mGNode] = createSwarm(f,N,{},1,xl,xu,xl,xu,-Vmax,Vmax,TabooCells,rgk);%recreate initial swarm
                        SwarmHeads{1} = mGNode;%update new swarm Head
                        PNodes = addToSwarmPool(PNodes,MainSwarm);%add new swarm to swarm pool
                    else
                        exitMainSwarm = 1;
                    end
                end
            end
        end
    end
    %%
    if displayOn == 1 && nx == 2
        figure(10)
        d = (max(xu)-min(xl))/100;
        [x1,x2] = meshgrid(xl(1):d:xu(1),xl(2):d:xu(2));
        y = obj_func2d(f,x1,x2);
        figure(10)
        clf
        hold on
        contour(x1,x2,y,'ShowText','on')
        mess = '';
        for si=1:Ns
            Np = length(PNodes);
            for i=1:Np
                Node = PNodes{i};
                if Node.swarm_id == si
                    plot([Node.xk(1);Node.xk1(1)],[Node.xk(2);Node.xk1(2)],[swarm_colors(si),'--'])
                    plot(Node.xk(1),Node.xk(2),[swarm_colors(si),'*'],'LineWidth',1)
                end
            end
            sGNode = SwarmHeads{si};
            plot(sGNode.xk(1),sGNode.xk(2),[swarm_colors(si),'o'],'LineWidth',2,'MarkerFaceColor',swarm_colors(si))
            plot(sGNode.xk(1),sGNode.xk(2),[swarm_colors(si),'o'],'LineWidth',1,'MarkerFaceColor','r')
            if si > 1
                TNode = TabooCells{si-1};
                viscircles([TNode.cx(1) TNode.cx(2)],rgk,'Color',swarm_colors(si))
            end
        end
        plot(gNode.xk(1),gNode.xk(2),'go','LineWidth',3,'MarkerFaceColor','g')
        plot(gNode.xk(1),gNode.xk(2),'go','LineWidth',2,'MarkerFaceColor','r')
        title(sprintf('DPSO: f(x^*) = %.5f - Iter:%d - Stall Iter:%d - Nr: %d',gNode.ub,iter,reMaxStalli,Ns))
        xlabel(mess)
        axis([xl(1) xu(1) xl(2) xu(2)])
        hold off
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
    options.optCurve(iter+1,1) = gNode.ub;
    options.optCurve(iter+1,2:end) = gNode.xk;
end
citer = iter;
fprintf('Maximum swarm heads reached....iter: %d\n',iter);
NBudget = maxIter - iter;%remaining number of iterations
%% Parallel Search Refinement
tic
pWorkers(1:maxClusters) = parallel.FevalFuture;
for idx = 1:maxClusters
    SubPop = getSubPopulation(idx+1,PNodes);
    %fprintf('SubSwarm Length: %d\n',length(SubPop))
    pWorkers(idx) = parfeval(@fineSearch,2,f,wmax,wmin,sr,cr,SubPop,idx+1,NBudget,maxIter,TabooCells);
end

SubPopulations = cell(maxClusters,1);
for idx=1:maxClusters
    [completedId,value] = fetchNext(pWorkers);
    fprintf('Got result with index: %d.\n', completedId);
    SubPopulations{completedId} = value;
end
toc
PNodes_out = cell(maxClusters*N,1);
PopHeads = cell(maxClusters);
k = 1;
for ci =1:maxClusters
    Cluster = SubPopulations{ci};
    Nj = length(Cluster);
    Head = PSO_NODE();
    for j=1:Nj
        Node = Cluster{j};
        %gNode
        PNodes_out{k} = Node;        
        if Node.ub < Head.ub
            Head = Node;
        end
        %Node
        if Node.ub < gNode.ub
            gNode=Node;
        end
        k=k+1;        
    end
    PopHeads{ci} = Head;
end

%% set output
if EnPostOptimisation == 1
    for i=1:maxClusters       
        curHead = PopHeads{i};
        fprintf('local search %d\n',i);
        [gk,BUB] = fmincon(f,curHead.xk,[],[],[],[],xl,xu,[],post_opts); 
        %[gk,BUB] = patternsearch(f,curHead.pk,[],[],[],[],xl,xu);
        curHead.xk = gk;
        curHead.ub = BUB;
        if curHead.ub < gNode.ub
            gNode = curHead;
        end
    end
    fprintf('Sub-population ID:%d\n',gNode.swarm_id);
end
xopt = gNode.xk;
fopt = gNode.ub;
options.elites =  [];
options.iter = iter;
xopt = xopt.*(xui-xli)+xli;
tStop = clock();
options.teps = etime(tStop,tStart);
fprintf('SN - PSO stop...\n');
end

%% createSwarm
function [SwarmNodes,cur_gNode] = createSwarm(func,num_par,SupportNodes,swarm_id,pxl,pxu,xli,xui,v_min,v_max,TabooCells,R)
SwarmNodes = {};
n_s = length(SupportNodes);%number of support nodes
Np = num_par - n_s;%number of nodes to be created
node_id = 1;
cur_gNode = PSO_NODE();
nx = length(xli);
Nt = length(TabooCells);
%add support node to swarm
for i=1:n_s
    Node = SupportNodes{i};
    Node.swarm_id = swarm_id;
    Node.id = node_id;
    node_id = node_id + 1;
    Node.xl = pxl;
    Node.xu = pxu;
    Node.xlk = xli;
    Node.xuk = xui;
    Node.vmin = v_min;
    Node.vmax = v_max;
    %objective function
    Node.ub = func(Node.xk);%evaluate node function value
    %% Taboo regions
    if Node.swarm_id == 1
        for ii=1:Nt
            TabooNode = TabooCells{ii};
            dk = norm(Node.xk - TabooNode.cx);
            if dk <= R
                Node.ub = inf;
                break;
            end
        end
    end
    %%
    Node.pk = Node.xk;
    %store previous coordinate
    Node.xk1 = Node.xk;
    if Node.ub < cur_gNode.ub
        cur_gNode = Node;%node becomes main swarm head
    end
    SwarmNodes{end+1} = Node;
end

for i=1:Np
    Node = PSO_NODE();
    Node.id = node_id;
    node_id = node_id +1;
    Node.xl = pxl;
    Node.xlk = xli;
    
    Node.xu = pxu;
    Node.xuk = xui;
    Node.swarm_id = swarm_id;
    Node.vmin = v_min;
    Node.vmax = v_max;
    Node.vk = 2*Node.vmax.*rand(nx,1)-Node.vmax;
    Node.xk = (xui-xli).*rand(nx,1)+xli;
    Node.ub = func(Node.xk);%evaluate node function value
    %% Taboo regions
    if Node.swarm_id == 1
        for ii=1:Nt
            TabooNode = TabooCells{ii};
            dk = norm(Node.xk - TabooNode.cx);
            if dk <= R
                Node.ub = inf;
                break;
            end
        end
    end
    %%
    Node.pk = Node.xk;
    %store previous coordinate
    Node.xk1 = Node.xk;
    if Node.ub < cur_gNode.ub
        cur_gNode = Node;%node becomes main swarm head
    end
    SwarmNodes{end+1} = Node;%add Node to Swarm
end
end

%This function takes num_clone Nodes that are closed to the refNode
function CloneNodes = cloneNodes(refNode,num_clone,SwarmPool)
CNodes = {};
Np = length(SwarmPool);
for i=1:Np
    Node = SwarmPool{i};
    if Node.swarm_id == refNode.swarm_id %add node to the pool
        Node.gDist = norm(Node.xk - refNode.cx);
        CNodes{end+1} = Node;
    end
end

Nc = length(CNodes);
for i=1:Nc
    for j=1:Nc-1
        Node1 = CNodes{j};
        Node2 = CNodes{j+1};
        if Node1.gDist > Node2.gDist
            tmpNode = Node1;
            Node1 = Node2;
            Node2 = tmpNode;
        end
        CNodes{j} = Node1;
        CNodes{j+1} = Node2;
    end
end

CloneNodes = cell(num_clone,1);
for i=1:num_clone
    CloneNodes{i} = CNodes{i};%copy clone nodes
end
end

%% add new swarm to swarm pool
function SwarmPool = addToSwarmPool(SwarmPool_in,New_Swarm)
ns = length(New_Swarm);
SwarmPool = SwarmPool_in;
for i=1:ns
    Node = New_Swarm{i};
    SwarmPool{end+1} = Node;
end
end

%% clear main swarm
function SwarmPool = clearMainSwarm(SwarmPool_in)
np = length(SwarmPool_in);
SwarmPool = SwarmPool_in;
ids = [];
for i=1:np
    Node = SwarmPool{i};
    if Node.swarm_id == 1
        ids(end+1) = i;
    end
end
SwarmPool(ids) = [];
end
%%
function [SubPopNodes,curPopHead] = fineSearch(f,wmax,wmin,sr,cr,PNodes_in,swarmId,NIter,maxIter,TabooCells)  
    SampleNode = PNodes_in{1};
    xli = SampleNode.xl;    
    nx = length(xli);
    %Nsb = getPopSize(popId,DNodes_in);
    
%% get population head    
    iteri =0;
    curPopHead = PSO_NODE();    
    Np = length(PNodes_in);
    for i=1:Np
        Node = PNodes_in{i};
        if Node.ub < curPopHead.ub
            curPopHead = Node;
        end
    end
%%  
    TNode = TabooCells{swarmId-1};
    while iteri < NIter
        w = wmax-(wmax-wmin)*(iteri+TNode.swarm_sIter)/(TNode.swarm_sIter+NIter);
        Tgk = curPopHead;
        for i=1:Np
            Node = PNodes_in{i};
            if Node.swarm_id ~= swarmId
                continue;
            end                 
            Node.xk1 = Node.xk;
            %get velocity for next iteration
            Node.vk = w*Node.vk + cr*rand(nx,1).*(Node.pk-Node.xk)+...
                sr*rand(nx,1).*(Tgk.xk-Node.xk);
            Node = Node.vBound();%close out the bounds
            Node = Node.nextIter();%move to the next iter
            %% Repulsion
            fpk = f(Node.xk);
            Node = Node.xBound();
            %%
            if fpk < Node.ub %update node experience
                Node.ub = fpk;
                Node.pk = Node.xk;
            end
            %% Optimum update
            if Node.ub < curPopHead.ub %udpate gBest per swarm
                curPopHead = Node;
            end              
            PNodes_in{i} = Node;%move to next generation
        end     
        iteri = iteri+1;        
    end    
    SubPopNodes = PNodes_in;
end


function SubPop = getSubPopulation(swarmId,PNodes_in)
  SubPop = {};
  Ne = length(PNodes_in);
  for i=1:Ne
      Node = PNodes_in{i};
      if Node.swarm_id == swarmId
          SubPop{end+1} = Node;
      end
  end
end

function rk = computeBasin(xu,xl,numClusters,method)
if nargin < 4
    rk = sqrt(length(xu))/(2*power(numClusters,1/length(xu)));
else
if strcmp(method,'user') == 1 
    rx = xu-xl;
    sx = 1;
    nx = length(xl);
    for ii=1:nx
        sx = sx*rx(ii);
    end
    rk = (sx*0.01)^(1/nx);
    rk = rk/2;
else
    error('method unknown..')
end    
end
end
