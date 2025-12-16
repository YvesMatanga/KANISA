function [xopt,fopt,options] = SN_DE_SOLVE(fr,xl,xu,options)
%% This is an implemntation of Sequential niching Differential evolution
%options flag = 0 -- maximum iteration reached
%options flag = 1 -- stopped by heurestic
fprintf('DE start...\n')
tStart = clock();
if nargin < 4
    maxIter = 40;
    N = 30;    
    F = 0.5;
    cr = 0.4;
    cross_over_option = 'default';
    maxStallIter = 10;
    geps = 10^(-3);
    EnStopHeurestic  = 0;
    EnPostOptimisation = 0;    
    post_opts = [];
    displayOn = 0;
    reMaxStallIter = 3;
    maxClusters = 5;    
    r = 0.5*computeBasin(xu,xl,maxClusters);    
else
    maxIter = options.maxIter;
    N = options.popSize;    
    F = options.F;
    cross_over_option = options.cross_over_option;
    maxStallIter = options.maxStallIter;
    reMaxStallIter = options.reMaxStallIter;
    geps = options.geps;
    EnStopHeurestic = options.EnStopHeurestic;
    EnPostOptimisation = options.EnPostOptimisation; 
    cr = options.cr;
    post_opts = options.post_opts;   
    displayOn = options.displayOn;
    maxClusters = options.maxClusters;
    r = 0.5*computeBasin(xu,xl,maxClusters);   
end
f = @(x)fr(x.*(xu-xl)+xl);
xli = xl;
xui = xu;

%%
nx = length(xl);
xu = ones(nx,1);
xl = zeros(nx,1);
fopt = +Inf;%set best upper bound
xopt = ones(nx,1)*Inf;
%% Initialisation
PopHeads = {};
TabooCells = {};
%PopPool = {};
[DNodes,gNode] = create_pop(f,xu,xl,xu,xl,N,{},1,TabooCells,r,0);%Initialisation DE evolution
%PopPool{end+1} = DNodes;

MainPopGNode = DE_NODE();%
%%
gNode
PopHeads{end+1} = gNode;
options.optCurve(1,1) = gNode.ub;
options.optCurve(1,2:(2+nx-1)) = gNode.xk';

reMaxStalli = 0;%restart maximum stall iteration
gFuncPrev = Inf;
NumCompPart = ceil(min(7*nx/2,15));%Number of completers particles
%%

iter = 0;
%%
if displayOn == 1  && nx < 3
    figureOn = 0;
    figureOn = displayPopulation(f,DNodes,gNode,xl,xu,iter,figureOn,TabooCells,r);
end
%%
exitMainSwarm = 0;
Ne = N;
%%
while iter < maxIter && (exitMainSwarm ~= 1)
%%
    mPopHead = PopHeads{1};%get main population head
%%
    for j=1:Ne
        Node = DNodes{j};        
        if Node.pop_id ~= 1
            continue;
        end     
%% Mutation
        VNode = create_mutant(Node,N,F,DNodes);
%% Crossover      
        XNode = cross_over(Node,VNode,cr,cross_over_option);        
%% Next generation selection        
        Node.ub = f(Node.xk);       
        XNode.ub = f(XNode.xk);        
%% Repulsion function
        Node.ub = objFunc_corrector(Node,TabooCells,r);
        XNode.ub = objFunc_corrector(XNode,TabooCells,r);
%% Next Generation particle selection
        if XNode.ub <= Node.ub
            Node = XNode;%next generation update            
        end      
%% Optimum update    
        if Node.ub < MainPopGNode.ub
            MainPopGNode = Node;
        end
        
        if MainPopGNode.ub < gNode.ub
            gNode = Node;            
        end
        %Node.id = j;%update id
        DNodes{j} = Node;%move to next generation
    end    
    iter = iter + 1;   
    fprintf('Iteration %d of %d\n',iter,maxIter);
%%  Niching Discovery Test
    if abs(MainPopGNode.ub-gFuncPrev) < geps
        reMaxStalli = reMaxStalli + 1;
    else
        reMaxStalli = max(reMaxStalli-1,0);
    end
    gFuncPrev = MainPopGNode.ub;
    %------------- || --------------%
    if reMaxStalli > reMaxStallIter
        reMaxStalli = 0;%optimisation restart reached
        gFuncPrev = Inf;       
        Np = length(PopHeads);%Number of subpopulations
        if Np < maxClusters+1 %check if the number of subpopulations have been reached
            isClose = closenessTest(MainPopGNode,TabooCells,r);
            if isClose == 0 %check if the node is not too close to the taboo cells
                fprintf('New Cluster found: %d \n',Np+1);
                MainPopGNode.cx = MainPopGNode.xk;               
                Np = Np +1;%increment the number of subpopulations
                xlk = MainPopGNode.cx - r;
                xuk = MainPopGNode.cx + r;     
                CloneNodes = cloneNodes(MainPopGNode,NumCompPart,DNodes,xuk,xlk);%take NumCompart nodes that are closest to the Pop Best and within the problem space                              
                %create new swarm particles
                [NewPop,curBestNode] = create_pop(f,xu,xl,xuk,xlk,N,CloneNodes,Np,TabooCells,r,1);%Initialisation DE evolution
                PopHeads{end+1} = curBestNode;
                %update optimum
                if curBestNode.ub < gNode.ub
                    gNode = Node;            
                end
                %-- create a Taboo Cell
                MainPopGNode.pop_id = Np;
                TabooCells{end+1} = MainPopGNode;
                %-- Add Sub Population to Population Pool
                DNodes = addToPopPool(DNodes,NewPop);
                %--  Generate Main Subpopulation
                DNodes = clearMainPop(DNodes);
                if Np < maxClusters+1                    
                    [NewPop,curBestNode] = create_pop(f,xu,xl,xu,xl,N,{},1,TabooCells,r,0);%Initialisation DE evolution
                    PopHeads{1} = curBestNode;
                    MainPopGNode = curBestNode;
                    DNodes = addToPopPool(DNodes,NewPop);%add new subpopulation to pop pool
                    Ne = length(DNodes);
                else
                    exitMainSwarm = 1;
                end
                %--
            end
        end
    end
%% Display
   if displayOn == 1 && nx < 3
       figureOn = displayPopulation(f,DNodes,gNode,xl,xu,iter,figureOn,TabooCells,r);
   end    
end
% for i=1:length(DNodes)
%     Node = DNodes{i};
%     Node
% end
fprintf('Maximum population clustered reached....iter: %d\n',iter);
size(DNodes)
%pause

%% Summary of Discovery Steps
NBudget = maxIter - iter;%remaining number of iterations
%% Parallel Search Refinement
tic
pWorkers(1:maxClusters) = parallel.FevalFuture;
for idx = 1:maxClusters
    SubPop = getSubPopulation(idx+1,DNodes);
    pWorkers(idx) = parfeval(@fineSearch,2,f,F,cr,cross_over_option,SubPop,idx+1,NBudget,TabooCells,r,0);
end

SubPopulations = cell(maxClusters,1);
for idx=1:maxClusters
    [completedId,value] = fetchNext(pWorkers);
    fprintf('Got result with index: %d.\n', completedId);
    SubPopulations{completedId} = value;
%     Cluster = SubPopulations{completedId};
%     class(Cluster)
%     Cluster
%     Node = Cluster{1};
%     disp('Population')
%     Node
end
toc
DNodes_out = cell(maxClusters*N,1);
PopHeads = cell(maxClusters);
k = 1;
for ci =1:maxClusters
    Cluster = SubPopulations{ci};
    Nj = length(Cluster);
    Head = DE_NODE();
    for j=1:Nj
        Node = Cluster{j};
        %gNode
        DNodes_out{k} = Node;        
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

if displayOn == 1
    displayPopulation(f,DNodes_out,gNode,xl,xu,iter,figureOn,TabooCells,r);
end
%f = parfeval(@fineSearch,2,f,F,cr,cross_over_option,DNodes,2,NBudget,TabooCells,r,0);
%value = fetchOutputs(f)
%[SubPopNodes,curPopHead] = fineSearch(f,F,cr,cross_over_option,DNodes,3,NBudget,TabooCells,r,displayOn)
%disp('here')
%%
if EnPostOptimisation == 1
    for i=1:maxClusters       
        curHead = PopHeads{i};
        [gk,BUB] = fmincon(f,curHead.xk,[],[],[],[],xl,xu,[],post_opts); 
        curHead.xk = gk;
        curHead.ub = BUB;
        if curHead.ub < gNode.ub
            gNode = curHead;
        end
    end
    fprintf('Sub-population ID:%d\n',gNode.pop_id);
end
xopt = gNode.xk;
fopt = gNode.ub;

xopt = xopt.*(xui-xli)+xli;
tStop = clock();
options.teps = etime(tStop,tStart);
fprintf('DE stop...\n')
end

function XNode = cross_over(Node,VNode,cr,option)
  XNode = Node; 
  if strcmp(option,'default') == 1
      nx = length(Node.xk);
      XNode.xk = zeros(nx,1);
      for i=1:nx
          p = rand();
          if p <= cr
              XNode.xk(i) = VNode.xk(i);
          else
              XNode.xk(i) = Node.xk(i);
          end
      end
  else
      error('Cross over option not set...');
  end
end


function [DNodes,bNode] = create_pop(f,xu,xl,xuk,xlk,N,SupportNodes,popId,TabooCells,rk,collision)
      bNode = DE_NODE();
      DNodes = cell(N,1);
      nx = length(xl);
      
      Ns = length(SupportNodes);%number of support nodes
      for i=1:Ns
          Node = SupportNodes{i};
          Node.id = i;
          Node.collision = collision;%set collision boundary
          Node.pop_id = popId;
          %Node.xk = (xu-xl).*rand(nx,1) + xl;
          Node.xu = xu;
          Node.xl = xl;     
          Node.xuk = xuk;
          Node.xlk = xlk;
          Node.ub = f(Node.xk);
          %--correct objective function if need be -----
          Node.ub = objFunc_corrector(Node,TabooCells,rk); 
          %---------------------------------------------
          if  Node.ub <= bNode.ub
              bNode = Node;
          end              
          DNodes{i} = Node; 
      end
      
      for i=(Ns+1):N
          Node = DE_NODE();
          Node.id = i;
          Node.pop_id = popId;
          Node.xk = (xuk-xlk).*rand(nx,1) + xlk;
          
          Node.collision = collision;%set collision boundary
          Node.xu = xu;
          Node.xl = xl;        
          Node.xuk = xuk;
          Node.xlk = xlk;  
          Node = Node.xBound();%limit node coverage
          Node.ub = f(Node.xk);
          %--correct objective function if need be -----
          Node.ub = objFunc_corrector(Node,TabooCells,rk); 
          %---------------------------------------------
          if  Node.ub <= bNode.ub
              bNode = Node;
          end       
          DNodes{i} = Node;
      end
      
      if bNode.ub == Inf
          error('Population singularity occured...')
      end
end

function VNode = create_mutant(Node,n,Fsc,DNodes)    
    ri = randperm(n);
    ri(find(ri==Node.id))=[];%remove node index
    %select three random indices different from index array
    r = ri(1);p = ri(2);q = ri(3);   
    VNode = Node;%DNodes{nodei};%create new Node
    Nodep = getNode(p, Node.pop_id,DNodes);%DNodes{p};
    Nodeq = getNode(q, Node.pop_id,DNodes);%DNodes{q};
    Noder = getNode(r, Node.pop_id,DNodes);%DNodes{r};
    
    VNode.xk = Nodep.xk + Fsc*(Nodeq.xk-Noder.xk);
    VNode.xl = Nodep.xl;
    VNode.xu = Nodep.xu;
    VNode.xlk = Nodep.xlk;
    VNode.xuk = Nodep.xuk;   
    VNode = VNode.xBound();%bound to problem space
end

function dNode = getNode(nodeId,popId,DNodes)
  n = length(DNodes);
  for j=1:n
      Node = DNodes{j};
      if Node.pop_id ~= popId
          continue;
      end
      
      if Node.id == nodeId
          dNode = Node;
          return;
      end
  end
  error('inconsistent results... get node')
end


function [figureOnOut] = displayPopulation(f,DNodes,gNode,xl,xu,iter,figureOn,TabooCells,rgk)
    figureOnOut = figureOn;
    pop_colors = ['r','k','b','y','m','c','g'];
    N = length(DNodes);  
    nx = length(xl);    
    %global figureOn
    if nx==1
        global xn yn
        if figureOn == 0            
            d = (max(xu)-min(xl))/1000;
            xn = xl:d:xu;
            yn = obj_func1d(f,xn);
            figureOnOut = 1;
        end
        
        figure(10)
        clf
        plot(xn,yn,'r')
        hold on
        for ii=1:N
            Node = DNodes{ii};
            plot(Node.xk,f(Node.xk),[pop_colors(Node.pop_id),'o'],'LineWidth',2,'MarkerFaceColor',pop_colors(Node.pop_id))
        end        
        plot(gNode.xk,f(gNode.xk),'go','LineWidth',2,'MarkerFaceColor','g')
        title(sprintf('DE: f(x^*) = %.5f - Iter %d',gNode.ub,iter))        
        axis([xl(1) xu(1) min(yn) max(yn)])    
    elseif nx == 2 %if problem dimension is 2 and displayOn is 1
        global x1 x2 y
        if figureOn == 0          
            d = (max(xu)-min(xl))/100;
            [x1,x2] = meshgrid(xl(1):d:xu(1),xl(2):d:xu(2));
            y = obj_func2d(f,x1,x2);
            figureOnOut = 1;
        end
        figure(10)
         clf
         hold on
         contour(x1,x2,y,'ShowText','on')
         for ii=1:N
            Node = DNodes{ii};
            if Node.xk(1) > xu(1) || Node.xk(1) < xl(1) || Node.xk(2) > xu(2) || Node.xl(2) < xl(2)
                error('inconsistency...')
            end
            plot(Node.xk(1),Node.xk(2),[pop_colors(Node.pop_id),'o'],'LineWidth',2,'MarkerFaceColor',pop_colors(Node.pop_id))
         end
         plot(gNode.xk(1),gNode.xk(2),'go','LineWidth',2,'MarkerFaceColor','g')
         title(sprintf('DE: f(x^*) = %.5f - Iter %d',gNode.ub,iter))        
         axis([xl(1) xu(1) xl(2) xu(2)])   
         
         NT = length(TabooCells);
         for ii=1:NT
             TNode = TabooCells{ii};
             viscircles([TNode.cx(1) TNode.cx(2)],rgk,'Color',pop_colors(TNode.pop_id))
         end
    end
end

function fk = objFunc_corrector(Node,TabooCells,r)
%Objective function corrector based on Taboo cells
   fk = Node.ub;
   NT = length(TabooCells);
   for i=1:NT
       TNode = TabooCells{i};
       if norm(Node.xk-TNode.xk) <= r
           fk = Inf;
           break;
       end
   end
end
%function to test if a node is close to the TabooCells
function bool = closenessTest(MainPopGNode,TabooCells,r)
   bool = 0;
   NT = length(TabooCells);
   for i=1:NT
       TNode = TabooCells{i};
       if norm(TNode.cx-MainPopGNode.xk) <= r
           bool = 1;
           break;
       end
   end   
end

%computer basic of convergence based on problem space
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

%This function takes num_clone Nodes that are closed to the refNode
function CloneNodes = cloneNodes(refNode,num_clone,PopPool,sxu,sxl)
%select the num_clone nodes near refNode that are within a certain boundary
CNodes = {};
Np = length(PopPool);
for i=1:Np
    Node = PopPool{i};
    if Node.pop_id == refNode.pop_id %add node to the pool
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
ids = [];
for i=1:num_clone
    cNode = CNodes{i};%copy clone nodes
    if isInBound(cNode.xk,sxu,sxl)~= 1
        ids(end+1) = i;
    end
    CloneNodes{i} = cNode;%copy clone nodes
end
CloneNodes(ids) = [];%delete nodes
end

%check if a point is within a hyberbox
function y = isInBound(x,xui,xli)
    nx = length(xui);
    y = 1;
    for j=1:nx
        if x(j) > xui(j) || x(j) < xli(j)
          y = 0;
          return
        end
    end
end
%%
function PopOut = addToPopPool(Pop_in,NewPop)
 ns = length(NewPop);
 PopOut = Pop_in;
 for i=1:ns
     PopOut{end+1} = NewPop{i};
 end
end

%% clear node of the main subpopulation
function Pop_out = clearMainPop(Pop_in)
  np = length(Pop_in);
  Pop_out = Pop_in;
  ids = [];
  for i=1:np
      Node = Pop_out{i};
      if Node.pop_id ==1
          ids(end+1) = i;
      end
  end
  Pop_out(ids) = [];
end

function [n] = getPopSize(popId,DNodes)
 np = length(DNodes);
 n = 0;
 for i=1:np
     Node = DNodes{i};
     if Node.pop_id == popId
         n=n+1;
     end
 end
end

function [SubPopNodes,curPopHead] = fineSearch(f,Fsc,crc,cross_over_option,DNodes_in,popId,NIter,TabooCells,rgk,dispOn)  
    SampleNode = DNodes_in{1};
    xli = SampleNode.xl;
    xui = SampleNode.xu;
    nx = length(xli);
    %Nsb = getPopSize(popId,DNodes_in);
    
    iteri =0;
    curPopHead = DE_NODE();    
    while iteri < NIter
        Ne = length(DNodes_in);
        for i=1:Ne
            Node = DNodes_in{i};
            if Node.pop_id ~= popId
                continue;
            end       
            %Node     
            %% Mutation
            VNode = create_mutant(Node,Ne,Fsc,DNodes_in);
            %% Crossover      
            XNode = cross_over(Node,VNode,crc,cross_over_option);        
            %% Next generation selection        
            Node.ub = f(Node.xk);       
            XNode.ub = f(XNode.xk); 
            %% Next Generation particle selection
            if XNode.ub <= Node.ub
                Node = XNode;%next generation update            
            end 
            %% Optimum update
            if Node.ub < curPopHead.ub
                curPopHead = Node;
            end                
            DNodes_in{i} = Node;%move to next generation
        end              
        iteri = iteri+1;
        %iter = iter + 1;
        if dispOn == 1 && nx < 3
            displayPopulation(f,DNodes_in,curPopHead,xli,xui,iteri,1,TabooCells,rgk);
        end   
    end    
    SubPopNodes = DNodes_in;
end

function SubPop = getSubPopulation(popId,DNodes_in)
  SubPop = {};
  Ne = length(DNodes_in);
  for i=1:Ne
      Node = DNodes_in{i};
      if Node.pop_id == popId
          SubPop{end+1} = Node;
      end
  end
end