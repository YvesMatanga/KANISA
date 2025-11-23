function [xopt,fopt,perf_options] = ABB_SOLVE(f,xl,xu,options)
%ABB_SOLVE : Object oriented Programming based aBB implementation
%time stamp
tStart = clock();
%% Settings
nx = length(xl);%get dimension of problem
perf_options.iter = 1;%initialise iteration to 1
xopt = [];
fopt = Inf;%initialise best 
perf_options.lbPruningCount = 0;%Initialis pruning number by lb
perf_options.intPruningCount = 0;%Initialise pruning by interval analysis
perf_options.mntPruningCount = 0;%Initialise pruning count by monotony
perf_options.ncvxPruningCount = 0;%Initialise runing count by convexity
perf_options.intVsCvxCount = [0 0 -Inf];%Count which convex relaxation goes more deeper (LBI-int)
perf_options.teps_opt = -1;%time to global optimum
perf_options.gOptIter = -1;
perf_options.optIters = [];%record the function candidate optimum progress
optFound = 0;%optimum check boolean variable
lbubGvEps = options.eps*100;%Non-sensiscal violation lclc
%lower bound above upper bound
%% Initialisation 
BNodes = {};%Initialise Node Structure
globalId = 1;%set globalId to 1
RootNode = BNODE();
RootNode.id = globalId;%initialise id
RootNode.xl = xl;%set function left bound
RootNode.xu = xu;%set function right bound
if options.intprune == 1 %if interval analysis is enabled
    RootNode.If = f(infsup(xl,xu));%get interval bounds
end
BNodes{end+1} = RootNode;%% add node structure
%% Iteration
while ~isempty(BNodes) && perf_options.iter <= options.maxIter
%% Search Strategy
  curId = searchStrategy(BNodes,options.search);
  Node = BNodes{curId};%select node
%% Pruning by lower bound
  if Node.lb > fopt
      BNodes(curId) = [];
      perf_options.lbPruningCount = perf_options.lbPruningCount + 1;%count number of pruning by lower bounding
      if options.displayOn == 1
        disp('region pruned by lower bound...');
      end
      continue;
  end
%% Pruning by Interval analysis
  if options.intprune == 1
      if Node.If.inf > fopt
          BNodes(curId) = [];
          perf_options.intPruningCount = perf_options.intPruningCount  + 1;%count number of pruning by interval analysis
          if options.displayOn == 1
            disp('region pruned by interval analysis...');
          end
          continue;
      end      
  end
%% Calculate Interval Hessian
    Node.Ih = f(hessianinit(infsup(Node.xl,Node.xu)));%get interval hessian
%% Monotonicity Test
    if options.mntprune == 1
        %Node
        [xlj,xuj,flag,dimj] = monotony_reduce(Node.Ih,Node.xl,Node.xu);     
        watchflag = 0;
        if flag == 1 && Node.HasNewMonotony(dimj) == 1
            Node.xl = xlj;
            Node.xu = xuj;
            %Node.Ih = f(hessianinit(infsup(Node.xl,Node.xu)));%get interval hessian
            Node.If = f(infsup(Node.xl,Node.xu));
            %Node.updateMonotonySet(dimj);%update list of monotonic dimensions
            Node.mntIds = dimj;
            watchflag = 1;
            perf_options.mntPruningCount = perf_options.mntPruningCount + 1;
            if options.displayOn ==1
               disp('region reduced by monotonicity...')           
            end         
            BNodes{curId} = Node;%update node
            continue;
        end
    end   
%% Non Convexity test
    if options.ncvxprune == 1
        [xlj,xuj,xlk,xuk,flag,dimj] = nonconvexity_reduce(Node.Ih,Node.xl,Node.xu,xl,xu,Node.mntIds);
        if flag == 4     
            Node.xl = xlj;
            Node.xu = xuj;
            %Node.Ih = f(hessianinit(infsup(Node.xl,Node.xu)));%get interval hessian
            %Node.If = Node.Ih.x;
            Node.If = f(infsup(Node.xl,Node.xu));
            %Node.updateMonotonySet(dimj);           
            Node.mntIds(end+1) = dimj;
            BNodes{curId} = Node;
            perf_options.ncvxPruningCount = perf_options.ncvxPruningCount + 1;
            if options.displayOn ==1
               disp('region reduced by nonconvexity...')           
            end
            %disp('after..')
            %Create second node from second edge piece
            Node2 = BNODE();
            Node2.xl = xlk;
            Node2.xu = xuk;
            %Node2.Ih = f(hessianinit(infsup(Node2.xl,Node2.xu)));%get interval hessian
            %Node2.If = Node2.Ih.x;
            Node2.If = f(infsup(Node2.xl,Node2.xu));
            %Node2.updateMonotonySet(dimj);
            Node2.mntIds = dimj;
            Node2.lb = Node.lb;
            Node2.alpha = Node.alpha;
            BNodes{end+1} = Node2;%add Node
            continue;
        elseif flag == 3
            Node.xl = xlk;
            Node.xu = xuk;
            %Node.Ih = f(hessianinit(infsup(Node.xl,Node.xu)));%get interval hessian
            Node.If = f(infsup(Node.xl,Node.xu));
            %Node.updateMonotonySet(dimj);           
            Node.mntIds(end+1) = dimj;
            BNodes{curId} = Node;
            perf_options.ncvxPruningCount = perf_options.ncvxPruningCount + 1;
            if options.displayOn ==1
               disp('region reduced by nonconvexity...')           
            end
            continue;
        elseif flag == 2
            Node.xl = xlj;
            Node.xu = xuj;
            %Node.Ih = f(hessianinit(infsup(Node.xl,Node.xu)));%get interval hessian
            Node.If = f(infsup(Node.xl,Node.xu));
            %Node.updateMonotonySet(dimj);           
            Node.mntIds(end+1) = dimj;
            BNodes{curId} = Node;
            perf_options.ncvxPruningCount = perf_options.ncvxPruningCount + 1;
            if options.displayOn ==1
               disp('region reduced by nonconvexity...')           
            end
            continue;
        elseif flag == 1
            BNodes(curId) = [];
            perf_options.ncvxPruningCount = perf_options.ncvxPruningCount + 1;
            if options.displayOn ==1
               disp('region pruned by nonconvexity...')           
            end            
            continue;
        end
    end
%% Calculate Upper Bound
   x0i = rand(nx,1).*(Node.xu-Node.xl)+Node.xl;%set inital point randomly
   [Node.xopt,Node.ub] =  fmincon(f,x0i,[],[],[],[],Node.xl,Node.xu,[],options.ubSolverOptions);%calculate upper bound
   if ~isempty(Node.elites)%verify if there is a previous solution
       xe = Node.elites(2:nx+1)';
       fe = Node.elites(1);       
       if options.mntprune == 1
           if watchflag == 1
              if fe < Node.If.inf
                  error('Mononoticity violation error: Non-sensical..')
              end
           end           
       end
       
       if fe < Node.ub %update upper bound if existing one is better
           Node.ub = fe;
           Node.xopt = xe;
       end
   else
       Node.elites = [Node.ub,Node.xopt'];%save best solution so far
   end
   perf_options.optIters(perf_options.iter,:) = [Node.ub,Node.xopt'];%store optiumum progression
%% Update Problem Upper bound
    if Node.ub < fopt
        fopt = Node.ub;
        xopt = Node.xopt;
    end    
%%  Record global optimum 
    if ~isempty(options.gVal)
        if optFound == 0
            if abs(Node.ub-options.gVal) <= options.eps
                optFound = 1;
                tOptimum = clock();
                perf_options.teps_opt = etime(tOptimum,tStart);%record the time at which the optimum has been found
                perf_options.gOptIter = perf_options.iter;%record iteration at which the optimum has been found
                break;
            end                    
        end
    end
  
%% Alpha Calculation
   if options.alpha_filtering == 0
       [lbd] = class_eig_bounds(Node.Ih.hx);
   else
       [lbd,rho] = rohn_eig_bounds(Node.Ih.hx);
       [lbd0] = class_eig_bounds(Node.Ih.hx);%scaled gershom            
       [lbd] = hladik_eig_filtering(Node.Ih.hx,max(lbd,lbd0),rho);%filter bounds
   end
   Node.alpha = max(0,-0.5*lbd);
   func_i = @(x)obj_func(f,Node.xl,Node.xu,Node.alpha,x);
%% Calculate Lower Bound
    if ~isempty(find(Node.alpha == Inf,1))
        If = f(infsup(Node.xl,Node.xu));
        Node.lb = If.inf;        
    else
        [lb_xopti,Node.lb] = fmincon(func_i,Node.xopt,[],[],[],[],Node.xl,Node.xu,[],options.lbSolverOptions);%calculate lower bound
    end
%% Branching Decision
   if Node.lb > Node.If.inf
     perf_options.intVsCvxCount(perf_options.iter,:) = [0 1 Node.lb-Node.If.inf];
   elseif Node.lb < Node.If.inf
     perf_options.intVsCvxCount(perf_options.iter,:) = [1 0 Node.lb-Node.If.inf];
   else
        if ~isempty(find(Node.alpha == Inf,1))
            if Node.If.inf ~= -Inf
                perf_options.intVsCvxCount(perf_options.iter,:) = [1 0 -inf-Node.If.inf];
            end
        end
   end
   
   if options.intprune == 1
       LBk = max(Node.lb,Node.If.inf);
   else
       LBk = Node.lb;
   end
   
   if LBk > Node.ub+lbubGvEps
       error('violation: lower bound above upper bound...')
   end
%      Node.lb
%      Node.Ih.hx
%      LBk
%      fopt
%    Node.lb
%    LBk
   if Node.ub - LBk > options.eps
       perf_options.iter = perf_options.iter + 1;%newer iteration
 %% Branching Strategy       
       if strcmp(options.branching,'nearSq') == 1
           [Node1,Node2] = Node.branchByNearSquare(f);
       else
           [Node1,Node2] = Node.branchByNearSquare(f);
       end       
       BNodes(curId) = [];%delete node
       BNodes{end+1} = Node1;%add node 1
       BNodes{end+1} = Node2;%add node 2
       if options.displayOn == 1
           disp('branching occured...');
       end
 %% Pruning by Characterisation
   else
       BNodes(curId) = [];%prune region
       if options.displayOn == 1
           disp('region pruned by characterisation...');
       end
   end
   if options.displayOn == 1
       fprintf('size of map: %d...\n',length(BNodes));
   end
end
perf_options.xopt = xopt;%store global optimum
perf_options.fopt = fopt;
tStop = clock();%capture overall runtime
perf_options.teps = etime(tStop,tStart);%measure algorithm running time
end
%% Search Strategies
function [id] = searchStrategy(BNodesIn,option)
   if strcmp(option,'bfs') == 1%default search strategy
       id = searchByAllBounds(BNodesIn);
   else
       id = searchByAllBounds(BNodesIn);
   end
end

function [id] = searchByAllBounds(BNodesIn)%search strategy by combination of lower bound and interval analysis
    minInf = Inf;
    id = -1;
    for k = 1:length(BNodesIn)
       BNode = BNodesIn{k};
       lbk = max(BNode.If.inf,BNode.lb);%find best bound
       if lbk < minInf
           minInf = lbk;
           id = k;
       end
    end 
end
%% Dependencies
function y = obj_func(f,xl,xu,alpha,x)
dim = length(xl);
y = f(x);
for i=1:dim
    y = y + alpha(i)*(x(i)-xl(i)).*(x(i)-xu(i));
end
end

