function [xopt,fopt,options] = DE_SOLVE(f,xl,xu,options)
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
else
    maxIter = options.maxIter;
    N = options.popSize;    
    F = options.F;
    cross_over_option = options.cross_over_option;
    maxStallIter = options.maxStallIter;
    geps = options.geps;
    EnStopHeurestic = options.EnStopHeurestic;
    EnPostOptimisation = options.EnPostOptimisation; 
    cr = options.cr;
    post_opts = options.post_opts;   
    displayOn = options.displayOn;
end

%%
nx = length(xl);
fopt = +Inf;%set best upper bound
xopt = ones(nx,1)*Inf;
%% Initialisation
[DNodes,gNode] = create_pop(f,xu,xl,N);%Initialisation DE evolution
iter = 0;
%%
if displayOn == 1
    global figureOn 
    figureOn = 0;
    displayPopulation(f,DNodes,gNode,xl,xu,iter);
end
%%
while iter < maxIter
    for j=1:N
        Node = DNodes{j};
%% Mutation
        VNode = create_mutant(j,N,F,DNodes);
%% Crossover      
        XNode = cross_over(Node,VNode,cr,cross_over_option);        
%% Next generation selection        
        Node.ub = f(Node.xk);
        XNode.ub = f(XNode.xk);        
        if XNode.ub <= Node.ub
            Node = XNode;%next generation update            
        end      
%% Optimum update        
        if Node.ub < gNode.ub
            gNode = Node;            
        end
        Node.id = j;%update id
        DNodes{j} = Node;%move to next generation
    end    
    iter = iter + 1;
%% Display
   if displayOn == 1
       displayPopulation(f,DNodes,gNode,xl,xu,iter);
   end
%%    
end


if EnPostOptimisation == 1
    [gNode.xk,gNode.ub] = fmincon(f,gNode.xk,[],[],[],[],xl,xu,[],post_opts); 
    %[gNode.xk,gNode.ub] = patternsearch(f,gNode.pk,[],[],[],[],xl,xu);
end
xopt = gNode.xk;
fopt = gNode.ub;

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


function [DNodes,bNode] = create_pop(f,xu,xl,N)
      bNode = DE_NODE();
      DNodes = cell(N,1);
      nx = length(xl);
      for i=1:N
          Node = DE_NODE();
          Node.id = i;
          Node.pop_id = 1;
          Node.xk = (xu-xl).*rand(nx,1) + xl;
          Node.xu = xu;
          Node.xl = xl;        
          Node = Node.xBound();
          Node.ub = f(Node.xk);
          if  Node.ub < bNode.ub
              bNode = Node;
          end              
          DNodes{i} = Node;
      end
end

function VNode = create_mutant(nodei,n,Fsc,DNodes)    
    ri = randperm(n);
    ri(find(ri==nodei))=[];%remove node index
    %select three random indices different from index array
    r = ri(1);p = ri(2);q = ri(3);   
    VNode = DNodes{nodei};%create new Node
    Nodep = DNodes{p};
    Nodeq = DNodes{q};
    Noder = DNodes{r};
    
    VNode.xk = Nodep.xk + Fsc*(Nodeq.xk-Noder.xk);
    VNode.xl = Nodep.xl;
    VNode.xu = Nodep.xu;
    VNode = VNode.xBound();%bound to problem space
end

function displayPopulation(f,DNodes,gNode,xl,xu,iter,figureOn)
    N = length(DNodes);  
    nx = length(xl);    
    global figureOn
    if nx==1
        global xn yn
        if figureOn == 0            
            d = (max(xu)-min(xl))/1000;
            xn = xl:d:xu;
            yn = obj_func1d(f,xn);
            figureOn = 1;
        end
        
        figure(10)
        clf
        plot(xn,yn,'r')
        hold on
        for ii=1:N
            Node = DNodes{ii};
            plot(Node.xk,f(Node.xk),'ro','LineWidth',2,'MarkerFaceColor','r')
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
            figureOn = 1;
        end
        figure(10)
         clf
         hold on
         contour(x1,x2,y,'ShowText','on')
         for ii=1:N
            Node = DNodes{ii};
            plot(Node.xk(1),Node.xk(2),'ro','LineWidth',2,'MarkerFaceColor','r')
         end
         plot(gNode.xk(1),gNode.xk(2),'go','LineWidth',2,'MarkerFaceColor','g')
         title(sprintf('DE: f(x^*) = %.5f - Iter %d',gNode.ub,iter))        
         axis([xl(1) xu(1) xl(2) xu(2)])   
    end
end