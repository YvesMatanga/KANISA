classdef BNODE
    %BNODE Summary of this class goes here
    %   Defines a single node of the branch and bound process    
    properties
        id %node id for identification
        xl %left bound
        xu %right bound
        lb %lower bound
        If %interval analysis: bound of interval analysis
        ub %current upper bound
        xopt %current optimum coordinations 
        alpha %alpha of node
        elites %get elite nodes from previous rounds: [f x]
        Ih %get interval hessian
        mntIds % store Ids of monotonous dimensions
    end
    
    methods
        function obj = BNODE()
            %BNODE Construct an instance of this class
            %   Detailed explanation goes here
            obj.id = 0;
            obj.xl = -Inf;%unbounded
            obj.xu = +Inf;%unbounded
            obj.lb = -Inf;%unbounded
            obj.If = [];
            obj.ub = Inf;            
            obj.xopt = 0;
            obj.alpha = Inf;
            obj.elites = [];
            obj.Ih = [];
            obj.mntIds = [];
        end
        
        %branch by near square
        function [Node1,Node2,itime] = branchByNearSquare(obj,func)
            %return also interval analysis total time
%%          branch by near square            
            hiScore = 0;
            nx = length(obj.xl);
            scorei = -1;%find the dimension that divides the plane the largest    
            for j=1:nx
                score = obj.xu(j) - obj.xl(j);
                if score > hiScore
                    scorei = j;
                    hiScore = score;
                end                
            end            
            Node1 = BNODE();
            Node2 = BNODE();
            Node1.xl = obj.xl;
            Node2.xl = obj.xl;
            Node1.xu = obj.xu;
            Node2.xu = obj.xu;            
            midx = 0.5*(obj.xu(scorei)+obj.xl(scorei));    
            Node1.xu(scorei) = midx;
            Node2.xl(scorei) = midx;
%%          store values accordingly
            Node1.id = obj.id + 1;%set nodes ids
            Node2.id = obj.id + 2;
            
            Node1.lb = obj.lb;%set nodes Lower bound
            Node2.lb = obj.lb;            
            %mn = 11;
            %set nodes intervals
            tIn = clock();
            Node1.If = func(infsup(Node1.xl,Node1.xu));%int2_mincing(func,Node1.xl,Node1.xu,mn);%func(infsup(Node1.xl,Node1.xu));
            Node2.If = func(infsup(Node2.xl,Node2.xu));%int2_mincing(func,Node2.xl,Node2.xu,mn);%func(infsup(Node2.xl,Node2.xu));                
            tOut = clock();
            itime = etime(tOut,tIn);
            %set nodes alpha
            
            Node1.alpha = obj.alpha;
            Node2.alpha = obj.alpha; 
            
            %set mononotonous dimensions
            Node1.mntIds = obj.mntIds;
            Node2.mntIds = obj.mntIds;
            
            %save up elite value in one of the nodes
            if ~isempty(obj.elites)
                optx = obj.elites(2:end)';                
                if inBounds(Node1.xl,Node1.xu,optx) == 1%check if elite is in Node 1
                    Node1.elites = obj.elites;
                else
                    Node2.elites = obj.elites;
                end
            end
        end
        
        function [bool] = HasNewMonotony(obj,dims) %verify if the dimens contains new monotony           
           if isempty(obj.mntIds) && ~isempty(dims)%if new variable found
               bool = 1;
               return;
           end
           
           if isempty(dims)
               bool = 0;
               return;
           end
           
%            obj.mntIds = sort(obj.mntIds);
%            dims =  sort(dims);
             no = length(obj.mntIds);%existing dimensions
             ni = length(dims);%new dimensions
             nd = min(no,ni);
%                       
%            for i=1:nd
%                if obj.mntIds(i) ~= dims(i)
%                    error('violation: Monotonicity logic error...')
%                end
%            end           
           if ni == nd
               bool = 0;
               return;
           elseif ni < no
               error('violation: Monotonicity logic error...')
           else
               bool = 1;
               return;
           end           
        end  
        
%         function updateMonotonySet(obj,dims)
%             obj.mntIds = dims;
%         end
        
%         function obj = getMonotonyDims(obj)
%             ret   urn obj.mntIds;
%         end
    end
end


