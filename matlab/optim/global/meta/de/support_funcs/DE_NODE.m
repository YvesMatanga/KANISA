classdef DE_NODE
    %PSO_NODE Summary of this class goes here
    %   Detailed explanation goes here    
    properties
        id %Node id (particle)
        xk %coordinate    
        ub %best value
        xlk %bound restriction
        xuk %bound restriction
        xl %absolute space bounds
        xu %absolute space bounds
        pop_id%id of the swarm it belongs to
        pop_sIter %swarm start iter
        collision%allow particle to move in subspace of the problem space
        gDist %distance from the swarm best        
        cx %center of particle
        geps %g improvement epsilon
        gStallIter %g improvement iteration control
        gMaxStallIter %g improvement max sall iter
        halt %stop node from being processed
        fArr %array of previous values
        fEvs %function evaluation history
        gIter %minimum iteration at which optimum found
    end
    
    methods
        function obj = DE_NODE()
            %PSO_NODE Construct an instance of this class
            %   Detailed explanation goes here
            obj.id = -1;
            obj.pop_id = -1;
            obj.ub = Inf;        
            obj.xk = [];
            obj.xlk = [];
            obj.xl = [];
            obj.xu = [];
            obj.xuk = [];
            obj.collision = 0;
            obj.gDist = [];
            obj.cx = [];
            obj.gMaxStallIter = 3;
            obj.gStallIter = 0;%iteration control of stall iter
            obj.halt = 0;
            obj.geps = 0.001;     
            obj.fArr = Inf*ones(obj.gMaxStallIter,1);
            obj.fEvs = [];%function evaluation history of a particle
            obj.gIter = -1;%mimimum iteration count at which global optimum found
        end
        
        function d = dist(obj,Node)
            %METHOD1 evaluate the distance between two nodes
            d = norm(obj.xk-Node.xk);          
        end
            
        function [obj] = xBound(obj)
            if obj.collision == 1
                obj.xk = min(obj.xk,obj.xuk);
                obj.xk = max(obj.xk,obj.xlk);
            end
            obj.xk = min(obj.xk,obj.xu);
            obj.xk = max(obj.xk,obj.xl);
        end
    end
end

