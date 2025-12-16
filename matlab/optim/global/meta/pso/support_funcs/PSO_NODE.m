classdef PSO_NODE
    %PSO_NODE Summary of this class goes here
    %   Detailed explanation goes here    
    properties
        id %Node id (particle)
        xk0 %initial particle location
        xk %coordinate
        xk1 %previous coordinates
        pk %best location
        vk %velocity of particle
        vmin %velocity of particle min
        vmax %velocity of particle max
        ub %best value
        xlk %bound restriction
        xuk %bound restriction
        xl %absolute space bounds
        xu %absolute space bounds
        swarm_id%id of the swarm it belongs to
        old_swarm_id%swarm_id before sub-clustering
        sub_swarm_id %subclusters within main cluster
        swarm_sIter %swarm start iter
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
        fprev
        r %node radius
        fik %radius 
        sck %successes
        frk %failures
        scMax %successes max
        frMax %failures max
    end
    
    methods
        function obj = PSO_NODE()
            %PSO_NODE Construct an instance of this class
            %   Detailed explanation goes here
            obj.id = -1;
            obj.swarm_id = -1;
            obj.sub_swarm_id = -1;
            obj.old_swarm_id = -1;
            obj.ub = Inf;
            obj.pk = [];
            obj.xk = [];
            obj.xk0 = [];%initial particle location
            obj.xk1 = [];%previous coordinate of the particle
            obj.xlk = [];
            obj.xl = [];
            obj.xu = [];
            obj.xuk = [];
            obj.vk = [];
            obj.swarm_sIter = 0;
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
            obj.fprev = Inf;
            obj.r = -1;%set node radius
            %guaranteed pso model data
            obj.fik = 0.1;
            obj.sck = 0;
            obj.frk = 0;
            obj.scMax = 15;%success max
            obj.frMax = 5;%failure max
        end
        
        function d = dist(obj,Node)
            %METHOD1 evaluate the distance between two nodes
            d = norm(obj.xk-Node.xk);          
        end
        
        function [obj] = vBound(obj) %boudn particle velocity
            obj.vk = min(obj.vk,obj.vmax);
            obj.vk = max(obj.vk,obj.vmin);
        end
            
        function [obj] = xBound(obj)
            if obj.collision == 1
                obj.xk = min(obj.xk,obj.xuk);
                obj.xk = max(obj.xk,obj.xlk);
            end
            obj.xk = min(obj.xk,obj.xu);
            obj.xk = max(obj.xk,obj.xl);
        end
        
        function [obj] = nextIter(obj) %move the particle to the next iteration            
            xkp = obj.xk + obj.vk;            
            if obj.collision == 1%if particle cannot exceed its set bounds
                obj.xk = min(xkp,obj.xuk);
                obj.xk = max(xkp,obj.xlk);
            else
                obj.xk = xkp;
            end
                obj.xk = min(obj.xk,obj.xu);%set absolute bounds
                obj.xk = max(obj.xk,obj.xl);            
        end
    end
end

