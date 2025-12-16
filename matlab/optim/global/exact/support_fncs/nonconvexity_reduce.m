function [xlj,xuj,xlk,xuk,flag,dimi] = nonconvexity_reduce(If,xli,xui,xl,xu,dimPool)
%monotony_reduce Summary of this function goes here
dimi = [];
flag = -1;
xlj = xli;
xuj = xui;
xlk = xli;
xuk = xui;
%flag = 1 prune
%flag = 2 reduce (take xlj)
%flag = 3 reduce (take xlk)
%flag = 4 %reduce and add
%% Monotonicity test
nx = length(xli);
for i=1:nx
    if isempty(find(dimPool==i,1))
        if If.hx(i,i).sup < 0 %non-convexity 
            dimi = i;        
            xuj(i) = xli(i);
            xlk(i) = xui(i);
            %--
            if xli(i) > xl(i) && xui(i) < xu(i)
               flag = 1;
               return;
            end        
            %--        
            if xli(i) == xl(i) && xui(i) < xu(i)            
                flag = 2;            
                return;
            end
            %--
            if xli(i) > xl(i) && xui(i) == xu(i)
                flag = 3;
                return;
            end
            %--
            if xli(i) == xl(i) && xui(i) == xu(i)
                flag = 4;
                return;
            end
        end
    end
end
end

