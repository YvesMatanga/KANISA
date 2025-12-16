function [xlj,xuj,flag,dimi] = monotony_reduce(If,xli,xui)
%monotony_reduce Summary of this function goes here
dimi = [];
flag = -1;
xlj = xli;
xuj = xui;
%% Monotonicity test
nx = length(xli);
for i=1:nx
    if If.dx(i).inf > 0 %mononitically increasing        
        dimi(end+1) = i;
        xuj(i) = xli(i);
        flag = 1;
        continue;
    end
       
    if If.dx(i).sup < 0 %mononitically decreasing
        dimi(end+1) = i;
        xlj(i) = xui(i);
        flag = 1;
        continue;
    end         
end
end

