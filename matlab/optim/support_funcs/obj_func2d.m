function [y] = obj_func2d(obj_func,x1,x2)
    y = zeros(size(x1,1),size(x2,1));    
    for x1i=1:size(x1,1)
        for x2i=1:size(x1,2)
            y(x1i,x2i) = obj_func([x1(x1i,x2i);x2(x1i,x2i)]);
        end
    end
end