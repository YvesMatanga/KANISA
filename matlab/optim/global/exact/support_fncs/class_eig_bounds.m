function [lbd] = class_eig_bounds(H)
%method to compute elower eigen bounds in
%a aBB convex relaxation
dim = size(H,2);%get dimension of vector
%lbd_min = +inf;
lbd = zeros(dim,1);
for i=1:dim
    ni = 1:dim;
    ni(ni==i) = [];
    sum_i = 0;
    for ii=ni
        sum_i = sum_i+max(abs(H.inf(i,ii)),abs(H.sup(i,ii)));
    end
    aii_l = H.inf(i,i);
    obj = aii_l - sum_i;
    lbd(i) = obj;
end
end