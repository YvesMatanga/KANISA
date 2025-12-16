function [Xks] = halton_sequence_gen(xu,xl,n)
%generate a sequence of n numbers evenly distributed in a n-dimensional
%space
    nx = length(xu);
    Xks = zeros(nx,n);
    
    
    p = get_nPrime(max(n,nx));%[2 3 5 7 11 13 17 19 23 29];%primes(nx);
    for x=1:nx
        rp = ceil(log10(n)/log10(p(x)));        
        for i=1:n
           digits = udec2base(i,p(x),rp);
           val = 0;
           for r=1:rp
               dg = digits(rp-r+1);
               if double(dg) >= 65
                    %error('letter found...')
                   cv = double(dg)-55;                            
               else
                   cv = str2double(dg);
               end
               
               val = val + cv/p(x)^r;
           end           
           Xks(x,i) = val*(xu(x)-xl(x))+xl(x);
        end
    end     
end


