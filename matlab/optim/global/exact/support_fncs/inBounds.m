function y = inBounds(xl,xu,x)
  N = length(xl);
  y = 1;
  for i=1:N
      if x(i) < xl(i) || x(i) > xu(i) 
          y = 0;
          break;
      end        
  end
end