function str= udec2base(val,base,minDigit)
%Universal decimal to any base converter: from 2 to upper bound
   if base <= 36
       str=dec2base(val,base,minDigit);
   else
       str='';
       quotient = idivide(int16(val),int16(base));
       %---
       ri = rem(val,base);
       if ri < 10
           ri = num2str(ri);
       else
           ri= sprintf('%c',(ri-10)+65);
       end       
       str = [ri,str];
       %---
       val = quotient;
       while val ~= 0
           quotient = idivide(int16(val),int16(base));
           %---
           ri = rem(val,base);
           if ri < 10
               ri = num2str(ri);
           else
               ri= sprintf('%c',(ri-10)+65);
           end       
           str = [ri,str];
           %---         
           val = quotient;
       end
       
       if length(str) < minDigit
           gap = minDigit-length(str);
           for i=1:gap
               str=['0',str];
           end           
       end
   end
end