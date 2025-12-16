%% List of test functions
  %This script contains a list of bounded multimodal
  %test functions for global optimisation
  %the following information must be stored:
  %
  %   - bounds: [xl,xu] 
  %   - function_name:
  %   - objective_func:
  %   - global optimum point
  %   - global optimum function value
function_list = [];
%% 5D functions configuration
%d51
structure_d51.bounds = [-100*ones(5,1) 100*ones(5,1)];
structure_d51.func_name = 'rastrigin5D ';
structure_d51.obj_func = @(x)rastrigin_func(x,5);
structure_d51.obj_gfunc = @(x)rastrigin_gfunc(x,5);
structure_d51.obj_value = 0;
structure_d51.obj_point = [0;0];
function_list{end+1} = structure_d51;
%d52
structure_d52.bounds =[-100*ones(5,1) 100*ones(5,1)];
structure_d52.func_name = 'expSchaff5D';
structure_d52.obj_func = @(x)exp_schaffer2_func(x,5);
structure_d52.obj_value = 0;
structure_d52.obj_point = [0;0];
function_list{end+1} = structure_d52;
%d53
structure_d53.bounds = [-100*ones(5,1), 100*ones(5,1)];
structure_d53.func_name = 'griewa5D';
structure_d53.obj_func = @(x)griewank_func(x,5);
structure_d53.obj_value = 0;
structure_d53.obj_point = [0;0;0];
function_list{end+1} = structure_d53;
%d54
structure_d54.bounds = [-100*ones(5,1) 100*ones(5,1)];
structure_d54.func_name = 'ackley5D';
structure_d54.obj_func = @(x)ackleynD_func(x,5);
structure_d54.obj_value = 0;
structure_d54.obj_point = [0;0];
function_list{end+1} = structure_d54;
%d55
structure_d55.bounds = [-100*ones(5,1) 100*ones(5,1)];
structure_d55.func_name = 'weiers5D';%
structure_d55.obj_func = @(x)weierstrass_func(x,5);
structure_d55.obj_value = 0;%
structure_d55.obj_point = [0.1;0.1];%
function_list{end+1} = structure_d55;
%% 10D functions configuration
%d101
structure_d101.bounds = [-100*ones(10,1) 100*ones(10,1)];
structure_d101.func_name = 'rastrigin10D ';
structure_d101.obj_func = @(x)rastrigin_func(x,10);
structure_d101.obj_gfunc = @(x)rastrigin_gfunc(x,10);
structure_d101.obj_value = 0;
structure_d101.obj_point = [0;0];
function_list{end+1} = structure_d101;
%d102
structure_d102.bounds =[-100*ones(10,1) 100*ones(10,1)];
structure_d102.func_name = 'expSchaff10D';
structure_d102.obj_func = @(x)exp_schaffer2_func(x,10);
structure_d102.obj_value = 0;
structure_d102.obj_point = [0;0];
function_list{end+1} = structure_d102;
%d103
structure_d103.bounds = [-100*ones(10,1), 100*ones(10,1)];
structure_d103.func_name = 'griewa10D';
structure_d103.obj_func = @(x)griewank_func(x,10);
structure_d103.obj_value = 0;
structure_d103.obj_point = [0;0;0];
function_list{end+1} = structure_d103;
%d104
structure_d104.bounds = [-100*ones(10,1) 100*ones(10,1)];
structure_d104.func_name = 'ackley10D';
structure_d104.obj_func = @(x)ackleynD_func(x,10);
structure_d104.obj_value = 0;
structure_d104.obj_point = [0;0];
function_list{end+1} = structure_d104;
%d105
structure_d105.bounds = [-100*ones(10,1) 100*ones(10,1)];
structure_d105.func_name = 'weiers10D';%
structure_d105.obj_func = @(x)weierstrass_func(x,10);
structure_d105.obj_value = 0;%
structure_d105.obj_point = [0.1;0.1];%
function_list{end+1} = structure_d105;
%% 15D functions configuration
structure_d41.bounds = [-2.512*ones(20,1) 5.12*ones(20,1)];
structure_d41.func_name = 'rastrigin20d';
structure_d41.obj_func = @(x)rastrigin_func(x,20);
structure_d41.obj_value = 0;
structure_d41.obj_point = [0;0;0;0];
function_list{end+1} = structure_d41;

%% 20D functions configuration
structure_d41.bounds = [-2.512*ones(20,1) 5.12*ones(20,1)];
structure_d41.func_name = 'rastrigin20d';
structure_d41.obj_func = @(x)rastrigin_func(x,20);
structure_d41.obj_value = 0;
structure_d41.obj_point = [0;0;0;0];
function_list{end+1} = structure_d41;


%% 1D functions
function [y] = obj_funcd11(x)
y = sin(x) + sin(10*x/3); %[2.7 7.5]
end

function [y] = obj_funcd12(x)
   y = 0;
  for k = 1:6
      y = y -k.*sin((k+1).*x + k);
  end
end
%% 2D functions


function [y] = rosenbrock_gfunc(x) 
 y = [400*(x(1).^2-x(2)).*x(1)+2*(x(1)-1);200*(x(2)-x(1).^2)];
end
%-----

function [y] = rastrigin_func(x,d)
%test func: rastrigin
 y = 10*d;
 for k=1:d
     y = y + x(k).^2 - 10*cos(2*pi*x(k));
 end
%[-5.12 5.12]
end

function [y] = rastrigin_gfunc(x,d)
%test func: rastrigin
 y = zeros(d,1);
 for k=1:d
     y(k) = 2*x(k) + 20*pi*sin(2*pi*x(k));
 end
%[-5.12 5.12]
end
%-------


function [y] = dejong5_func(x)
%de jong 5th
x1 = x(1);
x2 = x(2);
sum = 0;

A = zeros(2, 25);
a = [-32, -16, 0, 16, 32];
A(1, :) = repmat(a, 1, 5);
ar = repmat(a, 5, 1);
ar = ar(:)';
A(2, :) = ar;

for ii = 1:25
    a1i = A(1, ii);
    a2i = A(2, ii);
    term1 = ii;
    term2 = (x1 - a1i)^6;
    term3 = (x2 - a2i)^6;
    new = 1 / (term1+term2+term3);
    sum = sum + new;
end
y = 1 / (0.002 + sum);%[-65.536 +65.536]
end

function [y] = shekel_func(x)

%j i_1 i_2
c_a = ...
[0.806   9.681   0.66;...
 0.517   9.400   2.04;...
 0.100   8.025   9.15;...
 0.908   2.196   0.415;...
 0.965   8.074   8.77;...
 0.669   7.650   5.658;...
 0.524   1.256   3.60;...
 0.902   8.314   2.26;...
 0.531   0.226   8.85;...
 0.876   7.305   2.22;...
 0.462   0.652   7.027;...
 0.491   2.699   3.516;...
 0.463   8.327   3.897;...
 0.714   2.132   7.006;...
 0.352   4.707   5.57;...
 0.869   8.304   7.559;...
 0.813   8.632   4.40;...
 0.811   4.887   9.112;...
 0.828   2.440   6.686;...
 0.964   6.306   8.58;...
 0.789   0.652   2.34;...
 0.360   5.558   1.272;...
 0.369   3.352   7.54;...
 0.992   8.798   0.88;...
 0.332   1.460   8.05;...
 0.817   0.432   8.64;...
 0.632   0.679   2.800;...
 0.883   4.263   1.07;...
 0.608   9.496   4.83;...
 0.326   4.138   2.562];

y = 0;
for j=1:30
    cj = c_a(j,1);
    for i=1:2
        cj = cj + (x(i)-c_a(j,i+1)).^2;
    end
    y = y + 1/cj;
end
y = -y;
end


function [y] = levy13_func(x)
x1 = x(1);
x2 = x(2);

term1 = (sin(3*pi*x1))^2;
term2 = (x1-1)^2 * (1+(sin(3*pi*x2))^2);
term3 = (x2-1)^2 * (1+(sin(2*pi*x2))^2);

y = term1 + term2 + term3;
end

function [y] = schaffer2_func(x)
x1 = x(1);
x2 = x(2);

fact1 = (sin(x1^2-x2^2))^2 - 0.5;
fact2 = (1 + 0.001*(x1^2+x2^2))^2;

y = 0.5 + fact1/fact2;
end



function [y] = exp_schaffer2_func(x,n)  
y = schaffer2_func([x(n);x(1)]);
for i=1:n-1
   y = y + schaffer2_func([x(i);x(i+1)]);
end 
end


function [y] = shubert_func(x,n)
% x1 = x(1);
% x2 = x(2);
% sum1 = 0;
% sum2 = 0;
y = 1;
for j = 1:n
    sum1 = 0;
    for ii = 1:5
        new1 = ii * cos((ii+1)*x(j)+ii);
        %new2 = ii * cos((ii+1)*x2+ii);
        sum1 = sum1 + new1;
        %sum2 = sum2 + new2;
    end
    y = y*sum1;
end
%y = sum1 * sum2;
end


function [y] = michal_func(x, m)
if (nargin == 1)
    m = 10;
end

d = length(x);
sum = 0;

for ii = 1:d
	xi = x(ii);
	new = sin(xi) * (sin(ii*xi^2/pi))^(2*m);
	sum  = sum + new;
end
y = -sum;
end

function [y] = langer_func(x, m, c, A)
d = length(x);

if (nargin < 2)
    m = 5;
end

if (nargin < 3)
    if (m == 5)
        c = [1, 2, 5, 2, 3];
    else
        error('Value of the m-dimensional vector c is required.')
    end
end

if (nargin < 4)
    if (m==5 && d==2)
        A = [3, 5; 5, 2; 2, 1; 1, 4; 7, 9];
    else
        error('Value of the (mxd)-dimensional matrix A is required.')
    end
end

outer = 0;
for ii = 1:m
    inner = 0;
    for jj = 1:d
        xj = x(jj);
        Aij = A(ii,jj);
        inner = inner + (xj-Aij)^2;
    end
    new = c(ii) * exp(-inner/pi) * cos(pi*inner);
    outer = outer + new;
end

y = outer;

end


function [y] = rosenbrock_func(x)
%test func : rosenbrock
y = 100*(x(2)-x(1).^2).^2 + (1-x(1)).^2;%[-2 2]
end

function [y] = griewank_func(x,d)
sum = 0;
prod = 1;

for ii = 1:d
	xi = x(ii);
	sum = sum + xi^2/4000;
	prod = prod * cos(xi/sqrt(ii));
end

y = sum - prod + 1;%[-600 600]
end


function [y] = drop_func(xx)
x1 = xx(1);
x2 = xx(2);

frac1 = 1 + cos(12*sqrt(x1^2+x2^2));
frac2 = 0.5*(x1^2+x2^2) + 2;

y = -frac1/frac2;

end

function [y] = ackley_func(xx, a, b, c)
d = length(xx);

if (nargin < 4)
    c = 2*pi;
end
if (nargin < 3)
    b = 0.2;
end
if (nargin < 2)
    a = 20;
end

sum1 = 0;
sum2 = 0;
for ii = 1:d
	xi = xx(ii);
	sum1 = sum1 + xi^2;
	sum2 = sum2 + cos(c*xi);
end

term1 = -a * exp(-b*sqrt(sum1/d));
term2 = -exp(sum2/d);

y = term1 + term2 + a + exp(1);

end

function [y] = braninmodif_func(xx, a, b, c, r, s, t)
x1 = xx(1);
x2 = xx(2);

if (nargin < 7)
    t = 1 / (8*pi);
end
if (nargin < 6)
    s = 10;
end
if (nargin < 5)
    r = 6;
end
if (nargin < 4)
    c = 5/pi;
end
if (nargin < 3)
    b = 5.1 / (4*pi^2);
end
if (nargin < 2)
    a = 1;
end

term1 = a * (x2 - b*x1^2 + c*x1 - r)^2;
term2 = s*(1-t)*cos(x1);

y = term1 + term2 + s + 5*x1;

end

function [y] = permdb_func(xx, b)
if (nargin == 1)
    b = 0.5;
end

d = length(xx);
outer = 0;

for ii = 1:d
	inner = 0;
	for jj = 1:d
		xj = xx(jj);
        inner = inner + (jj^ii+b)*((xj/jj)^ii-1);
    end
	outer = outer + inner^2;
end

y = outer;

end

function [y] = stybtang_func(xx)
d = length(xx);
sum = 0;
for ii = 1:d
	xi = xx(ii);
	new = xi^4 - 16*xi^2 + 5*xi;
	sum = sum + new;
end

y = sum/2;

end

function y = easom_func(x)
% 
% Easom function 
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 2.
% 
y = -cos(x(1))*cos(x(2))*exp(-(x(1)-pi)^2-(x(2)-pi)^2);
end

function y = beale_func(x)
% 
% Beale function.
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 2.
% 
y = (1.5-x(1)*(1-x(2)))^2+(2.25-x(1)*(1-x(2)^2))^2+(2.625-x(1)*(1-x(2)^3))^2;
end

function scores = adjiman_func(x)
    
    %n = size(x, 2);
    %assert(n == 2, 'Adjiman function is only defined on a 2D space.')
    X = x(1);
    Y = x(2);
    
    scores = (cos(X) .* sin(Y)) - (X ./ ((Y .^ 2) + 1));
end

function scores = bird_func(x)    
    %n = size(x, 2);
    %assert(n == 2, 'Bird function is only defined on a 2D space.')
    X = x(1);
    Y = x(2);
    
    scores = sin(X) .* exp((1 - cos(Y)).^2) + ... 
        cos(Y) .* exp((1 - sin(X)) .^ 2) + ...
        (X - Y) .^ 2;
end

function scores = keane_func(x)
    %n = size(x, 2);
    %assert(n == 2, 'Keane function is defined only on a 2D space.')
    X = x(1);
    Y = x(2);
    
    numeratorcomp = (sin(X - Y) .^ 2) .* (sin(X + Y) .^ 2); 
    denominatorcomp = sqrt(X .^2 + Y .^2);
    scores =-numeratorcomp ./ denominatorcomp;
end

function y = h1_func(x)
   y = sin(x(1) - x(2)/8).^2 + sin(x(2) + x(1)/8)^2;
   y = -y./(sqrt((x(1)-8.6998).^2 + (x(2) - 6.7665 ).^2)+1);
end

function y = periodic_func(x)
    sin2x = sin(x') .^ 2;
    sumx2 = sum(x' .^2, 2);
    y = 1 + sum(sin2x, 2) -0.1 * exp(-sumx2);    
end

function y = logn_func(x,n)
y = 0;
  for i=1:n
      
      if i==1
          xj1 = x(1);
      else
          xj1 = x(i-1);
      end
      
      if i == n
          xjp = x(1);
      else
       
          xjp = x(i+1);
      end
      y = y + (log10(xj1.*x(i).^2)/log10(2) - log10(xjp.^5-5)/log10(3)).^2;
  end
end

function y = d3_func(x,n)
y = (1/n)*(x(1)+1).^2;
  for i=1:n
      
      if i==1
          xj1 = x(1);
      else
          xj1 = x(i-1);
      end
      
      if i == n
          xjp = x(1);
      else
       
          xjp = x(i+1);
      end
      y = y + (x(i) - 2*(xj1+xjp)-xj1*xjp-2).^2;
  end
end

function y = f10_func(x,n)
  y1=0;
  y2 =1;
  for i=1:n
      y1 = y1  + (x(i)-0.4).^2;
      y2 = y2*cos(x(i)-0.4);
  end
  y = -2*exp(-20*sqrt(n)*sqrt(y1)) + y2;
end

function y = weierstrass_func(x,n)
    y =0;
    for i=1:n
       yi =0;
       for j=1:21
           yi = yi + (0.5^(j-1))*cos(2*(x(i)+0.5)*pi*3^(j-1));
       end
       y = y + yi;
    end
    
    for j=1:21
        y = y - n*cos(pi*3^(j-1))*0.5^(j-1);
    end
end



function y = ackley_mod(x,d)
  y = 0;
  for i=1:d-1
      y = y + exp(-0.2)*sqrt(x(i).^2 + x(i+1).^2)+3*(cos(2*x(i))+sin(2*x(i+1)));
  end
end

% function y = biggs_func(x)
%    y = 0;
%    for i=1:10
%        ti = 0.1*i;
%        yi = exp(-ti) - 5*exp(10*ti);
%        y = y + (exp(-ti*x(1)) - 5*exp(-ti*x(2))-yi)^2;
%    end
% end

function y = camel6_func(x)
 y = (4-2.1*x(1)^2+(x(1)^4)/3)*x(1)^2+x(1)*x(2)+...
     (4*x(2)^2-4)*x(2)^2;
end

function y = chi_func(x)
   y = x(1)^2-12*x(1)+11+10*cos(pi*x(1)/2)+...
       8*sin(5*pi*x(1)/2)-((1/5)^0.5)*exp(-0.5*(x(2)-0.5)^2);
end


function y= deckker_func(x)
  y = (10^5)*x(1)^2+x(2)^2-(x(1)^2+x(2)^2)^2+(10^(-5))*(x(1)^2+x(2)^2)^4;
end


function y=giunta_func(x)
  y = 0.6;
  for i=1:2
      y = y + sin((16/15)*x(i)-1)+...
          sin((16/15)*x(i)-1).^2+...
          (1/50)*sin(4*((16/15)*x(i)-1));
  end
  
end

function y = hosaki_func(x)
 y = (1-8*x(1)+7*x(1).^2-(7/3).*x(1).^3+(1/4).*x(1).^4).*(x(2).^2).*exp(-x(2));
end

function y = trefethen_func(x)
  y = exp(sin(50*x(1)))+sin(60*exp(x(2)))+...
      sin(70*sin(x(1))) + sin(80*sin(x(2)))-...
      sin(10*(x(1)+x(2))) + 0.25*(x(1).^2+x(2).^2);
end

function y = sineEnv_funcnd(x,n)
  y = 0;
  for i=1:n-1
      y = y + 0.5 + (sin(sqrt(x(i+1)^2+x(i)^2)-0.5).^2)/((0.001*(x(i+1)^2+x(i)^2)+1)^2); 
  end
  y = -y;
end

function y = levy_func(x,n)
% 
% Levy function 
% Matlab Code by A. Hedar (Nov. 23, 2005).
% The number of variables n should be adjusted below.
% The default value of n =2.
% 
for i = 1:n; z(i) = 1+(x(i)-1)/4; end
s = sin(pi*z(1))^2;
for i = 1:n-1
    s = s+(z(i)-1)^2*(1+10*(sin(pi*z(i)+1))^2);
end 
y = s+(z(n)-1)^2*(1+(sin(2*pi*z(n)))^2);
end

function y = solomon_func(x,n)  
  sum1 = 0;
  for i=1:n
      sum1 = sum1+x(i).^2;
  end
  y = 1 - cos(2*pi*sqrt(sum1))+0.1*sqrt(sum1);
end

function scores = ackleynD_func(x,n)
    %n = size(x, 2);
    ninverse = 1 / n;
    %sum1 = sum(x .^ 2, 2);
    sum1 = 0;
    sum2 = 0;
    for i=1:n
        sum1 = sum1+x(i).^2;
        sum2 = sum2+cos(2*pi*x(i));
    end
    %sum2 = sum(cos(2 * pi * x), 2);    
    scores = 20 + exp(1) - (20 * exp(-0.2 * sqrt( ninverse * sum1))) - exp( ninverse * sum2);
end

function y = cosm_func(x,n)
  y = 0;
  for i=1:n
      y = y -0.1*cos(5*pi*x(i))-x(i).^2;
  end
end

function y = egg_crate_func(x)
 y = x(1).^2 + x(2).^2 + 25*(sin(x(1)).^2 + sin(x(2)).^2);
end

function y = withley_func(x,n)
  y = 0; 
  for i=1:n
      for j=1:n
          y = y + ((100*(x(i).^2-x(j)).^2 + (1-x(j)).^2).^2)/4000-...
              cos(100*(x(i).^2-x(j)).^2+(1-x(j)).^2)+1;         
      end
  end
end

function y = wavy_func(x,n)
  y = 0; 
  sum = 0;
  k = 10;
  for i=1:n
      sum = sum + cos(k*x(i)).*exp(-0.5*x(i).^2);
  end
  sum = (1/n)*sum;
  y = 1 - sum;
end

function y = deb_func(x,n)
  y = 0;
  for i=1:n
      y = y + sin(5*pi*x(i)).^6;
  end
  y = -y/n;
end

function y = corr_spring_func(x,n,a)
 k = 5;
 sum1 = 0;
 for i=1:n
     sum1 = sum1 + (x(i)-a).^2;
 end
 sum1 = k*sqrt(sum1);
 
 sum2 = 0;
 for i=1:n
     sum2 = sum2 + (x(i)-a).^2-cos(sum1);
 end
 y = 0.1*sum2;
end