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
  
%  Citation 1: CEC 2019 --100Digit challenge 
%   Price, K.V., Awad, N.H., Ali, M.Z. and Suganthan, P.N., 2018.
%   Problem definitions and evaluation criteria for the 100-digit challenge 
%   special session and competition on single objective numerical optimization.
%   In Technical Report. Nanyang Technological University.
  
% Citation 2: CEC2017
%    Wu, G., Mallipeddi, R. and Suganthan, P.N., 2017. 
%    Problem definitions and evaluation criteria for the CEC 2017 competition 
%    on constrained real-parameter optimization. National University of Defense Technology, Changsha, Hunan, 
%    PR China and Kyungpook National University, Daegu, South Korea and Nanyang Technological University, Singapore, Technical Report.

function_list = [];
%% nD functions configuration
%dn1
structure_dn1.bounds = [-100 100];
structure_dn1.func_name = 'F4';
structure_dn1.M2 = [0.1 0; 0 0.1];%[-3.0374827286555761e-01   9.5275232181884006e-01;...
    %9.5275232181884018e-01   3.0374827286555761e-01];
structure_dn1.obj_func = @(x,n,M,o)(1+rastrigin_func(M*(x-o),n));
structure_dn1.obj_gfunc = [];
structure_dn1.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn1.obj_point = [0;0];
function_list{end+1} = structure_dn1;

%dn2
structure_dn2.bounds = [-100 100];
structure_dn2.func_name = 'F5';
structure_dn2.M2 = [3.7904585567822813e-01   9.2537788999584381e-01;...
  -9.2537788999584381e-01   3.7904585567822813e-01];
structure_dn2.obj_func = @(x,n,M,o)(1+griewank_func(M*(x-o),n));
structure_dn2.obj_gfunc = [];
structure_dn2.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn2.obj_point = [0;0];
function_list{end+1} = structure_dn2;

%dn3
structure_dn3.bounds = [-0.5 0.5];
structure_dn3.func_name = 'F6';
structure_dn3.M2 = [  -9.9987388703608149e-01   1.5881121602620563e-02;...
   1.5881121602620618e-02   9.9987388703608149e-01];
structure_dn3.obj_func = @(x,n,M,o)(1+weierstrass_func(M*(x-o),n));
structure_dn3.obj_gfunc = [];
structure_dn3.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn3.obj_point = [0;0];
function_list{end+1} = structure_dn3;

%dn4
structure_dn4.bounds = [-100 100];
structure_dn4.func_name = 'F8';
structure_dn4.M2 = [    -7.1254101872216313e-01  -7.0163045589425632e-01;...
  -7.0163045589425643e-01   7.1254101872216324e-01];
structure_dn4.obj_func = @(x,n,M,o)(1+exp_schaffer6_func(M*(x-o),n));
structure_dn4.obj_gfunc = [];
structure_dn4.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn4.obj_point = [0;0];
function_list{end+1} = structure_dn4;

%dn5
structure_dn5.bounds = [-100 100];
structure_dn5.func_name = 'F9';
structure_dn5.M2 = [  -4.8481534461057663e-01  -8.7461653404799478e-01;...
  -8.7461653404799522e-01   4.8481534461057729e-01];
structure_dn5.obj_func = @(x,n,M,o)(1+griewank_rosen_func(M*(x-o),n));
structure_dn5.obj_gfunc = [];
structure_dn5.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn5.obj_point = [0;0];
function_list{end+1} = structure_dn5;

%dn6
structure_dn6.bounds = [-100 100];
structure_dn6.func_name = 'F10';
structure_dn6.M2 = [5.8945940367175020e-01   8.0779800162103932e-01
  -8.0779800162103910e-01   5.8945940367175020e-01];
structure_dn6.obj_func = @(x,n,M,o)(1+ackleynD_func(x,n));
structure_dn6.obj_gfunc = [];
structure_dn6.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn6.obj_point = [0;0];
function_list{end+1} = structure_dn6;

%dn7
structure_dn7.bounds = [-100 100];
structure_dn7.func_name = 'F3_*';
structure_dn7.M2 = [   9.2623966950037206e-01   3.7693510667466584e-01;... 
    -3.7693510667466584e-01   9.2623966950037206e-01];
structure_dn7.obj_func = @(x,n,M,o)(1+rosen_func(x,n));
structure_dn7.obj_gfunc = [];
structure_dn7.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn7.obj_point = [0;0];
function_list{end+1} = structure_dn7;

%dn8
structure_dn8.bounds = [-100 100];
structure_dn8.func_name = 'F6_*';
structure_dn8.M2 = [  -8.2473135160583888e-01   5.6552471005112226e-01;...
  -5.6552471005112270e-01  -8.2473135160583877e-01];
structure_dn8.obj_func = @(x,n,M,o)(1+schaffer7_func(x,n));
structure_dn8.obj_gfunc = [];
structure_dn8.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn8.obj_point = [0;0];
function_list{end+1} = structure_dn8;

%dn9
structure_dn9.bounds = [-100 100];
structure_dn9.func_name = 'F9_*';
structure_dn9.M2 = [  -  -6.1202819054620450e-01  -7.9083594631044751e-01;...
  -7.9083594631042786e-01   6.1202819054622637e-01];
structure_dn9.obj_func = @(x,n,M,o)(1+levy_func(x,n));
structure_dn9.obj_gfunc = [];
structure_dn9.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},ones(1,10));
structure_dn9.obj_point = [0;0];
function_list{end+1} = structure_dn9;

%% hybrid test functions
%dn10
structure_dn10.funcs = {@zak_func,@rosen_func,@rastrigin_func};
structure_dn10.bounds = [-100 100];
structure_dn10.hybrid_p = [0.2 0.4 0.4];
structure_dn10.func_name = 'HF_1';
structure_dn10.M2 = [-6.1202819054620450e-01  -7.9083594631044751e-01;...
  -7.9083594631042786e-01   6.1202819054622637e-01];
structure_dn10.obj_func = @(x,S,nQ,Ms,Os)(1+hf_func(x,S,nQ,Ms,Os,structure_dn10.funcs));
structure_dn10.obj_gfunc = [];
structure_dn10.obj_value = 0;
structure_dn10.obj_point = [0;0];
function_list{end+1} = structure_dn10;
%dn11
structure_dn11.funcs = {@elliptic_func,@ackleynD_func,@schaffer7_func,@rastrigin_func};
structure_dn11.bounds = [-100 100];
structure_dn11.hybrid_p = [0.2 0.2 0.2 0.4];
structure_dn11.func_name = 'HF_4';
structure_dn11.M2 = [-6.1202819054620450e-01  -7.9083594631044751e-01;...
  -7.9083594631042786e-01   6.1202819054622637e-01];
structure_dn11.obj_func = @(x,S,nQ,Ms,Os)(1+hf_func(x,S,nQ,Ms,Os,structure_dn11.funcs));
structure_dn11.obj_gfunc = [];
structure_dn11.obj_value = 0;
structure_dn11.obj_point = [0;0];
function_list{end+1} = structure_dn11;

% %dn12
% structure_dn12.funcs = {@bent_cigar_func,@hgbat_func,@rastrigin_func,@rosen_func};
% structure_dn12.bounds = [-100 100];
% structure_dn12.hybrid_p = [0.2 0.2 0.3 0.3];
% structure_dn12.func_name = 'HF_5';
% structure_dn12.M2 = [-6.1202819054620450e-01  -7.9083594631044751e-01;...
%   -7.9083594631042786e-01   6.1202819054622637e-01];
% structure_dn12.obj_func = @(x,S,nQ,Ms,Os)(1+hf_func(x,S,nQ,Ms,Os,structure_dn12.funcs));
% structure_dn12.obj_gfunc = [];
% structure_dn12.obj_value = 0;
% structure_dn12.obj_point = [0;0];
% function_list{end+1} = structure_dn12;

% %dn13
% structure_dn13.funcs = {@elliptic_func,@ackleynD_func,@rastrigin_func,@hgbat_func,@discuss_func};
% structure_dn13.bounds = [-100 100];
% structure_dn13.hybrid_p = [0.2 0.2 0.2 0.2 0.2];
% structure_dn13.func_name = 'HF_8';
% structure_dn13.M2 = [-6.1202819054620450e-01  -7.9083594631044751e-01;...
%   -7.9083594631042786e-01   6.1202819054622637e-01];
% structure_dn13.obj_func = @(x,S,nQ,Ms,Os)(1+hf_func(x,S,nQ,Ms,Os,structure_dn13.funcs));
% structure_dn13.obj_gfunc = [];
% structure_dn13.obj_value = 0;
% structure_dn13.obj_point = [0;0];
% function_list{end+1} = structure_dn13;

%dn14
structure_dn14.funcs = {@bent_cigar_func,@rastrigin_func,@griewank_rosen_func,@weierstrass_func,@exp_schaffer6_func};
structure_dn14.bounds = [-0.5 0.5];
structure_dn14.hybrid_p = [0.2 0.2 0.2 0.2 0.2];
structure_dn14.func_name = 'HF_9';
structure_dn14.M2 = [-6.1202819054620450e-01  -7.9083594631044751e-01;...
  -7.9083594631042786e-01   6.1202819054622637e-01];
structure_dn14.obj_func = @(x,S,nQ,Ms,Os)(1+hf_func(x,S,nQ,Ms,Os,structure_dn14.funcs));
structure_dn14.obj_gfunc = [];
structure_dn14.obj_value = 1;
structure_dn14.obj_point = [0;0];
function_list{end+1} = structure_dn14;

%% Composition Functions
% %dn15
% structure_dn15.funcs = {@rosen_func,@elliptic_func,@rastrigin_func};
% structure_dn15.bounds = [-100 100];
% structure_dn15.sigma = [10 20 30];
% structure_dn15.lbd = [1 10^(-6) 1];
% structure_dn15.bias = [0 100 200];
% structure_dn15.func_name = 'CF_1';
% structure_dn15.obj_func = @(x,n,sigma,lbd,bias,Ms,Os)(comp_func(x,n,sigma,lbd,bias,Ms,Os,structure_dn15.funcs));
% structure_dn15.obj_gfunc = [];
% structure_dn15.obj_value = 1;
% structure_dn15.obj_point = [0;0];
% function_list{end+1} = structure_dn15;
% %dn16
% structure_dn16.funcs = {@ackleynD_func,@elliptic_func,@griewank_func,@rastrigin_func};
% structure_dn16.bounds = [-100 100];
% structure_dn16.sigma = [10 20 30 40];
% structure_dn16.lbd = [10 10^(-6) 10 1];
% structure_dn16.bias = [0 100 200 300];
% structure_dn16.func_name = 'CF_4';
% structure_dn16.obj_func = @(x,n,sigma,lbd,bias,Ms,Os)(comp_func(x,n,sigma,lbd,bias,Ms,Os,structure_dn16.funcs));
% structure_dn16.obj_gfunc = [];
% structure_dn16.obj_value = 1;
% structure_dn16.obj_point = [0;0];
% function_list{end+1} = structure_dn16;
% %dn17
% structure_dn17.funcs = {@rastrigin_func,@happy_cat_func,@ackleynD_func,@discuss_func,@rosen_func};
% structure_dn17.bounds = [0.1 100];
% structure_dn17.sigma = [10 20 30 40];
% structure_dn17.lbd = [1 10^(-6) 10 1];
% structure_dn17.bias = [0 100 200 300];
% structure_dn17.func_name = 'CF_5';
% structure_dn17.obj_func = @(x,n,sigma,lbd,bias,Os)(1+comp_func(x,n,sigma,lbd,bias,Os,structure_dn17.funcs));
% structure_dn17.obj_gfunc = [];
% structure_dn17.obj_value = 1;
% structure_dn17.obj_point = [0;0];
% function_list{end+1} = structure_dn17;
%dn18
% structure_dn18.funcs = {@ackleynD_func,@griewank_func,@discuss_func,@rosen_func,@happy_cat_func,@exp_schaffer6_func};
% structure_dn18.bounds = [0.1 100];
% structure_dn18.sigma = [10 20 30 40 50 60];
% structure_dn18.lbd = [10 10 10^(-6) 1 1 (5*10^(-4))];
% structure_dn18.bias = [0 100 200 300 400 500];
% structure_dn18.func_name = 'CF_8';
% structure_dn18.obj_func = @(x,n,sigma,lbd,bias,Os)(1+comp_func(x,n,sigma,lbd,bias,Os,structure_dn18.funcs));
% structure_dn18.obj_gfunc = [];
% structure_dn18.obj_value = 1;
% structure_dn18.obj_point = [0;0];
% function_list{end+1} = structure_dn18;
%% Classical Functions
structure_cn1.bounds = [-35 35];
structure_cn1.func_name = 'SF_4';
structure_cn1.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn1.obj_func = @(x,n,M,o)(5.5901+ackleym_func(M*(x-o),n));
structure_cn1.obj_gfunc = [];
structure_cn1.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,1,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn1.obj_point = [0;0];
structure_cn1.isClassical = 1;
structure_cn1.canShift = 1;
function_list{end+1} = structure_cn1;

structure_cn2.bounds = [0.1 100];
structure_cn2.func_name = 'SF_7';
structure_cn2.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn2.obj_func = @(x,n,M,o)(alpine2_func(M*(x-o),n));
structure_cn2.obj_gfunc = [];
structure_cn2.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-97.3793,-984.4808,-9.636915199279638e+03,-9.742687698984319e+04,NaN,NaN,NaN,NaN,NaN]);
structure_cn2.obj_point = [0;0];
structure_cn2.isClassical = 1;
structure_cn2.canShift = 0;
function_list{end+1} = structure_cn2;

structure_cn3.bounds = [-10 10];
structure_cn3.func_name = 'SF_{38}';
structure_cn3.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn3.obj_func = @(x,n,M,o)(cosm_func(M*(x-o),n));
structure_cn3.obj_gfunc = [];
structure_cn3.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-2.002000000000000e+02,NaN,NaN,-5.005000000000000e+02,NaN,NaN,NaN,NaN,NaN]);
structure_cn3.obj_point = [0;0];
structure_cn3.isClassical = 1;
structure_cn3.canShift = 0;
function_list{end+1} = structure_cn3;

% structure_cn4.bounds = [-1 1];
% structure_cn4.func_name = 'SF_{40}';
% structure_cn4.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn4.obj_func = @(x,n,M,o)(csendes_func(M*(x-o),n));
% structure_cn4.obj_gfunc = [];
% structure_cn4.obj_value = 0;
% structure_cn4.obj_point = [0;0];
% structure_cn4.isClassical = 1;
% structure_cn4.canShift = 0;
% function_list{end+1} = structure_cn4;

structure_cn5.bounds = [-100 100];
structure_cn5.func_name = 'SF_{43}';
structure_cn5.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn5.obj_func = @(x,n,M,o)(deb1_func(M*(x-o),n));
structure_cn5.obj_gfunc = [];
structure_cn5.canShift = 1;
structure_cn5.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-0.999999999922546,NaN,NaN,-0.999999999804150,NaN,NaN,NaN,NaN,NaN]);
structure_cn5.obj_point = [0;0];
structure_cn5.isClassical = 1;
structure_cn5.canShift = 0;
function_list{end+1} = structure_cn5;

structure_cn6.bounds = [0.1 100];
structure_cn6.func_name = 'SF_{44}';
structure_cn6.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn6.obj_func = @(x,n,M,o)(deb3_func(M*(x-o),n));
structure_cn6.obj_gfunc = [];
structure_cn6.canShift = 0;
structure_cn6.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-0.999999999990792,NaN,NaN,-0.999999999988897,NaN,NaN,NaN,NaN,NaN]);
structure_cn6.obj_point = [0;0];
structure_cn6.isClassical = 1;
structure_cn6.canShift = 0;
function_list{end+1} = structure_cn6;

% structure_cn7.bounds = [0.1 1];
% structure_cn7.func_name = 'SF_{74}';
% structure_cn7.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn7.obj_func = @(x,n,M,o)(mishra1_func(M*(x-o),n));
% structure_cn7.obj_gfunc = [];
% structure_cn7.canShift = 1;
% structure_cn7.obj_value = 0;
% structure_cn7.obj_point = [0;0];
% structure_cn7.isClassical = 1;
% structure_cn7.canShift = 0;
% function_list{end+1} = structure_cn7;


% structure_cn8.bounds = [0.1 1];
% structure_cn8.func_name = 'SF_{75}';
% structure_cn8.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn8.obj_func = @(x,n,M,o)(mishra2_func(M*(x-o),n));
% structure_cn8.obj_gfunc = [];
% structure_cn8.canShift = 1;
% structure_cn8.obj_value = 0;
% structure_cn8.obj_point = [0;0];
% structure_cn8.isClassical = 1;
% function_list{end+1} = structure_cn8;

% structure_cn9.bounds = [0.1 100];
% structure_cn9.func_name = 'SF_{80}';
% structure_cn9.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn9.obj_func = @(x,n,M,o)(mishra7_func(M*(x-o),n));
% structure_cn9.obj_gfunc = [];
% structure_cn9.canShift = 1;
% structure_cn9.obj_value = 0;
% structure_cn9.obj_point = [0;0];
% structure_cn9.isClassical = 1;
% function_list{end+1} = structure_cn9;

structure_cn10.bounds = [-100 100];
structure_cn10.func_name = 'SF_{87}';
structure_cn10.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn10.obj_func = @(x,n,M,o)(pathos_func(M*(x-o),n));
structure_cn10.obj_gfunc = [];
structure_cn10.canShift = 0;
structure_cn10.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-0.999999628068652,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn10.obj_point = [0;0];
structure_cn10.isClassical = 1;
function_list{end+1} = structure_cn10;

structure_cn11.bounds = [-10 10];
structure_cn11.func_name = 'SF_{89}';
structure_cn11.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn11.obj_func = @(x,n,M,o)(pinter_func(M*(x-o),n));
structure_cn11.obj_gfunc = [];
structure_cn11.canShift = 1;
structure_cn11.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,0,NaN,NaN,0,NaN,NaN,NaN,NaN,NaN]);
structure_cn11.obj_point = [0;0];
structure_cn11.isClassical = 1;
function_list{end+1} = structure_cn11;

% structure_cn12.bounds = [-10 10];
% structure_cn12.func_name = 'SF_{98}';
% structure_cn12.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn12.obj_func = @(x,n,M,o)(qing_func(M*(x-o),n));
% structure_cn12.obj_gfunc = [];
% structure_cn12.canShift = 1;
% structure_cn12.obj_value = 0;
% structure_cn12.obj_point = [0;0];
% structure_cn12.isClassical = 1;
% function_list{end+1} = structure_cn12;

structure_cn13.bounds = [-100 100];
structure_cn13.func_name = 'SF_{110}';
structure_cn13.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn13.obj_func = @(x,n,M,o)(salomon_func(M*(x-o),n));
structure_cn13.obj_gfunc = [];
structure_cn13.canShift = 1;
structure_cn13.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,0,NaN,NaN,0,NaN,NaN,NaN,NaN,NaN]);
structure_cn13.obj_point = [0;0];
structure_cn13.isClassical = 1;
function_list{end+1} = structure_cn13;

% structure_cn14.bounds = [-1 1];
% structure_cn14.func_name = 'SF_{111}';
% structure_cn14.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn14.obj_func = @(x,n,M,o)(sargan_func(M*(x-o),n));
% structure_cn14.obj_gfunc = [];
% structure_cn14.canShift = 1;
% structure_cn14.obj_value = 0;
% structure_cn14.obj_point = [0;0];
% structure_cn14.isClassical = 1;
% function_list{end+1} = structure_cn14;

% structure_cn15.bounds = [-100 100];
% structure_cn15.func_name = 'SF_{120}';
% structure_cn15.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn15.obj_func = @(x,n,M,o)(schwefel124_func(M*(x-o),n));
% structure_cn15.obj_gfunc = [];
% structure_cn15.canShift = 1;
% structure_cn15.obj_value = 0;
% structure_cn15.obj_point = [0;0];
% structure_cn15.isClassical = 1;
% function_list{end+1} = structure_cn15;

% structure_cn16.bounds = [0 10];
% structure_cn16.func_name = 'SF_{127}';
% structure_cn16.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn16.obj_func = @(x,n,M,o)(schwefel225_func(M*(x-o),n));
% structure_cn16.obj_gfunc = [];
% structure_cn16.canShift = 1;
% structure_cn16.obj_value = 0;
% structure_cn16.obj_point = [0;0];
% structure_cn16.isClassical = 1;
% function_list{end+1} = structure_cn16;

structure_cn17.bounds = [0 5];
structure_cn17.func_name = 'SF_{133}';
structure_cn17.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn17.obj_func = @(x,n,M,o)(shubert_func(M*(x-o),n));
structure_cn17.obj_gfunc = [];
structure_cn17.canShift = 1;
structure_cn17.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-1.867309088310232e+02,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn17.obj_point = [0;0];
structure_cn17.isClassical = 1;
function_list{end+1} = structure_cn17;

structure_cn18.bounds = [0 5];
structure_cn18.func_name = 'SF_{134}';
structure_cn18.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn18.obj_func = @(x,n,M,o)(shubert3_func(M*(x-o),n));
structure_cn18.obj_gfunc = [];
structure_cn18.canShift = 0;
structure_cn18.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-20.513790574335903,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn18.obj_point = [0;0];
structure_cn18.isClassical = 1;
function_list{end+1} = structure_cn18;

structure_cn19.bounds = [0 5];
structure_cn19.func_name = 'SF_{135}';
structure_cn19.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn19.obj_func = @(x,n,M,o)(shubert4_func(M*(x-o),n));
structure_cn19.obj_gfunc = [];
structure_cn19.canShift = 0;
structure_cn19.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-25.741770995451301,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn19.obj_point = [0;0];
structure_cn19.isClassical = 1;
function_list{end+1} = structure_cn19;

structure_cn20.bounds = [-5 5];
structure_cn20.func_name = 'SF_{144}';
structure_cn20.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn20.obj_func = @(x,n,M,o)(stybt_func(M*(x-o),n));
structure_cn20.obj_gfunc = [];
structure_cn20.canShift = 0;
structure_cn20.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-78.332331407542810,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn20.obj_point = [0;0];
structure_cn20.isClassical = 1;
function_list{end+1} = structure_cn20;

% structure_cn21.bounds = [-10 10];
% structure_cn21.func_name = 'SF_{150}';
% structure_cn21.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn21.obj_func = @(x,n,M,o)(trid6_func(M*(x-o),n));
% structure_cn21.obj_gfunc = [];
% structure_cn21.canShift = 1;
% structure_cn21.obj_value = 0;
% structure_cn21.obj_point = [0;0];
% structure_cn21.isClassical = 1;
% function_list{end+1} = structure_cn21;

structure_cn22.bounds = [-500 500];
structure_cn22.func_name = 'SF_{153}';
structure_cn22.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn22.obj_func = @(x,n,M,o)(trig1_func(M*(x-o),n));
structure_cn22.obj_gfunc = [];
structure_cn22.canShift = 0;
structure_cn22.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,0,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn22.obj_point = [0;0];
structure_cn22.isClassical = 1;
function_list{end+1} = structure_cn22;

structure_cn23.bounds = [-500 500];
structure_cn23.func_name = 'SF_{154}';
structure_cn23.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn23.obj_func = @(x,n,M,o)(trig2_func(M*(x-o),n));
structure_cn23.obj_gfunc = [];
structure_cn23.canShift = 0;
structure_cn23.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,1,NaN,NaN,1,NaN,NaN,NaN,NaN,NaN]);
structure_cn23.obj_point = [0;0];
structure_cn23.isClassical = 1;
function_list{end+1} = structure_cn23;

structure_cn24.bounds = [-100 100];
structure_cn24.func_name = 'SF_{166}';
structure_cn24.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn24.obj_func = @(x,n,M,o)(wavy_func(M*(x-o),n));
structure_cn24.obj_gfunc = [];
structure_cn24.canShift = 0;
structure_cn24.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,0,NaN,NaN,0,NaN,NaN,NaN,NaN,NaN]);
structure_cn24.obj_point = [0;0];
structure_cn24.isClassical = 1;
function_list{end+1} = structure_cn24;

structure_cn25.bounds = [-10 10];
structure_cn25.func_name = 'SF_{167}';
structure_cn25.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn25.obj_func = @(x,n,M,o)(whitley_func(M*(x-o),n));
structure_cn25.obj_gfunc = [];
structure_cn25.canShift = 0;
structure_cn25.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-3.972101902067051,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]);
structure_cn25.obj_point = [0;0];
structure_cn25.isClassical = 1;
function_list{end+1} = structure_cn25;

structure_cn26.bounds = [-20 20];
structure_cn26.func_name = 'SF_{171}';
structure_cn26.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
    9.5275232181884018e-01   3.0374827286555761e-01];
structure_cn26.obj_func = @(x,n,M,o)(xinshe_func(M*(x-o),n));
structure_cn26.obj_gfunc = [];
structure_cn26.canShift = 1;
structure_cn26.obj_value = containers.Map({'1','2','3','4','5','6','7','8','9','10'},[NaN,-1,NaN,NaN,-1,NaN,NaN,NaN,NaN,NaN]);
structure_cn26.obj_point = [0;0];
structure_cn26.isClassical = 1;
function_list{end+1} = structure_cn26;
% structure_cn27.bounds = [-20 20];
% structure_cn27.func_name = 'SF_{173}';
% structure_cn27.M2 = [-3.0374827286555761e-01   9.5275232181884006e-01;...
%     9.5275232181884018e-01   3.0374827286555761e-01];
% structure_cn27.obj_func = @(x,n,M,o)(zak_func(M*(x-o),n));
% structure_cn27.obj_gfunc = [];
% structure_cn27.canShift = 1;
% structure_cn27.obj_value = 0;
% structure_cn27.obj_point = [0;0];
% structure_cn27.isClassical = 1;
% function_list{end+1} = structure_cn27;
%% Basic functions
function [y] = rastrigin_func(x,d)
%test func: rastrigin
 y = 10*d;
 for k=1:d
     y = y + x(k).^2 - 10*cos(2*pi*x(k));
 end
end

function [y] = schaffer6_func(x)
x1 = x(1);
x2 = x(2);

fact1 = (sin(x1^2-x2^2))^2 - 0.5;
fact2 = (1 + 0.001*(x1^2+x2^2))^2;

y = 0.5 + fact1/fact2;
end

function [y] = exp_schaffer6_func(x,n)  
y = schaffer6_func([x(n);x(1)]);
for i=1:n-1
   y = y + schaffer6_func([x(i);x(i+1)]);
end 
end

function [y] = schaffer7_func(x,n)
  y = 0;
  for i=1:n-1
      si = sqrt(x(i).^2 + x(i+1).^2);      
      y = y + sqrt(si).*(sin(50*si.^(0.2))+1);
  end
  y = y/(n-1);
  y = y.^2;
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


function y = griewank_rosen_func(x,n)
  y = griewank_func(rosen_func([x(n);x(1)],2),1);
  for i=1:n-1
   y = y + griewank_func(rosen_func([x(i);x(i+1)],2),1);
  end 
end


function y = iHilbert_func(x,d) 
   n = ceil(sqrt(d));
   I = eye(n,n);
   H = zeros(n,n);
   Z = zeros(n,n);
   
   for i=1:n
       for k=1:n
           H(i,k) = 1./(i+k-1);
           Z(i,k) = x(i+n*(k-1));           
       end
   end
   
   W = H*Z-I;   
   y = 0;
   for i=1:n
       for k=1:n
           y = y + abs(W(i,k));
       end
   end   
end

function y  = lj_min_func(x,d)
   n = ceil(d/3);
   y = 12.7120622568;   
   for i=1:n-1
       for j=i+1:n
           dij = 0;
           for k=0:2
               dij = dij + (x(3*i+k-2)-x(3*j+k-2)).^2;
           end
           dij = dij.^3;
           
           cij = 1./(dij.^2) - 2./(dij);
           y = y + cij;
       end
   end   
end


function y = happy_cat_func(x,n)  
  s1 = 0;
  s2 = 0;  
  for i=1:n
      s1 = s1+x(i).^2;
      s2 = s2+x(i);
  end  
  y = abs(s1-n) + (0.5*s1 + s2)/n + 0.5;
end

function y = rosen_func(x,n)
 y = 0;
 for i=1:n-1
     y = y + 100*(x(i).^2 - x(i+1)).^2+(x(i)-1).^2;
 end
end 


function y = zak_func(x,n)  
  s1 = 0;
  s2 = 0;
  for i=1:n
      s1 = s1+x(i).^2; 
      s2 = s2 + 0.5*x(i);
  end
  y = s1 + s2.^2 + s2.^4;
end


function y = elliptic_func(x,n)
  y = 0;
  for i=1:n
    y = y + ((10^6).^((i-1)/(n-1))).*x(i).^2;
  end
end

function y = hgbat_func(x,n)  
  s1 = 0;
  s2 = 0;  
  for i=1:n
      s1 = s1+x(i).^2;
      s2 = s2+x(i);
  end  
  y = sqrt(abs(s1.^2-s2.^2)) + (0.5*s1 + s2)/n + 0.5;
end


function y = bent_cigar_func(x,n)
  y = x(1).^2;
  for i=2:n
      y = y + (10^6).*x(i).^2;
  end
end

function y = discuss_func(x,n)
  y = (10^6)*(x(1).^2);
  for i=2:n
      y = y + x(i).^2;
  end
end
%% Hybrid functions
function y = hf_func(x,S,nQ,Ms,Os,funcs)
%S = Permutation Set of indices
%x - problem dimension
%nQ - vector with number of variable per component
%Ms - Rotation Matrices cells
%Os - Shit matrices cells
%funcs - Function List
nF = length(Os);
y = 0;
ii = 1;
for i=1:nF
    hf = funcs{i};
    Mi = Ms{i};
    Oi = Os{i};
    y = y + hf(Mi*(x(S(ii:ii+nQ(i)-1),1)-Oi),nQ(i));
    ii = ii+nQ(i);
end
end

%% Composition Function
function y = comp_func(x,n,sigma,lbd,bias,Ms,Os,funcs)
%funcs - Function List
nF = length(sigma);
y = 0;
wi = cell(nF,1);
sum_wi = 0;
wmax = 0;
for i=1:nF
  Oi = Os{i};
  sumi = 0;
  for j=1:n
      sumi = sumi+(x(j)-Oi(j)).^2;
  end  
  wi{i} = exp(-sumi/(2*n*sigma(i).^2))./(sqrt(sumi));
  sum_wi = sum_wi + wi{i};
  if wi{i} > wmax
      wmax = wi{i};
  end
end

if wmax < 0.01
    sum_wi = nF;
    for i=1:nF
        wi{i} = 1;
    end
end
%sum_wi
for i=1:nF
    sf = funcs{i};  
    Oi = Os{i};
    Mi = Ms{i};
    omegai = wi{i}/(sum_wi);
    y = y + omegai*(lbd(i)*sf(Mi*(x-Oi),n)+bias(i));
    %omegai    
end
end
%% Classical Functions
function y = ackleym_func(x,n)
  y = 0;
  for i=1:n-1
      y = y + exp(-0.2)*sqrt(x(i).^2 + x(i+1).^2)+3*(cos(2*x(i))+sin(2*x(i+1)));
  end
end

function y = alpine2_func(x,n)
  y = 1;
  for i=1:n
      y = y.*sqrt(x(i)).*sin(x(i));
  end
end

function y = cosm_func(x,n)
   y = 0;
   for i=1:n
       y = y - 0.1*cos(5*pi*x(i)) - x(i).^2;
   end
end

function y = csendes_func(x,n)
  y = 0;
  for i=1:n
   y = y + (x(i).^6).*(2+sin(1./x(i)));
  end
end

function y = deb1_func(x,n)
  y = 0;
  for i=1:n
   y = y + sin(5*pi*x(i)).^6;
  end
  y = -y/n;
end


function y = deb3_func(x,n)
  y = 0;
  for i=1:n
   y = y + sin(5*pi*(x(i).^(0.75)-0.05)).^6;
  end
  y = -y/n;
end

function y = mishra1_func(x,n)
    s1 = 0;
    for i=1:n-1
        s1 = s1 + x(i);
    end
    y = (1+n-s1).^(n-s1);
end

function y = mishra2_func(x,n)
    s1 = 0;
    for i=1:n-1
        s1 = s1 + 0.5*(x(i)+x(i+1));
    end
    y = (1+n-s1).^(n-s1);
end

function y = mishra7_func(x,n)
  y = 1;
  for i=1:n
      y = y.*x(i);
  end
  y = (y - factorial(n)).^2;
end

function y = pathos_func(x,n)
y = 0;
  for i=1:n-1
      y = y + 0.5 + (sin(sqrt(100*x(i)^2+x(i+1).^2))-0.5)./(1+0.001*(x(i).^2-2*x(i).*x(i+1)+x(i+1).^2).^2);
  end
end

function y = pinter_func(x,n)  
  y1 = 0;
  y2 = 0;
  y3 = 0;  
  for i=1:n
       y1 = y1+ i*x(i).^2;
       if i==1
           x_1 = x(n);
       else
           x_1 = x(i-1);
       end
       
       if i==n
           x_p = x(1);
       else
           x_p = x(i+1);
       end
       
       A = x_1.*sin(x(i))+sin(x_p);
       B = x_1^2-2*x(i)+3*x_p-cos(x(i))+1;
       
       y2 = y2 + 20*i*(sin(A)).^2;      
       y3 = y3+i*log10(1+i*B^2);
  end  
  y = y1+y2+y3;
end

function y = qing_func(x,n)
    y = 0;
    for i=1:n
        y = y + (x(i).^2-i).^2;
    end
end

function y = salomon_func(x,n)  
  sum1 = 0;
  for i=1:n
      sum1 = sum1+x(i).^2;
  end
  y = 1 - cos(2*pi*sqrt(sum1))+0.1*sqrt(sum1);
end

function y = sargan_func(x,n)
   y = 0;
   for i=1:n
       yt = 0;
       for j=2:n
           yt = yt + x(i).*x(j);
       end
       y = y + x(i).^2 + 0.4*yt;
   end
   y = n*y;
end

function y = schwefel124_func(x,n)
  y = 0;
  for i=1:n
      y = y + (x(i)-1).^2 + (x(1)- x(i).^2).^2;
  end
end


function y = schwefel225_func(x,n)
  y = 0;
  for i=2:n
      y = y + (x(i)-1).^2 + (x(1)- x(i).^2).^2;
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



function [y] = shubert3_func(x,n)
y = 0;
for i=1:n
    sum1 = 0;
    for j=1:5
        sum1 = sum1 + j*sin((j+1)*x(i)+j);
    end
    y = y + sum1;
end
end

function [y] = shubert4_func(x,n)
y = 0;
for i=1:n
    sum1 = 0;
    for j=1:5
        sum1 = sum1 + j*cos((j+1)*x(i)+j);
    end
    y = y + sum1;
end
end

function y = stybt_func(x,n)
    y = 0;
    for i=1:n
      y = y + x(i).^4 - 16*x(i).^2 + 5*x(i);
    end
    y = 0.5*y;
end

function y = trid6_func(x,n)
  y1 = 0;
  y2 = 0;
  for i=1:n
      y1 = y1 +(x(i)-1).^2;
  end
  
  for i=2:n
      y2 = y2 -x(i).*x(i-1);
  end
  
  y = y1 + y2;
end

function y = trig1_func(x,n)
  s = 0;
  for i=1:n
      s = s + cos(x(i));
  end
  
  y = 0;
  for i=1:n
      y = y + (n-s+i*(1-cos(x(i))-sin(x(i)))).^2;
  end
end

function y = trig2_func(x,n)
  y = 1;
  for i=1:n
     y = y + 8*(sin(7*(x(i)-0.9).^2)).^2+ 6*(sin(14*(x(i)-0.9).^2)).^2+(x(i)-0.9).^2;      
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

function y = whitley_func(x,n)
   y = 0;
   for i=1:n
       for j=1:n
           y = y + ((100*(x(i).^2 - x(j)).^2 + (1-x(j)).^2).^2)/4000-...
               cos(100*(x(i).^2-x(j)).^2+(1-x(j)).^2+1);
       end
   end
end

function y = xinshe_func(x,n)
  b = 15;
  m = 5;
  s1 = 0;
  s2 = 0;
  s3 = 1;
  for i=1:n
      s1 = s1 -(x(i)/b).^(2*m);
      s2 = s2 + x(i).^2;
      s3 = s3*cos(x(i)).^2;
  end
  
  y = exp(s1)-2*exp(-s2).*s3;
end

