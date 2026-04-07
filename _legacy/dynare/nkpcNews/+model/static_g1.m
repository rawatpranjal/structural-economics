function g1 = static_g1(T, y, x, params, T_flag)
% function g1 = static_g1(T, y, x, params, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T         [#temp variables by 1]  double   vector of temporary terms to be filled by function
%   y         [M_.endo_nbr by 1]      double   vector of endogenous variables in declaration order
%   x         [M_.exo_nbr by 1]       double   vector of exogenous variables in declaration order
%   params    [M_.param_nbr by 1]     double   vector of parameter values in declaration order
%                                              to evaluate the model
%   T_flag    boolean                 boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   g1
%

if T_flag
    T = model.static_g1_tt(T, y, x, params);
end
g1 = zeros(12, 12);
g1(1,6)=(-T(2));
g1(1,10)=1-params(2);
g1(2,7)=T(3);
g1(2,9)=(-T(3));
g1(2,10)=T(3);
g1(3,1)=(-1);
g1(3,4)=1;
g1(3,11)=(-(1-params(1)));
g1(4,3)=(-params(5));
g1(4,11)=(-params(6));
g1(4,12)=1;
g1(5,3)=(-1);
g1(5,4)=1;
g1(6,1)=1-params(3);
g1(7,2)=1-params(4);
g1(8,1)=(-((1-params(3))*T(1)*(-params(5))));
g1(8,2)=(-(1-params(4)));
g1(8,7)=1;
g1(9,8)=1;
g1(9,9)=(-1);
g1(9,10)=1;
g1(10,1)=(-T(1));
g1(10,5)=1;
g1(11,4)=(-1);
g1(11,5)=1;
g1(11,6)=1;
g1(12,7)=(-1);
g1(12,9)=1;
if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
end
end
