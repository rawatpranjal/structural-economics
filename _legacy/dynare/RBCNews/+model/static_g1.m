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
g1 = zeros(16, 16);
g1(1,2)=T(9)-T(5)*params(1)/params(7)*T(9);
g1(1,3)=(-(T(2)*params(5)*exp(y(5))*T(3)*T(10)));
g1(1,4)=(-(T(2)*params(5)*exp(y(5))*T(10)*(-(exp(y(3))*exp(y(4))))/(exp(y(4))*exp(y(4)))));
g1(1,5)=(-(T(2)*T(4)));
g1(2,2)=params(2)*exp(y(2))*getPowerDeriv(exp(y(2)),params(3),1)/(1-exp(y(4)));
g1(2,4)=(-(T(6)*(-exp(y(4)))))/((1-exp(y(4)))*(1-exp(y(4))));
g1(2,7)=(-exp(y(7)));
g1(3,3)=params(7)*exp(y(3))-exp(y(3))*(1-params(4));
g1(3,8)=(-exp(y(8)));
g1(4,1)=exp(y(1));
g1(4,2)=(-exp(y(2)));
g1(4,8)=(-exp(y(8)));
g1(5,1)=exp(y(1));
g1(5,3)=(-(T(8)*exp(y(5))*exp(y(3))*getPowerDeriv(exp(y(3)),params(5),1)));
g1(5,4)=(-(T(7)*exp(y(4))*getPowerDeriv(exp(y(4)),1-params(5),1)));
g1(5,5)=(-(T(7)*T(8)));
g1(6,1)=(-(exp(y(1))*(1-params(5))/exp(y(4))));
g1(6,4)=(-((-(exp(y(4))*exp(y(1))*(1-params(5))))/(exp(y(4))*exp(y(4)))));
g1(6,7)=exp(y(7));
g1(7,1)=(-(exp(y(1))*params(5)*4/exp(y(3))));
g1(7,3)=(-((-(exp(y(3))*exp(y(1))*params(5)*4))/(exp(y(3))*exp(y(3)))));
g1(7,6)=1;
g1(8,5)=1-params(6);
g1(9,9)=1;
g1(10,10)=1;
g1(11,11)=1;
g1(12,12)=1;
g1(13,13)=1;
g1(14,14)=1;
g1(15,15)=1;
g1(16,16)=1;
if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
end
end
