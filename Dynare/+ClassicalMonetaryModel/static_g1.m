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
    T = ClassicalMonetaryModel.static_g1_tt(T, y, x, params);
end
g1 = zeros(12, 12);
g1(1,1)=(-(T(2)*getPowerDeriv(y(1),params(6),1)));
g1(1,2)=1;
g1(1,5)=(-(T(1)*getPowerDeriv(y(5),params(7),1)));
g1(2,3)=(-((-params(2))/(y(3)*y(3))));
g1(2,11)=1;
g1(3,6)=1;
g1(3,11)=(-((-1)/(y(11)*y(11))));
g1(4,4)=(-T(3));
g1(4,5)=(-(y(4)*getPowerDeriv(y(5),1-params(1),1)));
g1(4,8)=1;
g1(5,2)=1;
g1(5,4)=(-((1-params(1))*T(4)));
g1(5,5)=(-(y(4)*(1-params(1))*getPowerDeriv(y(5),(-params(1)),1)));
g1(6,3)=(-y(7));
g1(6,6)=1;
g1(6,7)=(-y(3));
g1(7,3)=(-(exp(y(9))*1/params(2)*getPowerDeriv(y(3),params(8),1)));
g1(7,6)=1;
g1(7,9)=(-T(5));
g1(8,1)=1;
g1(8,8)=(-1);
g1(9,4)=1/y(4)-params(3)*1/y(4);
g1(10,12)=1/y(12)-params(4)*1/y(12);
g1(11,9)=1-params(5);
g1(12,3)=(-(4*1/y(3)));
g1(12,10)=1;
if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
end
end
