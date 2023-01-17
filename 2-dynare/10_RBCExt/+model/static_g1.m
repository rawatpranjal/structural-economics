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
g1 = zeros(5, 5);
g1(1,1)=exp(y(1));
g1(1,4)=(-(exp(y(5))*exp(y(4))*getPowerDeriv(exp(y(4)),params(1),1)));
g1(1,5)=(-T(1));
g1(2,3)=T(5)-T(4)*params(2)*T(5);
g1(2,4)=(-(T(2)*params(2)*exp(y(5))*params(1)*exp(y(4))*getPowerDeriv(exp(y(4)),params(1)-1,1)));
g1(2,5)=(-(T(2)*params(2)*T(3)));
g1(3,2)=(-exp(y(2)));
g1(3,4)=exp(y(4))-exp(y(4))*(1-params(5));
g1(4,1)=exp(y(1));
g1(4,2)=(-exp(y(2)));
g1(4,3)=(-exp(y(3)));
g1(5,5)=1-params(6);
if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
end
end
