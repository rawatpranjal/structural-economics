function T = static_resid_tt(T, y, x, params)
% function T = static_resid_tt(T, y, x, params)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T         [#temp variables by 1]  double   vector of temporary terms to be filled by function
%   y         [M_.endo_nbr by 1]      double   vector of endogenous variables in declaration order
%   x         [M_.exo_nbr by 1]       double   vector of exogenous variables in declaration order
%   params    [M_.param_nbr by 1]     double   vector of parameter values in declaration order
%
% Output:
%   T         [#temp variables by 1]  double   vector of temporary terms
%

assert(length(T) >= 8);

T(1) = exp(y(2))^(-params(3));
T(2) = T(1)*params(1)/params(7);
T(3) = exp(y(3))/exp(y(4));
T(4) = params(5)*exp(y(5))*T(3)^(params(5)-1);
T(5) = T(4)+1-params(4);
T(6) = params(2)*exp(y(2))^params(3);
T(7) = exp(y(5))*exp(y(3))^params(5);
T(8) = exp(y(4))^(1-params(5));

end
