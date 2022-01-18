function residual = static_resid(T, y, x, params, T_flag)
% function residual = static_resid(T, y, x, params, T_flag)
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
%   residual
%

if T_flag
    T = ClassicalMonetaryModel.static_resid_tt(T, y, x, params);
end
residual = zeros(12, 1);
lhs = y(2);
rhs = T(1)*T(2);
residual(1) = lhs - rhs;
lhs = y(11);
rhs = params(2)/y(3);
residual(2) = lhs - rhs;
lhs = y(6);
rhs = 1/y(11);
residual(3) = lhs - rhs;
lhs = y(8);
rhs = y(4)*T(3);
residual(4) = lhs - rhs;
lhs = y(2);
rhs = y(4)*(1-params(1))*T(4);
residual(5) = lhs - rhs;
lhs = y(6);
rhs = y(3)*y(7);
residual(6) = lhs - rhs;
lhs = y(7);
rhs = T(5);
residual(7) = lhs - rhs;
lhs = y(1);
rhs = y(8);
residual(8) = lhs - rhs;
lhs = log(y(4));
rhs = log(y(4))*params(7)+x(1);
residual(9) = lhs - rhs;
lhs = log(y(12));
rhs = log(y(12))*params(8)+x(2);
residual(10) = lhs - rhs;
lhs = y(9);
rhs = y(9)*params(9)+x(3);
residual(11) = lhs - rhs;
lhs = y(10);
rhs = 4*log(y(3));
residual(12) = lhs - rhs;
if ~isreal(residual)
  residual = real(residual)+imag(residual).^2;
end
end
