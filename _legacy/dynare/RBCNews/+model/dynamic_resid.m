function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
% function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   residual
%

if T_flag
    T = model.dynamic_resid_tt(T, y, x, params, steady_state, it_);
end
residual = zeros(16, 1);
lhs = exp(y(12))^(-params(3));
rhs = T(1)*T(4);
residual(1) = lhs - rhs;
lhs = T(5)/(1-exp(y(14)));
rhs = exp(y(17));
residual(2) = lhs - rhs;
lhs = params(7)*exp(y(13));
rhs = (1-params(4))*exp(y(1))+exp(y(18));
residual(3) = lhs - rhs;
lhs = exp(y(11));
rhs = exp(y(12))+exp(y(18));
residual(4) = lhs - rhs;
lhs = exp(y(11));
rhs = T(6)*T(7);
residual(5) = lhs - rhs;
lhs = exp(y(17));
rhs = exp(y(11))*(1-params(5))/exp(y(14));
residual(6) = lhs - rhs;
lhs = y(16);
rhs = exp(y(11))*params(5)*4/exp(y(1));
residual(7) = lhs - rhs;
lhs = y(15);
rhs = params(6)*y(2)+x(it_, 2)+y(10);
residual(8) = lhs - rhs;
lhs = y(19);
rhs = x(it_, 1);
residual(9) = lhs - rhs;
lhs = y(20);
rhs = y(3);
residual(10) = lhs - rhs;
lhs = y(21);
rhs = y(4);
residual(11) = lhs - rhs;
lhs = y(22);
rhs = y(5);
residual(12) = lhs - rhs;
lhs = y(23);
rhs = y(6);
residual(13) = lhs - rhs;
lhs = y(24);
rhs = y(7);
residual(14) = lhs - rhs;
lhs = y(25);
rhs = y(8);
residual(15) = lhs - rhs;
lhs = y(26);
rhs = y(9);
residual(16) = lhs - rhs;

end
