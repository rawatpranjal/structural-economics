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
    T = model2.dynamic_resid_tt(T, y, x, params, steady_state, it_);
end
residual = zeros(20, 1);
lhs = y(20);
rhs = params(2)*y(32)+y(16)*T(3);
residual(1) = lhs - rhs;
lhs = y(16);
rhs = T(2)*(y(19)-y(32)-y(17))+y(31);
residual(2) = lhs - rhs;
lhs = y(14);
rhs = y(11)+(1-params(1))*y(21);
residual(3) = lhs - rhs;
lhs = y(22);
rhs = params(5)*y(13)+params(6)*y(21);
residual(4) = lhs - rhs;
lhs = y(14);
rhs = y(13);
residual(5) = lhs - rhs;
lhs = y(11);
rhs = params(3)*y(1)+x(it_, 1)+y(10);
residual(6) = lhs - rhs;
lhs = y(12);
rhs = params(4)*y(2)+x(it_, 3);
residual(7) = lhs - rhs;
lhs = y(17);
rhs = y(12)*(1-params(4))+y(11)*(1-params(3))*T(1)*(-params(5));
residual(8) = lhs - rhs;
lhs = y(18);
rhs = y(19)-y(32);
residual(9) = lhs - rhs;
lhs = y(15);
rhs = T(1)*y(11);
residual(10) = lhs - rhs;
lhs = y(16);
rhs = y(14)-y(15);
residual(11) = lhs - rhs;
lhs = y(19);
rhs = y(17)+y(20)*params(9)+y(16)*params(10);
residual(12) = lhs - rhs;
lhs = y(23);
rhs = x(it_, 2);
residual(13) = lhs - rhs;
lhs = y(24);
rhs = y(3);
residual(14) = lhs - rhs;
lhs = y(25);
rhs = y(4);
residual(15) = lhs - rhs;
lhs = y(26);
rhs = y(5);
residual(16) = lhs - rhs;
lhs = y(27);
rhs = y(6);
residual(17) = lhs - rhs;
lhs = y(28);
rhs = y(7);
residual(18) = lhs - rhs;
lhs = y(29);
rhs = y(8);
residual(19) = lhs - rhs;
lhs = y(30);
rhs = y(9);
residual(20) = lhs - rhs;

end
