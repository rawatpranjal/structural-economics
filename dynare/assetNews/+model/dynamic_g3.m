function g3 = dynamic_g3(T, y, x, params, steady_state, it_, T_flag)
% function g3 = dynamic_g3(T, y, x, params, steady_state, it_, T_flag)
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
%   g3
%

if T_flag
    T = model.dynamic_g3_tt(T, y, x, params, steady_state, it_);
end
g3_i = zeros(14,1);
g3_j = zeros(14,1);
g3_v = zeros(14,1);

g3_i(1)=1;
g3_i(2)=1;
g3_i(3)=1;
g3_i(4)=1;
g3_i(5)=1;
g3_i(6)=1;
g3_i(7)=1;
g3_i(8)=1;
g3_i(9)=1;
g3_i(10)=1;
g3_i(11)=2;
g3_i(12)=2;
g3_i(13)=2;
g3_i(14)=2;
g3_j(1)=1;
g3_j(2)=8;
g3_j(3)=2;
g3_j(4)=71;
g3_j(5)=65;
g3_j(6)=11;
g3_j(7)=638;
g3_j(8)=632;
g3_j(9)=578;
g3_j(10)=92;
g3_j(11)=193;
g3_j(12)=466;
g3_j(13)=274;
g3_j(14)=547;
g3_v(1)=(-(T(3)*T(4)*T(7)+T(4)*(T(4)*T(3)*T(4)+T(3)*T(7))+T(3)*T(4)*T(7)+T(3)*params(5)*(y(1)+y(1))/(y(1)*y(1)*y(1)*y(1))));
g3_v(2)=(-(T(4)*T(4)*params(4)*T(3)+params(4)*T(3)*T(7)));
g3_v(3)=(-(T(4)*T(4)*params(3)*T(3)+params(3)*T(3)*T(7)));
g3_v(4)=(-(T(4)*params(4)*params(4)*T(3)));
g3_v(5)=(-(T(4)*params(4)*params(3)*T(3)));
g3_v(6)=(-(T(4)*params(3)*params(3)*T(3)));
g3_v(7)=(-(params(4)*params(4)*params(4)*T(3)));
g3_v(8)=(-(params(4)*params(4)*params(3)*T(3)));
g3_v(9)=(-(params(4)*params(3)*params(3)*T(3)));
g3_v(10)=(-(params(3)*params(3)*params(3)*T(3)));
g3_v(11)=T(8);
g3_v(12)=(-T(9));
g3_v(13)=y(3)*getPowerDeriv(y(4),(-params(2)),3);
g3_v(14)=(-(T(9)+T(9)+T(9)+(y(7)+y(6))*params(1)*getPowerDeriv(y(7),(-params(2)),3)));
g3 = sparse(g3_i,g3_j,g3_v,3,729);
end
