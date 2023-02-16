function g1 = dynamic_g1(T, y, x, params, steady_state, it_, T_flag)
% function g1 = dynamic_g1(T, y, x, params, steady_state, it_, T_flag)
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
%   g1
%

if T_flag
    T = model.dynamic_g1_tt(T, y, x, params, steady_state, it_);
end
g1 = zeros(2, 6);
g1(1,1)=(-(exp(params(4)*log(y(1))+params(3)*x(it_, 1))*params(4)*1/y(1)));
g1(1,3)=1;
g1(1,6)=(-(params(3)*exp(params(4)*log(y(1))+params(3)*x(it_, 1))));
g1(2,2)=T(1);
g1(2,4)=(-T(2));
g1(2,3)=y(2)*getPowerDeriv(y(3),(-params(2)),1);
g1(2,5)=(-(T(2)+(y(5)+y(4))*params(1)*getPowerDeriv(y(5),(-params(2)),1)));

end
