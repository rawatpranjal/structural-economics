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
    T = ClassicalMonetaryModel.dynamic_g1_tt(T, y, x, params, steady_state, it_);
end
g1 = zeros(12, 23);
g1(1,6)=(-(T(2)*getPowerDeriv(y(6),params(6),1)));
g1(1,7)=1;
g1(1,10)=(-(T(1)*getPowerDeriv(y(10),params(7),1)));
g1(2,6)=(-(y(20)/y(17)*params(2)*(-y(18))/(y(6)*y(6))*T(8)/y(19)));
g1(2,18)=(-(y(20)/y(17)*params(2)*T(8)*1/y(6)/y(19)));
g1(2,19)=(-((-T(4))/(y(19)*y(19))));
g1(2,16)=1;
g1(2,17)=(-(T(3)*(-y(20))/(y(17)*y(17))/y(19)));
g1(2,20)=(-(T(3)*1/y(17)/y(19)));
g1(3,11)=1;
g1(3,16)=(-((-1)/(y(16)*y(16))));
g1(4,9)=(-T(5));
g1(4,10)=(-(y(9)*getPowerDeriv(y(10),1-params(1),1)));
g1(4,13)=1;
g1(5,7)=1;
g1(5,9)=(-((1-params(1))*T(6)));
g1(5,10)=(-(y(9)*(1-params(1))*getPowerDeriv(y(10),(-params(1)),1)));
g1(6,19)=(-y(12));
g1(6,11)=1;
g1(6,12)=(-y(19));
g1(7,8)=(-(exp(y(14))*1/params(2)*getPowerDeriv(y(8),params(8),1)));
g1(7,11)=1;
g1(7,14)=(-T(7));
g1(8,6)=1;
g1(8,13)=(-1);
g1(9,2)=(-(params(3)*1/y(2)));
g1(9,9)=1/y(9);
g1(9,21)=(-1);
g1(10,5)=(-(params(4)*1/y(5)));
g1(10,17)=1/y(17);
g1(10,22)=(-1);
g1(11,4)=(-params(5));
g1(11,14)=1;
g1(11,23)=(-1);
g1(12,1)=(-(4*(-(1/y(1)))));
g1(12,6)=(-(4*1/y(6)));
g1(12,8)=(-(4*1/y(8)));
g1(12,3)=(-(4*(-(params(9)*(-(1/y(3)))))));
g1(12,11)=(-(4*(-(params(9)*1/y(11)))));
g1(12,15)=1;

end
