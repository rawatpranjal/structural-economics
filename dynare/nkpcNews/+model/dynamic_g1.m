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
g1 = zeros(12, 18);
g1(1,8)=(-T(3));
g1(1,12)=1;
g1(1,16)=(-params(2));
g1(2,8)=1;
g1(2,15)=(-1);
g1(2,9)=T(2);
g1(2,11)=(-T(2));
g1(2,16)=T(2);
g1(3,3)=(-1);
g1(3,6)=1;
g1(3,13)=(-(1-params(1)));
g1(4,5)=(-params(5));
g1(4,13)=(-params(6));
g1(4,14)=1;
g1(5,5)=(-1);
g1(5,6)=1;
g1(6,1)=(-params(3));
g1(6,3)=1;
g1(6,17)=(-1);
g1(7,2)=(-params(4));
g1(7,4)=1;
g1(7,18)=(-1);
g1(8,3)=(-((1-params(3))*T(1)*(-params(5))));
g1(8,4)=(-(1-params(4)));
g1(8,9)=1;
g1(9,10)=1;
g1(9,11)=(-1);
g1(9,16)=1;
g1(10,3)=(-T(1));
g1(10,7)=1;
g1(11,6)=(-1);
g1(11,7)=1;
g1(11,8)=1;
g1(12,9)=(-1);
g1(12,11)=1;

end
