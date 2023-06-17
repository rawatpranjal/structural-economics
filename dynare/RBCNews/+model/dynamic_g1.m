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
g1 = zeros(16, 31);
g1(1,12)=exp(y(12))*getPowerDeriv(exp(y(12)),(-params(3)),1);
g1(1,27)=(-(T(4)*params(1)/params(7)*exp(y(27))*getPowerDeriv(exp(y(27)),(-params(3)),1)));
g1(1,13)=(-(T(1)*params(5)*exp(y(29))*T(2)*T(8)));
g1(1,28)=(-(T(1)*params(5)*exp(y(29))*T(8)*(-(exp(y(13))*exp(y(28))))/(exp(y(28))*exp(y(28)))));
g1(1,29)=(-(T(1)*T(3)));
g1(2,12)=params(2)*exp(y(12))*getPowerDeriv(exp(y(12)),params(3),1)/(1-exp(y(14)));
g1(2,14)=(-(T(5)*(-exp(y(14)))))/((1-exp(y(14)))*(1-exp(y(14))));
g1(2,17)=(-exp(y(17)));
g1(3,1)=(-((1-params(4))*exp(y(1))));
g1(3,13)=params(7)*exp(y(13));
g1(3,18)=(-exp(y(18)));
g1(4,11)=exp(y(11));
g1(4,12)=(-exp(y(12)));
g1(4,18)=(-exp(y(18)));
g1(5,11)=exp(y(11));
g1(5,1)=(-(T(7)*exp(y(15))*exp(y(1))*getPowerDeriv(exp(y(1)),params(5),1)));
g1(5,14)=(-(T(6)*exp(y(14))*getPowerDeriv(exp(y(14)),1-params(5),1)));
g1(5,15)=(-(T(6)*T(7)));
g1(6,11)=(-(exp(y(11))*(1-params(5))/exp(y(14))));
g1(6,14)=(-((-(exp(y(14))*exp(y(11))*(1-params(5))))/(exp(y(14))*exp(y(14)))));
g1(6,17)=exp(y(17));
g1(7,11)=(-(exp(y(11))*params(5)*4/exp(y(1))));
g1(7,1)=(-((-(exp(y(1))*exp(y(11))*params(5)*4))/(exp(y(1))*exp(y(1)))));
g1(7,16)=1;
g1(8,2)=(-params(6));
g1(8,15)=1;
g1(8,31)=(-1);
g1(8,10)=(-1);
g1(9,30)=(-1);
g1(9,19)=1;
g1(10,3)=(-1);
g1(10,20)=1;
g1(11,4)=(-1);
g1(11,21)=1;
g1(12,5)=(-1);
g1(12,22)=1;
g1(13,6)=(-1);
g1(13,23)=1;
g1(14,7)=(-1);
g1(14,24)=1;
g1(15,8)=(-1);
g1(15,25)=1;
g1(16,9)=(-1);
g1(16,26)=1;

end
