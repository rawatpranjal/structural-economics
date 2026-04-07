function T = dynamic_g2_tt(T, y, x, params, steady_state, it_)
% function T = dynamic_g2_tt(T, y, x, params, steady_state, it_)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double  vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double  vector of endogenous variables in the order stored
%                                                    in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double  matrix of exogenous variables (in declaration order)
%                                                    for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double  vector of steady state values
%   params        [M_.param_nbr by 1]        double  vector of parameter values in declaration order
%   it_           scalar                     double  time period for exogenous variables for which
%                                                    to evaluate the model
%
% Output:
%   T           [#temp variables by 1]       double  vector of temporary terms
%

assert(length(T) >= 9);

T = model.dynamic_g1_tt(T, y, x, params, steady_state, it_);

T(7) = params(5)*(-1)/(y(1)*y(1));
T(8) = getPowerDeriv(y(4),(-params(2)),2);
T(9) = params(1)*getPowerDeriv(y(7),(-params(2)),2);

end
