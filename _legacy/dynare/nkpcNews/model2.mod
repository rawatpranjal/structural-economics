% ---------------------------------------------- %
% optimal rule with feedback to target variables %
% ---------------------------------------------- %
@#define MONPOL = 2
@#include "NK_linear_common.inc"

SIGMA   = 1;
VARPHI  = 5;
THETA   = 3/4;
RHOZ    = 0.5;
RHOA    = 0.9;
BETA    = 0.99;
ALPHA   = 1/4;
EPSILON = 9;

PHI_PIE = 1.5;
PHI_Y   = 0.125;

steady;
check;

stoch_simul;

estimated_params;
PHI_PIE, 1.5, 0, 2;
PHI_Y, 0.125, 0, 2;
end;

varobs a  z  c  y  y_nat  y_gap  r_nat  r_real  ii  pie  n  w; % dynare_sensitivity requires varobs block
                                                               % alternative and quick way to assume all variables are observbable:
                                                               % options_.varobs = M_.endo_names; 

