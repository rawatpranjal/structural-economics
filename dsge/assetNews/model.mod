// ---------------------------------------------------------------------------
// Lucas-tree asset pricing with news shocks - model specification.
// This file is documentation only; run.py does not execute Dynare.
// To run in Dynare, install via `brew install dynare` (macOS) and call
// `dynare model.mod` from inside Octave or MATLAB. See dynare.org.
// ---------------------------------------------------------------------------

var p d;
varexo z n;
parameters beta gamma sigma1 sigma2 rho;
beta = 0.99;
gamma = 2.0;
rho = 0.9;
sigma1 = 0.1;
sigma2 = 0.1;

model;
	d=exp(rho*log(d(-1))+sigma1*n(-1)+sigma2*z); 
	p*d^(-gamma) = beta*d(+1)^(-gamma)*(p(+1)+d(+1)); 
end;

steady_state_model;
d=1;
p=d/(1/beta-1);
end;

initval;
d=1;
end;


resid;
steady;
check;

shocks;
var z; stderr 1;
var n; stderr 1;
end;

stoch_simul(order=3);
rplot d;
rplot p;

	