// ---------------------------------------------------------------------------
// Linearized New Keynesian DSGE - model specification.
// This file is documentation only; run.py does not execute Dynare.
// To run in Dynare, install via `brew install dynare` (macOS) and call
// `dynare model.mod` from inside Octave or MATLAB. See dynare.org.
//
// The calibration here (phi_pi=0.33, kappa=0.95) violates the Taylor
// principle and is preserved as a teaching counterexample. The Python
// solver (run.py) uses the standard determinate calibration phi_pi=1.5,
// kappa=0.3 instead - see the README for the contrast.
// ---------------------------------------------------------------------------

var y pi i;
varexo e;
parameters sigma beta phi_pi phi_y kappa rho se;

phi_y  = 0.025;
phi_pi = 0.33;
beta   = 0.99;
rho    = -log(beta);
kappa  = 0.95;
sigma  = 1.3;
se     = 0.01;

model;
    y  = y(+1) - sigma^(-1)*(i - pi(+1) - rho);
    pi = beta*pi(+1) + kappa*y;
    i  = rho + phi_pi*pi + phi_y*y + e;
end;

initval;
    y  = 0;
    pi = 0;
    i  = rho;
end;

shocks;
    var e; stderr se;
end;

steady;
check;

stoch_simul(order=1, irf=40);
