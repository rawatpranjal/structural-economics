var C Y I G; 
varexo e;
parameters alpha beta rho sigma Gbar;

alpha = 0.3;
beta = 0.8;
rho = 0.9;
sigma = 0.01;
Gbar = 1;

model;
C = beta * Y(-1);
G = rho * G(-1) + (1-rho)*Gbar + e;
I = alpha * (C - C(-1));
Y = C + I + G;
end;

initval;
e = 0;
G = 0;
Y = 0;
C = 0;
I = 0;
end;
steady;
check;

shocks;
var e;
stderr sigma;
end;

stoch_simul(irf=20,order=1);
