var y x;
varexo e;
parameters alpha beta rho sigma;

alpha = 0.7;
beta = 0.7;
rho = 0.9;
sigma = 0.01;

model;
y = alpha * y(+1) + beta * x;
x = rho * x(-1) + e;
end;

initval;
e = 0;
x = 0;
y = 0;
end;

steady;
check;

shocks;
var e;
stderr sigma;
end;

stoch_simul(irf=20,order=1);