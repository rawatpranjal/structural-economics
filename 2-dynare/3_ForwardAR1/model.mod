var y;
varexo e;
parameters rho se;

rho = 0.9;
se = 0.01;

model;
y = rho * y(+1) + e;
end;

initval;
e = 0;
y = 1;
end;
steady;
check;

shocks;
var e; stderr se;
end;

stoch_simul(periods=20, drop=0, order=1);
rplot y;
rplot e;
