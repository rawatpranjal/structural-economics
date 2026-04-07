var y x;
varexo e;
parameters a b rho se;

a = 0.1;
b = 1;
rho = 0.9;
se = 0.02;

model;
y = a * y(+1) + b * x;
x = rho * x(-1) + e;
end;

% y = (1/(1-a)) * (e + rho*e(-1)+rho^2*e(-2) ...)

initval;
e = 0;
x = 0;
y = 0;
end;
steady;

check;

shocks;
var e; stderr se;
end;

stoch_simul(periods = 20, drop = 0, order =1);
rplot y;
rplot x;
rplot e;