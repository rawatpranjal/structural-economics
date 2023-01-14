var y x;
varexo e;
parameters a b c rho se;

a = 0.1;
b = 0.7;
c = 1;
rho = 0.9;
se = 0.02;

model;
y = a*y(+1)+b*y(-1)+c*x;
x = rho*x(-1)+e;
end;

initval;
y = 0;
x = 0;
e = 0;
end;

steady;
check;

shocks;
var e; stderr se;
end;

stoch_simul(order=2);


