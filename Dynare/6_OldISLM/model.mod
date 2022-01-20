var Y C I r;
varexo v;
parameters a b i0 h r0 k se1 se2;

a = 100;
b = 0.8;
i0 = 10;
h = 20;
r0 = 5;
k = 0.01;
se1 = 1;
se2 = 1;

model;
Y = C + I;
C = a + b * Y(-1);
I = i0 - h*r - v;
r = r0 + k*Y; %LM
end;

initval;
Y = 0;
C = 0;
I = 0;
r = 0;
end;

steady;
check;

shocks;
var v; stderr se2;
end;

stoch_simul(order=2);


