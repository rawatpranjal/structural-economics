% AR(1) Model
% x(t)=rho*x(t-1)+e(t)

var x;
varexo e; 
parameters rho sigma;

rho = 0.9;
sigma = 0.01;

model;
x=rho*x(-1)+e;
end;

initval;
e=0;
x=0;
end;
steady;

check;

shocks;
var e; 
stderr sigma;
end;

stoch_simul(periods = 100, drop = 0, order =1);
rplot x;
rplot e;