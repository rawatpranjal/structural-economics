var y i c k a;
varexo e;
parameters alpha beta gamma sigma delta rho se;

alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho = 0.95;
sigma = 1;
se = 0.01;

model;
exp(y) = exp(a)*exp(k(-1))^(alpha);
exp(c)^(-sigma)=beta*exp(c(+1))^(-sigma)*(alpha*exp(a(+1))*exp(k)^(alpha-1)+(1-delta));
exp(k) = exp(i) + (1-delta)*exp(k(-1));
exp(y)= exp(c) + exp(i);
a = rho * a(-1) + e;
end;

initval;
k = log(29);
y = log(2);
a = 0;
c = log(2.5);
i = log(1.5);
end;

shocks;
var e; stderr se;
end;

steady;

stoch_simul(hp_filter=1600,order=1,irf=40);



