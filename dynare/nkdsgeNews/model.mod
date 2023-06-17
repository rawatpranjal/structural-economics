var y pi i;
varexo e n;
parameters sigma beta phi_pi phi_y k rho sigma2;

r_nat=-sigma*psi_y*(1-rho_a)*a+(1-rho_z)*z;
phi_y = 0.025;
phi_pi = 0.33;
beta = 0.99;
rho = -log(beta);
k = 0.95;
sigma = 1.3;
se = 0.01;
sigma2 = 0.1;

model;
y = y(+1) - sigma^(-1)*(i - pi(+1) - rho);
pi = beta*pi(+1) + k*y;
i = rho+phi_pi*pi+phi_y*y+e;
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



