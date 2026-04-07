var p y;
varexo z;
parameters beta gamma sigma rho;
beta = 0.9;
gamma = 2.0;
rho = 0.9;
sigma = 0.1;

model;
	y = exp(rho*log(y(-1))+sigma*z);
	p*y^(-gamma) = beta*y(+1)^(-gamma)*(p(+1)+y(+1));
end;

steady_state_model;
y=1;
p=y/(1/beta-1);
end;

initval;
y=1;
end;


resid;
steady;
check;

shocks;
var z; stderr 1;
end;

stoch_simul(order=1,periods=1000) y p;
rplot y;
rplot p;

	