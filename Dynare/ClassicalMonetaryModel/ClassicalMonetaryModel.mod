var C W Pi A N i R Y V dM Q Z;
varexo eps_a eps_z eps_v;
parameters alpha beta sigma gamma phi eta rho_a rho_z rho_v;

alpha = 1/4;
beta = 0.99;
rho_a = 0.9;
rho_z = 0.5;
rho_v = 0.5;
sigma = 1;
phi = 5;
gamma = 1.5;
eta = 3.77;

model;
W=C^sigma*N^phi;
Q=beta*(C(+1)/C)^(-sigma)*(Z(+1)/Z)*1/Pi(+1);
i=1/Q;
Y=A*N^(1-alpha);
W=(1-alpha)*A*N^(-alpha);
i=R*Pi(+1);
R=1/beta*Pi^gamma*exp(V);
C=Y;
log(A)=rho_a*log(A(-1))+eps_a;
log(Z)=rho_z*log(Z(-1))+eps_z;
V=rho_v*V(-1)+eps_v;
dM=4*(log(C)-log(C(-1))-eta*(log(i)-log(i(-1)))+log(Pi));
end;

shocks;
var eps_a; stderr 1;
var eps_z; stderr 1;
var eps_v; stderr 1;
end;

steady_state_model;
A=1;
Z=1;
i=1/beta;
Pi=1;
Q=1/i;
R=i;
N=(1-alpha)^(1/((1-sigma)*alpha+phi+sigma));
C=A*N^(1-alpha);
W=(1-alpha)*A*N^(-alpha);
Y=C;
dM=0;
end;


resid(1);
steady;
check;

write_latex_dynamic_model;
stoch_simul(irf=20,order=1) Y C Pi i R dM;