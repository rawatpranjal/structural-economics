var Y N C Pi pi i r p w W A Z v;
varexo eps_a eps_z eps_v;
parameters alpha sigma phi rho rho_a rho_z eta;


model;

a = rho_a*a(-1) + eps_a; % tech
z = rho_z*z(-1) + eps_z; % pref
v = rho_v*v(-1) + eps_v; % mon

y = a + (1-alpha)*n; % prodn
w = p + a - alpha * n + log(1-alpha); %ld
w = p - sigma * c + phi * n; %ls
c = c(+1) - (1/sigma) * (r + rho) + (1/sigma)*(1-rho_z)*z; %IS/Euler
m = p + y - eta * i; % money demand
i = rho + pi + v; %monpol

y = c; %eqbm
r = i - Pi(+1); %fisher
Pi = p - p(-1); %inf
W = w - p; %realwage

end;

shocks;
var eps_a; stderr 1;
var eps_z; stderr 1;
var eps_v; stderr 1;
end;

steady_state_model;
a = 1;
z = 1;
v = 0;
y = 0;
n = 0;
w