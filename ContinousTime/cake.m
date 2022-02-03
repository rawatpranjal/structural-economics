
% max summa ln(c(t))exp(-rho*t) s.t. dx(t)/dt = -c(t), x(0)=x0, lim x(t)>=0. 
rho = 2;
x0 = 1;

% Analytical Soln
syms z(t);
ode = diff(z,t,1) == 1-2*z(t);
initial = z(0) == 1;
z(t) = dsolve(ode);

% Analytical Soln
syms z(t);
ode = diff(z,t,1) == -1+rho*z(t);
initial = x(0) == x0;
terminal = x(100) == 0;
z(t) = dsolve(ode);