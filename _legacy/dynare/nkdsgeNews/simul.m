se = 0.01;
m = 2; % number of states
n = 3; % number of controls
psi = oo_.dr.ghx;
omega = oo_.dr.ghu;
ss = oo_.dr.ys;
T = 200;
es = se*randn(T,1);
Xsim = zeros(n+m,T);
Xsim(:,1) = omega*es(1,1);
for j=2:T
	Xsim(:,j) = psi*Xsim(3:4,j-1) + omega*es(j,1);
end
for j=1:T
	Xsim(:,j) = Xsim(:,j) + ss;
end

plot(Xsim(1,:));
plot(Xsim(2,:));
plot(Xsim(3,:));
plot(Xsim(4,:));
plot(Xsim(5,:));

