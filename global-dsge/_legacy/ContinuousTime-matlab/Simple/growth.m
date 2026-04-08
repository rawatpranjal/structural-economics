%% This is a fixed point iteration loop to solve the Hamilton Jacobi BellmanPDE
% for the Neoclassical Growth Model
% Written by Benjamin Moll
% Based on code by Adam Oberman, using theory from
% Convergent Difference Schemes for Nonlinear Elliptic and Parabolic Equations:
% Hamilton-Jacobi Equations and Free Boundary Problems,
% SIAM Journal on Numerical Analysis, Vol 44 (2006)

clear all; clc;

tic;

s = 2;
a = 0.3;
d = 0.05;
r = 0.05;
A = 1;

kss = (a*A/(r+d))^(1/(1-a));

I=150;
kmin = 0.001*kss;
kmax = 2*kss;
k = linspace(kmin,kmax,I)';
dk = (kmax-kmin)/(I-1);

maxit=10000;
crit = 10^(-6);

dVf = zeros(I,1);
dVb = zeros(I,1);
c = zeros(I,1);

%INITIAL GUESS
v0 = (A.*k.^a).^(1-s)/(1-s)/r;
v = v0;

maxit=10000;
for n=1:maxit
    V = v;
    % forward difference
    dVf(1:I-1) = (V(2:I)-V(1:I-1))/dk;
    dVf(I) = 0; %will never be used
    % backward difference
    dVb(2:I) = (V(2:I)-V(1:I-1))/dk;
    dVb(1) = 0; %will never be used
    
    I_concave = dVb > dVf; %indicator whether value function is concave (problems arise if this is not the case)
    
    %consumption and savings with forward difference
    cf = dVf.^(-1/s);
    muf = A.*k.^a - d.*k - cf;
    %consumption and savings with backward difference
    cb = dVb.^(-1/s);
    mub = A.*k.^a - d.*k - cb;
    %consumption and derivative of value function at steady state
    c0 = A.*k.^a - d.*k;
    dV0 = c0.^(-s);
    
    % dV_upwind makes a choice of forward or backward differences based on
    % the sign of the drift    
    If = muf > 0; %below steady state
    Ib = mub < 0; %above steady state
    I0 = (1-If-Ib); %at steady state
    %make sure the right approximations are used at the boundaries
    Ib(1) = 0; If(1) = 1; Ib(I) = 1; If(I) = 0;
    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; %important to include third term
    
    c = dV_Upwind.^(-1/s);
    Vchange = c.^(1-s)/(1-s) + dV_Upwind.*(A.*k.^a - d.*k - c) - r.*V;
       
    %% This is the update
    % the following CFL condition seems to work well in practice
    Delta = .9*dk/max(A.*k.^a - d.*k);
    v = v + Delta*Vchange;
    
    dist(n) = max(abs(Vchange));
    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
toc;

% Graphs
set(gca,'FontSize',14)
plot(dist,'LineWidth',2)
grid
xlabel('Iteration')
ylabel('||V^{n+1} - V^n||')

Verr = c.^(1-s)/(1-s) + dV_Upwind.*(A.*k.^a - d.*k - c) - r.*V;

set(gca,'FontSize',14)
plot(k,Verr,'LineWidth',2)
grid
xlabel('k')
ylabel('Error in HJB Equation')
xlim([kmin kmax])

kdot = A.*k.^a - d.*k - c;

set(gca,'FontSize',12)
plot(k,V,'LineWidth',2)
grid
xlabel('k')
ylabel('V(k)')
xlim([kmin kmax])

set(gca,'FontSize',14)
plot(k,c,'LineWidth',2)
grid
xlabel('k')
ylabel('c(k)')
xlim([kmin kmax])

set(gca,'FontSize',14)
plot(k,kdot,k,zeros(1,I),'--','LineWidth',2)
grid
xlabel('$k$','FontSize',16,'interpreter','latex')
ylabel('$s(k)$','FontSize',16,'interpreter','latex')
xlim([kmin kmax])
print -depsc HJB_NGM.eps