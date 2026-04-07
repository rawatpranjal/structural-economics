clear all; clc;

tic;

s = 2;
r = 0.045;
rho = 0.05;
w = .1;

I=500;
amin = -0.02;
amax = 1;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

maxit=20000;
crit = 10^(-6);

dVf = zeros(I,1);
dVb = zeros(I,1);
c = zeros(I,1);

%INITIAL GUESS
v0 = (w + r.*a).^(1-s)/(1-s)/rho;
v = v0;

for n=1:maxit
    V = v;
    % forward difference
    dVf(1:I-1) = (V(2:I)-V(1:I-1))/da;
    dVf(I) = 0; %will never be used
    % backward difference
    dVb(2:I) = (V(2:I)-V(1:I-1))/da;
    dVb(1) = (w + r.*amin).^(-s); %state constraint boundary condition
    
    I_concave = dVb > dVf; %indicator whether value function is concave (problems arise if this is not the case)
    
    %consumption and savings with forward difference
    cf = dVf.^(-1/s);
    muf = w + r.*a - cf;
    %consumption and savings with backward difference
    cb = dVb.^(-1/s);
    mub = w + r.*a - cb;
    %consumption and derivative of value function at steady state
    c0 = w + r.*a;
    dV0 = c0.^(-s);
    
    % dV_upwind makes a choice of forward or backward differences based on
    % the sign of the drift    
    If = muf > 0; %positive drift --> forward difference
    Ib = mub < 0; %negative drift --> backward difference
    I0 = (1-If-Ib); %at steady state
    %make sure the right approximations are used at the boundaries
    %STATE CONSTRAINT: USE BOUNDARY CONDITION UNLESS muf > 0
    Ib(I) = 1; If(I) = 0;
    
    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; %important to include third term
    
    c = dV_Upwind.^(-1/s);
    Vchange = c.^(1-s)/(1-s) + dV_Upwind.*(w + r.*a - c) - rho.*V;
       
    %% This is the update
    % the following CFL condition seems to work well in practice
    Delta = .9*da/max(w + r.*a);
    v = v + Delta*Vchange;
    
    dist(n) = max(abs(Vchange));
    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
toc;

% %CHECK THAT CONSTRAINT BINDS
% bind = (w + r*amin)^(-s) - dVf(1)

% Graphs
set(gca,'FontSize',14)
plot(dist,'LineWidth',2)
grid
xlabel('Iteration')
ylabel('||V^{n+1} - V^n||')

Verr = c.^(1-s)/(1-s) + dVb.*(w + r.*a - c) - rho.*V;

set(gca,'FontSize',14)
plot(a,Verr,'LineWidth',2)
grid
xlabel('k')
ylabel('Error in HJB Equation')
xlim([amin amax])

adot = w + r.*a - c;

set(gca,'FontSize',12)
plot(a,V,'LineWidth',2)
grid
xlabel('a')
ylabel('V(a)')
xlim([amin amax])

set(gca,'FontSize',14)
plot(a,c,'LineWidth',2)
grid
xlabel('a')
ylabel('c(a)')
xlim([amin amax])

set(gca,'FontSize',14)
plot(a,adot,a,zeros(1,I),'--','LineWidth',2)
grid
xlabel('a')
ylabel('s(a)')
xlim([amin amax])

%Approximation at borrowing constraint

u1 = (w+r*amin)^(-s); u2 = -s*(w+r*amin)^(-s-1);
nu = sqrt(-2*(rho-r)*u1/u2);
s_approx = -nu*(a-amin).^(1/2);

set(gca,'FontSize',14)
plot(a,adot,a,zeros(1,I),'--',a,s_approx,'-.','LineWidth',2)
grid
xlabel('a')
ylabel('s(a)')
xlim([amin amax])