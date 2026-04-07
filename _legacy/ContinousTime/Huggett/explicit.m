clear all; clc;

tic;

s = 2; %CRRA utility with parameter s
r = 0.03; %interest rate
rho = 0.05; %discount rate
z1 = .1;
z2 = .2;
z = [z1,z2];
la1 = 0.02; %lambda_1
la2 = 0.03; %lambda_2
la = [la1,la2];


I=500;
amin = -0.02; %borrowing constraint
amax = 2;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

aa = [a,a];
zz = ones(I,1)*z;

maxit= 20000;
crit = 10^(-6);

dVf = zeros(I,2);
dVb = zeros(I,2);
c = zeros(I,2);

%INITIAL GUESS
v0(:,1) = (z(1) + r.*a).^(1-s)/(1-s)/rho;
v0(:,2) = (z(2) + r.*a).^(1-s)/(1-s)/rho;
v = v0;

for n=1:maxit
    V = v;
    % forward difference
    dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
    dVf(I,:) = 0; %will never be used
    % backward difference
    dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
    dVb(1,:) = (z + r.*amin).^(-s); %state constraint boundary condition
    
    I_concave = dVb > dVf; %indicator whether value function is concave (problems arise if this is not the case)
    
    %consumption and savings with forward difference
    cf = dVf.^(-1/s);
    muf = zz + r.*aa - cf;
    %consumption and savings with backward difference
    cb = dVb.^(-1/s);
    mub = zz + r.*aa - cb;
    %consumption and derivative of value function at steady state
    c0 = zz + r.*aa;
    dV0 = c0.^(-s);
    
    % dV_upwind makes a choice of forward or backward differences based on
    % the sign of the drift    
    If = muf > 0; %positive drift --> forward difference
    Ib = mub < 0; %negative drift --> backward difference
    I0 = (1-If-Ib); %at steady state
    %make sure backward difference is used at amax
    Ib(I,:) = 1; If(I,:) = 0;
    %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
    %already taken care of automatically
    
    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; %important to include third term
    
    c = dV_Upwind.^(-1/s);
    V_switch = [V(:,2),V(:,1)];
    Vchange = c.^(1-s)/(1-s) + dV_Upwind.*(zz + r.*aa - c) + ones(I,1)*la.*(V_switch - V) - rho.*V;
       
    %% This is the update
    % the following CFL condition seems to work well in practice
    Delta = .9*da/max(z2 + r.*a);
    v = v + Delta*Vchange;
    
    dist(n) = max(max(abs(Vchange)));
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

Verr = c.^(1-s)/(1-s) + dV_Upwind.*(zz + r.*aa - c) + ones(I,1)*la.*(V_switch - V) - rho.*V;

set(gca,'FontSize',14)
plot(a,Verr,'LineWidth',2)
grid
xlabel('k')
ylabel('Error in HJB Equation')
xlim([amin amax])

adot = zz + r.*aa - c;

set(gca,'FontSize',12)
plot(a,V,'LineWidth',2)
grid
xlabel('a')
ylabel('V_i(a)')
xlim([amin amax])

set(gca,'FontSize',14)
plot(a,c,'LineWidth',2)
grid
xlabel('a')
ylabel('c_i(a)')
xlim([amin amax])

% set(gca,'FontSize',14)
% plot(a,adot,a,zeros(1,I),'--','LineWidth',2)
% grid
% xlabel('a')
% ylabel('s_i(a)')
% xlim([amin amax])

%Approximation at borrowing constraint
u1 = (z1+r*amin)^(-s); u2 = c(1,2)^(-s); u11 = -s*(z1+r*amin)^(-s-1);
nu = sqrt(-2*((rho-r)*u1 + la1*(u1 - u2))/u11);
s_approx = -nu*(a-amin).^(1/2);

set(gca,'FontSize',14)
h1 = plot(a,adot,a,s_approx,'-.',a,zeros(1,I),'--','LineWidth',2)
legend(h1,'s_1(a)','s_2(a)','Approximate s_1(a)','Location','SouthWest')
grid
xlabel('a')
ylabel('s_i(a)')
xlim([amin amax])
%print -depsc HJB_stateconstraint.eps