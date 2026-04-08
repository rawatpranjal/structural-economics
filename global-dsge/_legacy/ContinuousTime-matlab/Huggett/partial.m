%Optimized for speed by SeHyoun Ahn

clear all; clc; close all;

tic;

s = 1.2;
r = 0.035;
% s = 0.5;
% r = 0.045;
rho = 0.05;
z1 = .1;
z2 = .2;
z = [z1,z2];
% la1 = 0.02;
% la2 = 0.03;
la1 = 1.5;
la2 = 1;
la = [la1,la2];


I=500;
amin = -0.02;
amax = 3;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

aa = [a,a];
zz = ones(I,1)*z;


maxit= 100;
crit = 10^(-6);
Delta = 1000;

dVf = zeros(I,2);
dVb = zeros(I,2);
c = zeros(I,2);

Aswitch = [-speye(I)*la(1),speye(I)*la(1);speye(I)*la(2),-speye(I)*la(2)];

%INITIAL GUESS
v0(:,1) = (z(1) + r.*a).^(1-s)/(1-s)/rho;
v0(:,2) = (z(2) + r.*a).^(1-s)/(1-s)/rho;

% z_ave = la2/(la1+la2)*z(1) + la1/(la1+la2)*z(2);
% v0(:,1) = (z_ave + r.*a).^(1-s)/(1-s)/rho;
% v0(:,2) = (z_ave + r.*a).^(1-s)/(1-s)/rho;

v = v0;

for n=1:maxit
    V = v;
    V_n(:,:,n)=V;
    % forward difference
    dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
    dVf(I,:) = (z + r.*amax).^(-s); %will never be used, but impose state constraint a<=amax just in case
    % backward difference
    dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
    dVb(1,:) = (z + r.*amin).^(-s); %state constraint boundary condition
    
    I_concave = dVb > dVf; %indicator whether value function is concave (problems arise if this is not the case)
    
    %consumption and savings with forward difference
    cf = dVf.^(-1/s);
    ssf = zz + r.*aa - cf;
    %consumption and savings with backward difference
    cb = dVb.^(-1/s);
    ssb = zz + r.*aa - cb;
    %consumption and derivative of value function at steady state
    c0 = zz + r.*aa;
    dV0 = c0.^(-s);
    
    % dV_upwind makes a choice of forward or backward differences based on
    % the sign of the drift    
    If = ssf > 0; %positive drift --> forward difference
    Ib = ssb < 0; %negative drift --> backward difference
    I0 = (1-If-Ib); %at steady state
    %make sure backward difference is used at amax
    %Ib(I,:) = 1; If(I,:) = 0;
    %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
    %already taken care of automatically
    
    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; %important to include third term
    c = dV_Upwind.^(-1/s);
    u = c.^(1-s)/(1-s);
    
    %CONSTRUCT MATRIX
    X = - min(ssb,0)/da;
    Y = - max(ssf,0)/da + min(ssb,0)/da;
    Z = max(ssf,0)/da;
    
    A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
    A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);

    A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;
    
    if max(abs(sum(A,2)))>10^(-9)
       disp('Improper Transition Matrix')
       break
    end
    
    B = (1/Delta + rho)*speye(2*I) - A;
    
    u_stacked = [u(:,1);u(:,2)];
    V_stacked = [V(:,1);V(:,2)];
    
    b = u_stacked + V_stacked/Delta;
    V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS
    
    V = [V_stacked(1:I),V_stacked(I+1:2*I)];
    
    Vchange = V - v;
    v = V;

    dist(n) = max(max(abs(Vchange)));
    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOKKER-PLANCK EQUATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%
AT = A';
b = zeros(2*I,1);

%need to fix one value, otherwise matrix is singular
i_fix = 1;
b(i_fix)=.1;
row = [zeros(1,i_fix-1),1,zeros(1,2*I-i_fix)];
AT(i_fix,:) = row;

%Solve linear system
gg = AT\b;
g_sum = gg'*ones(2*I,1)*da;
gg = gg./g_sum;

g = [gg(1:I),gg(I+1:2*I)];

check1 = g(:,1)'*ones(I,1)*da;
check2 = g(:,2)'*ones(I,1)*da;

adot = zz + r.*aa - c;
set(gca,'FontSize',14)
plot(a,adot,a,zeros(1,I),'--','LineWidth',2)
grid
xlabel('a')
ylabel('s_i(a)')
xlim([amin amax])

S = g(:,1)'*a*da + g(:,2)'*a*da;


%Approximation at borrowing constraint
u1 = (z1+r*amin)^(-s); u2 = c(1,2)^(-s); u11 = -s*(z1+r*amin)^(-s-1);
nu = sqrt(-2*((rho-r)*u1 + la1*(u1 - u2))/u11);
s_approx = -nu*(a-amin).^(1/2);

amax1 = 1;
set(gca,'FontSize',14)
h1 = plot(a,adot,a,s_approx,'-.',a,zeros(1,I),'--','LineWidth',2)
legend(h1,'s_1(a)','s_2(a)','Approximate s_1(a)','Location','SouthWest')
grid
xlabel('a')
ylabel('s_i(a)')
xlim([amin amax1])
print -depsc HJB_stateconstraint.eps

set(gca,'FontSize',14)
h1 = plot(a,g,'LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
grid
xlabel('a')
ylabel('g_i(a)')
xlim([amin amax1])
print -depsc densities.eps