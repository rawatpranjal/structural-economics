%Optimized for speed by SeHyoun Ahn

clear all; clc; close all;

tic;

s = 2;
rho = 0.05;
z1 = .1;
z2 = .2;
z = [z1,z2];
la1 = 0.001;
la2 = 0.001;
% la1 = 0.02;
% la2 = 0.03;
la1 = 1.5;
la2 = 1;
la = [la1,la2];


I=500;
amin = -0.15;
amax = 10;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

aa = [a,a];
zz = ones(I,1)*z;


maxit= 100;
crit = 10^(-6);
Delta = 40;

dVf = zeros(I,2);
dVb = zeros(I,2);
c = zeros(I,2);

Aswitch = [-speye(I)*la(1),speye(I)*la(1);speye(I)*la(2),-speye(I)*la(2)];

Ir = 20;
% rmin = 0.00001;
rmin = -0.05;
rmax = 0.04;
r_grid = linspace(rmin,rmax,Ir);

%INITIAL GUESS
r = r_grid(1);
% v0(:,1) = (z(1) + r.*a).^(1-s)/(1-s)/rho;
% v0(:,2) = (z(2) + r.*a).^(1-s)/(1-s)/rho;
v0(:,1) = (z(1) + max(r,0.01).*a).^(1-s)/(1-s)/rho;
v0(:,2) = (z(2) + max(r,0.01).*a).^(1-s)/(1-s)/rho;

for ir=1:Ir;

r = r_grid(ir);

if ir>1
v0 = V_r(:,:,ir-1);
end

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

g_r(:,:,ir) = g;
adot(:,:,ir) = zz + r.*aa - c;
V_r(:,:,ir) = V;
dV_r(:,:,ir) = dV_Upwind;
c_r(:,:,ir) = c;

S(ir) = g(:,1)'*a*da + g(:,2)'*a*da;
end

% ir = 9;
% set(gca,'FontSize',14)
% h1 = plot(a,dV_r(:,:,ir),a,dV_r(:,:,ir+1),'LineWidth',2)
% legend(h1,'v_1\prime(a,r1)','v_2(a,r1)','v_1(a,r2)','v_2(a,r2)')
% grid
% xlabel('a')
% ylabel('s_i(a)')
% xlim([amin amax])


ir = 1;
set(gca,'FontSize',14)
h1 = plot(a,adot(:,:,ir),a,adot(:,:,ir+1),a,zeros(1,I),'--','LineWidth',2)
legend(h1,'s_1(a,r1)','s_2(a,r1)','s_1(a,r2)','s_2(a,r2)')
grid
xlabel('a')
ylabel('s_i(a)')
xlim([amin amax])

ir = 9;
set(gca,'FontSize',14)
h1 = plot(a,c_r(:,:,ir),a,c_r(:,:,ir+1),a,zeros(1,I),'--','LineWidth',2)
legend(h1,'c_1(a,r1)','c_2(a,r1)','c_1(a,r2)','c_2(a,r2)')
grid
xlabel('a')
ylabel('c_i(a)')
xlim([amin amax])

amax1 = .8;
set(gca,'FontSize',14)
h1 = plot(a,g_r(:,:,ir),'LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
grid
xlabel('a')
ylabel('g_i(a)')
xlim([amin amax1])

Smax = max(S);
amin1 = 1.1*amin;
aaa = linspace(amin1,Smax,Ir);
rrr = linspace(rmin,0.06,Ir);

set(gca,'FontSize',14)
plot(S,r_grid,zeros(1,Ir),rrr,zeros(1,Ir)+amin,rrr,'--',aaa,ones(1,Ir)*rho,'--','LineWidth',2)
text(-0.1,0.045,'$r = \rho$','FontSize',16,'interpreter','latex')
text(-0.05,0,'$S(r)$','FontSize',16,'interpreter','latex')
text(-0.145,0.01,'$a=\underline{a}$','FontSize',16,'interpreter','latex')
ylabel('$r$','FontSize',16,'interpreter','latex')
xlabel('$S(r)$','FontSize',16,'interpreter','latex')
ylim([rmin 0.06])
xlim([amin1 Smax])
print -depsc asset_supply.eps