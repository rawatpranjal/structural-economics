%Optimized for speed by SeHyoun Ahn

clear all; clc; close all;

tic;

s = 2;
rho = 0.05;
z1 = .1;
z2 = .2;
z = [z1,z2];
la1 = 1.2;
la2 = 1.2;
la = [la1,la2];

r0 = 0.03;
rmin = 0.01;
rmax = 0.04;

I= 1000;
amin = -0.15;
amax = 5;
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

Ir = 40;
crit_S = 10^(-5);


%INITIAL GUESS
r = r0;
v0(:,1) = (z(1) + r.*a).^(1-s)/(1-s)/rho;
v0(:,2) = (z(2) + r.*a).^(1-s)/(1-s)/rho;

for ir=1:Ir;

r_r(ir)=r;
rmin_r(ir)=rmin;
rmax_r(ir)=rmax;
    
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
    cf = max(dVf,10^(-10)).^(-1/s);
    ssf = zz + r.*aa - cf;
    %consumption and savings with backward difference
    cb = max(dVb,10^(-10)).^(-1/s);
    ssb = zz + r.*aa - cb;
    %consumption and derivative of value function at steady state
    c0 = zz + r.*aa;
    
    % upwind method makes a choice of forward or backward differences based on
    % the sign of the drift    
    If = ssf > 0; %positive drift --> forward difference
    Ib = ssb < 0; %negative drift --> backward difference
    I0 = (1-If-Ib); %at steady state
    
    c = cf.*If + cb.*Ib + c0.*I0;
    u = c.^(1-s)/(1-s);
    
    %CONSTRUCT MATRIX
    X = -min(ssb,0)/da;
    Y = -max(ssf,0)/da + min(ssb,0)/da;
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

S(ir) = g(:,1)'*a*da + g(:,2)'*a*da;

%UPDATE INTEREST RATE
if S(ir)>crit_S
    disp('Excess Supply')
    rmax = r;
    r = 0.5*(r+rmin);
elseif S(ir)<-crit_S;
    disp('Excess Demand')
    rmin = r;
    r = 0.5*(r+rmax);
elseif abs(S(ir))<crit_S;
    display('Equilibrium Found, Interest rate =')
    disp(r)
    break
end

end

amax1 = 0.6;
amin1 = amin-0.03;

figure(1)
set(gca,'FontSize',16)
h1 = plot(a,adot(:,1,ir),'b',a,adot(:,2,ir),'r',linspace(amin1,amax1,I),zeros(1,I),'k--','LineWidth',2)
legend(h1,'s_1(a)','s_2(a)','Location','NorthEast')
text(-0.155,-.105,'$\underline{a}$','FontSize',16,'interpreter','latex')
line([amin amin], [-.1 .08],'Color','Black','LineStyle','--')
xlabel('Wealth, $a$','interpreter','latex')
ylabel('Savings, $s_i(a)$','interpreter','latex')
xlim([amin1 amax1])

figure(2)
set(gca,'FontSize',16)
h1 = plot(a,g_r(:,1,ir),'b',a,g_r(:,2,ir),'r','LineWidth',2)
legend(h1,'g_1(a)','g_2(a)')
text(-0.155,-.12,'$\underline{a}$','FontSize',16,'interpreter','latex')
line([amin amin], [0 max(max(g_r(:,:,ir)))],'Color','Black','LineStyle','--')
xlabel('Wealth, $a$','interpreter','latex')
ylabel('Densities, $g_i(a)$','interpreter','latex')
xlim([amin1 amax1])
ylim([0 3])