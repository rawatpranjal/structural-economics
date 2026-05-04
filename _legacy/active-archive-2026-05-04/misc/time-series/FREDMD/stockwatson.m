% Lifecycle model
% written by Benjamin Moll
clear all; clc; close all;

tic

%--------------------------------------------------
%PARAMETERS
ga = 2;       % CRRA utility with parameter gamma
sig2 = (0.8)^2;  % sigma^2 O-U
Corr = exp(-0.9);  % persistence -log(Corr) O-U
rho = 0.05;   % discount rate
r = 0.035;
w = 1;

zmean = exp(sig2/2);

J=15;         % number of z points 
zmin = 0.75;   % Range z
zmax = 2.5;
amin = 0;    % borrowing constraint
amax = 100;    % range a
I=300;        % number of a points 

T=75;       %maximum age
N=300;      %number of age steps
%N=75;      %number of age steps
%N=10;      %number of age steps
dt=T/N;

%simulation parameters
maxit  = 100;     %maximum number of iterations in the HJB loop
crit = 10^(-10); %criterion HJB loop

%ORNSTEIN-UHLENBECK IN LOGS
the = -log(Corr);

%--------------------------------------------------
%VARIABLES 
a = linspace(amin,amax,I)';  %wealth vector
da = (amax-amin)/(I-1);      
z = linspace(zmin,zmax,J);   % productivity vector
dz = (zmax-zmin)/(J-1);
dz2 = dz^2;
aa = a*ones(1,J);
zz = ones(I,1)*z;

mu = -the.*z.*log(z)+sig2/2*z;   %DRIFT (FROM ITO'S LEMMA)
s2 = sig2.*z.^2;                   %VARIANCE (FROM ITO'S LEMMA)

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J);             
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

%CONSTRUCT MATRIX Aswitch SUMMARIZING EVOLUTION OF z
yy = - s2/dz2 - mu/dz;
chi =  s2/(2*dz2);
zeta = mu/dz + s2/(2*dz2);

%This will be the upperdiagonal of the matrix Aswitch
updiag=zeros(I,1); %This is necessary because of the peculiar way spdiags is defined.
for j=1:J
    updiag=[updiag;repmat(zeta(j),I,1)];
end

%This will be the center diagonal of the matrix Aswitch
centdiag=repmat(chi(1)+yy(1),I,1);
for j=2:J-1
    centdiag=[centdiag;repmat(yy(j),I,1)];
end
centdiag=[centdiag;repmat(yy(J)+zeta(J),I,1)];

%This will be the lower diagonal of the matrix Aswitch
lowdiag=repmat(chi(2),I,1);
for j=3:J
    lowdiag=[lowdiag;repmat(chi(j),I,1)];
end

%Add up the upper, center, and lower diagonal into a sparse matrix
Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);


%Preallocation
v = zeros(I,J,N);
gg = cell(N+1,1);
maxit = 1000;
convergence_criterion = 10^(-5);

% terminal condition on value function: value of death \approx 0
small_number1 = 10^(-8); small_number2 = 10^(-8);
v_terminal = small_number1*(small_number2 + aa).^(1-ga)/(1-ga);

V = v_terminal;
    
    for n=N:-1:1
        disp(['age = ', num2str(n*dt)])
        v(:,:,n)=V;
        % forward difference
        dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVf(I,:) = (w*z + r.*amax).^(-ga); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVb(1,:) = (w*z + r.*amin).^(-ga); %state constraint boundary condition
        
        %consumption and savings with forward difference
        cf = dVf.^(-1/ga);
        ssf = w*zz + r.*aa - cf;
        %consumption and savings with backward difference
        cb = dVb.^(-1/ga);
        ssb = w*zz + r.*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = w*zz + r.*aa;
        
        %upwind method
        If = ssf > 0; %positive drift --> forward difference
        Ib = ssb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
        
        c = cf.*If + cb.*Ib + c0.*I0;
        u = c.^(1-ga)/(1-ga);
        
        %CONSTRUCT MATRIX
        X = - min(ssb,0)/da;
        Y = - max(ssf,0)/da + min(ssb,0)/da;
        Z = max(ssf,0)/da;
        
        updiag=[0];
        for j=1:J
            updiag=[updiag;Z(1:I-1,j);0];
        end
        
        centdiag = reshape(Y,I*J,1);
        
        lowdiag=X(2:I,1);
        for j=2:J
            lowdiag=[lowdiag;0;X(2:I,j)];
        end
        
        A=Aswitch+spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);

       if max(abs(sum(A,2)))>10^(-9)
           disp('Improper Transition Matrix')
           break
       end
       
        %%Note the syntax for the cell array
        A_t{n} = A;
        B = (1/dt + rho)*speye(I*J) - A;
        
        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);
        
        b = u_stacked + V_stacked/dt;
        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS
        
        V = reshape(V_stacked,I,J);
        c_t{n} = c;
        ss_t{n} = w*zz + r.*aa - c;
    end

toc
    
plot(a,c_t{1})
plot(a,ss_t{1},a,zeros(I,1),'k--')

plot(a,c_t{N-1})
plot(a,ss_t{N-1},a,zeros(I,1),'k--')

subplot(1,2,1)
set(gca,'FontSize',16)
plot(a,c_t{1/dt}(:,1),a,c_t{1/dt}(:,J),a,c_t{40/dt}(:,1),a,c_t{40/dt}(:,J),a,c_t{70/dt}(:,1),a,c_t{70/dt}(:,J),'Linewidth',2)
legend('Age 1, Lowest Income','Age 1, Highest Income','Age 40, Lowest Income','Age 40, Highest Income','Age 70, Lowest Income','Age 70, Highest Income')
ylim([0 5])
xlabel('Wealth')
ylabel('Consumption, c(a,z,t)')

subplot(1,2,2)
set(gca,'FontSize',16)
plot(a,ss_t{1/dt}(:,1),a,ss_t{1/dt}(:,J),a,ss_t{40/dt}(:,1),a,ss_t{40/dt}(:,J),a,ss_t{70/dt}(:,1),a,ss_t{70/dt}(:,J),a,zeros(I,1),'k--','Linewidth',2)
legend('Age 1, Lowest Income','Age 1, Highest Income','Age 40, Lowest Income','Age 40, Highest Income','Age 70, Lowest Income','Age 70, Highest Income')
ylim([-5 2])
xlabel('Wealth')
ylabel('Saving, s(a,z,t)')

print -depsc lifecycle.eps