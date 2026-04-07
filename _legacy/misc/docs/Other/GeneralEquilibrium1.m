% Non-Linear Optimization 
% 2-Sector Model of Production with Exogenous Growth
clc; close all;

% Planner Problem
% Max U(X1,X2) s.t. F1(L1,K1) = X1,  F2(L2,K2) = X2, L1+L2=L, K1+K2=K

%Params
a = 0.8 %labor share in sector 1 "Handicraft"
b = 0.2 %labor share in sector 2 "IT"
c = 0.3 %spend share for good 1 "Instagram Addiction"
Lmax = 1; %total labor supply
Kmax = 1; %total capital supply

% Simulations
N = 100;
Lmax = 1*1.01.^(0:N-1);
Kmax = 1*1.02.^(0:N-1);
X = zeros(N, 5)
for i = 1:N
    [X(i, 1:4), X(i, 5)] = GEsolve(Lmax(i), Kmax(i), a, b, c);
end

figure(1)
hold on;
plot(X(:,1),'-x')
plot(X(:,2),'-o')
legend('Handicraft', 'IT')
title('Labor Over Time')
hold off;
figure(2)
hold on;
plot(X(:,3),'-x')
plot(X(:,4),'-o')
legend('Handicraft', 'IT')
title('Capital Over Time')
hold off;
figure(3)
hold on;
plot(X(:,5),'-o')
legend('Utility Over Time')
hold off;

function [x, utility] = GEsolve(Lmax, Kmax, a, b, c)
F1 = @(L, K, a) L^a*K^(1-a);
F2 = @(L, K, b) L^b*K^(1-b);
U = @(x) - F1(x(1),x(3), a)^c * F2(x(2),x(4),b)^(1-c);
x0 = [0,0,0,0];
lb = [0,0,0,0];
ub = [Lmax,Lmax,Kmax,Kmax];
Aeq = [1,1,0,0; 0,0,1,1];
beq = [Lmax ; Kmax];
[x,fval] = fmincon(U,x0,[],[],Aeq,beq,lb,ub);
utility = -fval;
end