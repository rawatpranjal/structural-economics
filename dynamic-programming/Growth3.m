% Infinite Horizon Deterministic Dynamic Programming: 
% One Sector Neoclassical Growth Model with endogenous labor and saving
clc; close all;

% Household Problem:
% V(k) = max u(c,l) + b * V(k')
% c + k' = W*l + R*k - k(1-d) = F(k, l) - k(1-d)
% c, k' > 0, l in [0,1]
% W = F2(k,l), R = F1(k,l), Y = F(k,l) = k^alpha*l^(1-alpha)

% Params
alpha = 0.3;
delta = 0.05;
beta = 0.9;
A = 18.5;
u = @(c,l) log(c)-l^2/2;
F = @(k,l) A*k^alpha*l^(1-alpha);
F1 = @(k,l) A*alpha*k^(alpha-1)*l^(1-alpha);
F2 = @(k,l) A*(1-alpha)*k^(alpha-1)*l^(1-alpha);

% State and Control Space
NK = 100;
NL = 100;
NJ = 200;
lambda = 1; % sampling parameter
MAX = 20;
MIN = 0.2;
Kstate = linspace(MIN^(1/lambda), MAX^(1/lambda), NK).^lambda;
Lstate = linspace(0.01^(1/lambda), 1^(1/lambda), NK).^lambda;

% Utility, Policy and Value Functions
U = zeros(NK, NL, NK);
V = zeros(NK, NJ);
KP = zeros(NK, NJ);
L = zeros(NL, NJ);

% Utility
for ik = 1:NX
    for il = 1:NL
        for ikp = 1:NK
        if (Kstate(ikp) < F(Kstate(ik),Lstate(il))-Kstate(ik)*(1-delta))&(Lstate(il)<=1)
            % Feasible Controls
            U(ik, il, ikp) = u(F(Kstate(ik),Lstate(il))-Kstate(ik)*(1-delta)-Kstate(ikp),Lstate(il));
        else
            % Infeasible Controls
            U(ik, il, ikp) = -1000000000;
        end
    end
    end
end

% Value Function Iteration
for ij = NJ-1:-1:1  
    for ik = 1:NK
        % V(k) = max u(c,l) + b * V(k')
        Vrhs = squeeze(U(ik, :, :)) + beta*repmat(V(:, ij+1), 1, NL)';
        val = max(max(Vrhs));
        [row col] = find(Vrhs == val);
        KP(ik, ij) = Kstate(col);
        L(ik, ij) = Lstate(row);
        end 
end
    
% Value Functions
figure(1);
hold on;
plot(Kstate, V(:, 1), '-x');
title('Value Functions')
hold off; 

% Policy Functions
figure(2);
plot(Kstate, KP(:, 1), '-x');
plot(Kstate, L(:, 1), '-x');

title('Policy Functions')
hold off; 

% Simulations
x0 = 0.5;
T = 100;
rng(1); 
zSim = simulate(mc, T)/25;
xSim, cSim = Simulate(zSim, x0, control, Zstate, Y, r)

figure(3);
hold on;
plot(zSim(2:T+1), '-o');
plot(xSim(2:T+1), '-x');
plot(cSim, '-');
legend('Income', 'Asset/Saving', 'Consumption');
title('Simulation');
hold off;

function [xSim, cSim] = Simulate(zSim, x0, control, Zstate, Y, r)
T = length(zSim);
xSim = zeros(1, T+1);
cSim = zeros(1, T);
[val,idx] = min(abs(control-x0));
xSim(1) = control(idx);
for t = 1:T
    [val,idz] = min(abs(Zstate-zSim(t)));
    [val,idx] = min(abs(control-xSim(t)));
    xSim(t+1) = Y(idx,idz,1);
    cSim(t+1) = xSim(t)*(1+r)+zSim(t)-xSim(t+1);
end
end
