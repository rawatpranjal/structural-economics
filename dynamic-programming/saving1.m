% Infinite Horizon Stochastic Dynamic Programming: 
% Consumption Saving with Markovian Income Shocks
clc; close all;

% x: Asset / Saving
% z: Income Shock
% V(x, z) = max u(x(1+r) + z - x') + b * E[V(x', z')|z]
% s.t. 0 <= x' <= x(1+r) + z

% Params
beta = 0.9;
r = 0.05;
u = @(c) log(c);

% NZ-state Markovian Income 
Z = [1, 2, 3, 4, 5, 6, 7, 8]/25;
NZ = length(Z); 
mc = mcmix(NZ); 
P = mc.P ;
zSim = simulate(mc, T+1)/25;

% State and Control Space
NX = 100;
NY = 100;
NJ = 200;
lambda = 1; % sampling parameter
MAX = 1.0;
MIN = 0.01;
Xstate = linspace(MIN^(1/lambda), MAX^(1/lambda), NX).^lambda;
Zstate = Z;
control = linspace(MIN^(1/lambda), MAX^(1/lambda), NY).^lambda;

% Utility, Policy and Value Functions
U = zeros(NX, NZ, NY);
V = zeros(NX, NZ, NJ);
Y = zeros(NX, NZ, NJ);

% Utility
for ix = 1:NX
    for iz = 1:NZ
        for iy = 1:NY 
        if control(iy) < Xstate(ix)*(1+r)+Zstate(iz)
            % Feasible Controls
            U(ix, iz, iy) = u(Xstate(ix)*(1+r)+Zstate(iz)-control(iy));
        else
            % Infeasible Controls
            U(ix, iz, iy) = -1000000000;
        end
    end
    end
end

% Value Function Iteration
for ij = NJ-1:-1:1  
    for ix = 1:NX 
        for iz = 1:NZ
        [V(ix, iz, ij), iymax] = max(squeeze(U(ix, iz, :)) + beta*(dot(repmat(P(iz,:), NY, 1) , V(:, :, ij+1),2)));    
        Y(ix, iz, ij) = control(iymax);
        end 
    end
end
    
% Value Functions
figure(1);
hold on;
for i = 1:NZ
    plot(Xstate, V(:, i, 1), '-x');
end
title('Value Functions')
hold off; 

% Policy Functions
figure(2);
hold on;
for i = 1:NZ
    plot(Xstate, Y(:, i, 1), '-x');
end
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
