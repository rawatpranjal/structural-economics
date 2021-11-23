% Infinite Horizon Deterministic Dynamic Programming: 
% Consumption-Saving with Fluctuating Income
clc; clear all;close all;

% x: Savings
% z: Income
% V(x, z) = max u(x(1+r) + z - x') + b * V(x', z')
% s.t. 0 <= x' <= x(1+r) + z

% Params
beta = 0.9;
r = 0.05;
u = @(c) log(c);

% State and Control Space
NX = 100;
NY = 100;
NZ = 100; 
NJ = 100;
lambda = 5; % sampling parameter
MAX = 2.0;
MIN = 0.01;
Xstate = linspace(MIN^(1/lambda), MAX^(1/lambda), NX).^lambda;
Zstate = linspace(MIN^(1/lambda), MAX^(1/lambda), NX).^lambda;
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
            U(ix, iz, iy) = log(1e-30);
        end
    end
    end
end

% Value Function Iteration
for ij = NJ-1:-1:1  
    for ix = 1:NX 
        for iz = 1:NZ
        [V(ix, iz, ij), iymax] = max(reshape(U(ix, iz, :),[1,NY]) + beta * reshape(V(ix, :, ij+1), [1,NY]));
        Y(ix, iz, ij) = control(iymax);
        end 
    end
end

hold on;
figure(1)
mesh(Xstate, Zstate, V(:, :, 1))
title('Value Surface')

figure(2)
mesh(Xstate, Zstate,Y(:, :, 1))
title('Policy Surface')
hold off;

% Simulation
x0 = 0.5;
T = 10;
zSim = ones(100)*0.1
T = length(zSim)
xSim = zeros(1, T);
[val,idx] = min(abs(control-x0));
xSim(1) = control(idx);
for t = 1:T
    [val,idz] = min(abs(Zstate-zSim(t)));
    [val,idx] = min(abs(control-xSim(t)));
    xSim(t+1) = Y(idx,idz,1);
end
figure(3)
hold on;
plot(zSim, '-x');
plot(xSim, '-x');
title('Cake over time')
hold off;


