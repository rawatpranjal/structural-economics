% Infinite Horizon Dynamic Programming: 
% Cake Eating
clc; clear all;close all;

% x: cake left at period t
% V(x) = max u(x - x') + b * V(x')
% s.t. 0 <= x' <= x

% Params
beta = 0.9;
u = @(c) log(c);

% State and Control Space
NX = 1000;
NY = 1000;
NJ = 500;
lambda = 10; % sampling parameter
MAX = 1.0;
MIN = 0.01;
state = linspace(MIN^(1/lambda), MAX^(1/lambda), NX).^lambda;
control = linspace(MIN^(1/lambda), MAX^(1/lambda), NY).^lambda;

% Utility, Policy and Value Functions
U = zeros(NX, NY);
V = zeros(NX, NJ);
V(:, NJ) = u(state);
Y = zeros(NX, NJ);

% Utility
for ix = 1:NX
    for iy = 1:NY 
        if control(iy) < state(ix)
            % Feasible Controls
            U(ix, iy) = u(state(ix) - control(iy));
        else
            % Infeasible Controls
            U(ix, iy) = log(1e-30);
        end
    end
end

% Value Function Iteration
for ij = NJ-1:-1:1  
    for ix = 1:NX 
        [V(ix, ij), iymax] = max(reshape(U(ix, :),[1,NY]) + beta * reshape(V(:, ij+1), [1,NY]));
        Y(ix, ij) = control(iymax);
    end 
end

% Analytical Solution
V_true = @(x) (1/(1-beta))*log((1-beta).*x) + beta*log(beta)/((1-beta)^2);
Y_true = @(x) beta.*x;

hold on;
figure(1);
plot(state, V_true(state), '-o');
plot(state, V(:,1), '-x');
title('Value Function')
hold off;
figure(2);
hold on; 
plot(state, Y_true(state), '-o');
plot(state, Y(:,1), '-x');
title('Policy Function')
hold off;

% Simulation
x0 = 0.5;
T = 10;
xSim = zeros(1, T);
[val,idx] = min(abs(control-x0));
xSim(1) = control(idx);
for t = 1:T
    [val,idx] = min(abs(control-xSim(t)));
    xSim(t+1) = Y(idx, 1);
end
figure(3)
plot(xSim, '-x');
title('Cake over time')


