% Infinite Horizon Deterministic Dynamic Programming: 
% Optimal Growth
clc; clear all;close all;

% k: capital at period t
% V(k) = max u(F(k) - k')+ b * V(k')
% s.t. 0 <= k' <= F(k')

% Params
alpha = 0.3;
A = 18.5;
beta = 0.9;
u = @(c) log(c);
f = @(k) A*k.^alpha;

% State and Control Space
NX = 1000;
NY = 1000;
NJ = 100;
lambda = 2; % sampling parameter
MAX = 20
MIN = MAX/NX
state = linspace(MIN^(1/lambda), MAX^(1/lambda), NX).^lambda;
control = linspace(MIN^(1/lambda), MAX^(1/lambda), NY).^lambda;

% Utility, Policy and Value Functions
U = zeros(NX, NY);
V = zeros(NX, NJ);
Y = zeros(NX, NJ);

% Utility
for ix = 1:NX
    for iy = 1:NY 
        if control(iy) < f(state(ix))
            % Feasible Controls
            U(ix, iy) = u(f(state(ix)) - control(iy));
        else
            % Infeasible Controls
            U(ix, iy) = -10000;
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
E = (1/(1-beta))*(log(A*(1-alpha*beta))+beta*alpha*log(A*alpha*beta)/(1-alpha*beta));
F = alpha/(1-alpha*beta);
V_true = @(k) E + F*log(k);
Y_true = @(k) alpha*beta*A*k.^alpha;

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
x0 = 1;
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



