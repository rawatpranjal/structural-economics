% Infinite Horizon Stochastic Dynamic Programming: 
% Optimal Growth with IID Shocks
clc; clear all;close all;

% x: Cake
% z: Endowment Shock
% V(k,z) = max u(F(k) + z - k')+ b * E[V(k', z')|z]
% s.t. 0 <= k' <= F(k') + z

% Params
alpha = 0.3;
A = 18.5;
beta = 0.9;
u = @(c) log(c);
f = @(k) A*k.^alpha;
P = [0.4 0.3 0.2 0.1];
Z = [2, 4, 8, 10];

% State and Control Space
NX = 500;
NY = 500;
NZ = 4; 
NJ = 100;
lambda = 1; % sampling parameter
MAX = 20;
MIN = MAX/NX;
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
        if control(iy) < f(Xstate(ix))+Zstate(iz)
            % Feasible Controls
            U(ix, iz, iy) = u(f(Xstate(ix))+Zstate(iz)-control(iy));
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
         [V(ix, iz, ij), iymax] = max(squeeze(U(ix, iz, :)) + beta*(0.4*V(:, 1, ij+1)+0.3*V(:, 2, ij+1)+0.2*V(:, 3, ij+1)+0.1*V(:, 4, ij+1)));    
        %[V(ix, iz, ij), iymax] = max(reshape(U(ix, iz, :),[1,NY]) + beta*reshape(dot(repmat(P,NX,1),V(:, :, ij+1),2), [1,NY]));    
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
legend('z=0.1', 'z=0.2', 'z=0.3', 'z=0.4')
hold off; 

% Policy Functions
figure(2);
hold on;
for i = 1:NZ
    plot(Xstate, Y(:, i, 1), '-x');
end
title('Policy Functions')
legend('z=0.1', 'z=0.2', 'z=0.3', 'z=0.4')
hold off; 

% Simulations
x0 = 0.2;
T = 15;
zSim = datasample(Z,T,'Weights',P)
xSim = Simulate(zSim, x0, control, Zstate, Y)

figure(3);
hold on;
plot(zSim, '-o');
plot(xSim, '-x');
legend('Income', 'Capital');
title('Simulation');
hold off;

function xSim = Simulate(zSim, x0, control, Zstate, Y)
T = length(zSim)
xSim = zeros(1, T);
[val,idx] = min(abs(control-x0));
xSim(1) = control(idx);
for t = 1:T
    [val,idz] = min(abs(Zstate-zSim(t)));
    [val,idx] = min(abs(control-xSim(t)));
    xSim(t+1) = Y(idx,idz,1);
end
end
