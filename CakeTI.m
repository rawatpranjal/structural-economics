% Cake Eating: VFI
clc; clear all;close all;

% x: cake left at period t
% c: cake eaten in period t
% Objective: max sum[t] b^t * u(c(t)) s.t. 0 < c(t), x(t+1); c(t) + x(t+1) < x(t)
% Value Function: V(x) = max u(x - x') + b * V(x') s.t. 0 < x' < x
% Euler Equation: u'(c(t)) = b*u'(c(t+1))

% Params
beta = 0.9;
u = @(c) log(c);
uprime = @(c) 1/c;


state = linspace(0.1,1,100)
policy = 0.5*state
plot(state, policy)


for j = 1:20
    for i = 1:100       
        eulerdiff = @(y) abs(uprime(state(i) - y) - beta*uprime(y - pchip(state, policy, y)));
        x =fminbnd(eulerdiff,0,state(i));
        policy(i) = x;
    end
end

plot(state, policy)


