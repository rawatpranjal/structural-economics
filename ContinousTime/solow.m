
% Parameters (annual)
s = 0.3;        %average share of real investment in real GDP (around 20%)
delta = 0.05;   %average ratio of depreciation to GDP (around 5%)
n = 0.04;       %population growth (around 2%)
g = 0.02;       %technology growth
alpha = 1/3;    %current level of capital share in the economy (around 33%)

% steady state
k_ss = (s/(g+n+delta))^(1/(1-alpha));

% initial condition
k0 = k_ss + 2; 

%differential equation
k_dot = @(t, k) s * k ^ alpha - (g + n + delta) * k;

% ODE solver
tspan = [0 100]; %integrates differential equations from 0 to 100
[time, k] = ode45(k_dot, tspan, k0); %solve it with ode45

% Plotting parameters
fsizenum = 14;
lwidnum = 2;
figure % plot series of interest
plot(time, k, 'b', time, k_ss*ones(length(time)),'r', 'LineWidth', lwidnum), title('Evolution of Capital starting at k0');
xlabel('Time'), set(gca,'FontSize',fsizenum);
legend('Capital','Steady State','Location','Southeast'), legend boxoff;


plot(time(2:end), dk);

