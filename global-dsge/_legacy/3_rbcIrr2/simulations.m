result=iter_model;

% Policy Rules
plot(result.var_state.K, result.var_policy.K_next);
plot(result.var_state.K, result.var_policy.c);
plot(result.var_state.K, result.var_policy.mu);
plot(result.var_state.K, result.var_policy.K_next-(1-0.025)*result.var_state.K);

% Simulations

% Histogram
simulation=simulate_model(result);
histogram(simulation.K); title('Histogram for K');
histogram(simulation.c); title('Histogram for K');
histogram(simulation.shock); title('Histogram for Mu');
histogram(simulation.Inv); title('Histogram for Inv');

%Sample Paths
plot(simulation.K(1:2,1:1000)'); title('Sample Paths of K');
plot(simulation.c(1:2,1:1000)'); title('Sample Paths of K');
plot(simulation.Inv(1,1:100)'); title('Sample Paths of c');
plot(simulation.shock(1:2,1:1000)'); title('Sample Paths of z');
