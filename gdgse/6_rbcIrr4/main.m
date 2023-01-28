clear;
% gndsge_codegen('rbc');

IterRslt = iter_rbc;

figure; subplot(2,1,1);
plot(IterRslt.var_state.K, IterRslt.var_aux.Inv);
% ylim([0.92,1.0]);
xlabel('K');
legend({'$z=z_L$','$z=z_H$'}, 'interpreter','latex','Location','East','FontSize',14);
title('Policy Functions for Investment');

subplot(2,1,2);
plot(IterRslt.var_state.K, IterRslt.var_policy.mu);
xlabel('K');
legend({'$z=z_L$','$z=z_H$'}, 'interpreter','latex','Location','NorthEast','FontSize',14);
title('Policy Functions for Multiplier of Investment Irreversibility');
% print('figures/policy_combined.png','-dpng','-r300');
