IterRslt=iter_rbcIrr
[K_,Z_] = meshgrid(K,z);
I_ = IterRslt.var_policy.invst;
surf(Z_,K_,reshape(I_(1,:,:),[21,201]));
SimuRslt=simulate_rbcIrr(IterRslt);
histogram(SimuRslt.K); title('Histogram for K');
histogram(SimuRslt.c); title('Histogram for c');
histogram(SimuRslt.Inv); title('Histogram for c');
plot(SimuRslt.K(1:2,1:1000)'); title('Sample Paths of K');
plot(SimuRslt.c(1:2,1:1000)'); title('Sample Paths of K');
plot(SimuRslt.Inv(1:2,1:1000)'); title('Sample Paths of c');
plot(SimuRslt.z(1:2,1:1000)'); title('Sample Paths of c');
plot(SimuRslt.shock(1:2,1:1000)'); title('Sample Paths of z');


exp(-0.001 + 0.9*log(0.99)+0.0044);


exp(-0.05+1.0e-03)

plot(ans.var_state.K, ans.var_policy.K_next-(1-delta)*ans.var_state.K)

