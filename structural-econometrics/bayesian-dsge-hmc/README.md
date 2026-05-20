# Bayesian DSGE Estimation by HMC/NUTS

## Overview

A central banker wants the posterior over the deep parameters of a small New Keynesian model after seeing output, inflation, and the policy rate for a few decades. Maximum likelihood gives a point estimate. The Bayesian alternative carries uncertainty about every parameter through to every impulse response.

The estimation pipeline is end-to-end gradient-based. The structural parameters enter a four-by-four rational-expectations system. Klein QZ selects the stable saddle path. The stable path becomes a state-space model. A Kalman filter scores the observed series. The Klein-to-Kalman chain is differentiable. Hamiltonian Monte Carlo exploits those gradients and explores the posterior much faster than random-walk Metropolis on the same target.

The Klein step uses the JAX port from `lib/perturbation_jax.py`. The underlying complex-Schur primitive in JAX has no autodiff rule, so the policy function gets gradients from an implicit-function-theorem JVP on the Klein equations. The Kalman recursion in `lib/kalman_jax.py` runs in plain JAX and gets its gradients from standard autodiff. The two compose cleanly into a single differentiable log posterior, and BlackJAX NUTS samples from it.

## Equations

The three-equation New Keynesian block is the same one used in
`dsge/nkdsge/`. The IS curve, Phillips curve, and Taylor rule are

$$
y_t = \mathbb{E}_t y_{t+1} - \tfrac{1}{\sigma}(i_t - \mathbb{E}_t \pi_{t+1}) + d_t,
$$

$$
\pi_t = \beta\,\mathbb{E}_t \pi_{t+1} + \kappa\,y_t,
$$

$$
i_t = \phi_\pi\,\pi_t + \phi_y\,y_t + v_t.
$$

Two AR(1) shocks: a Taylor-rule wedge $v_t$ with persistence $\rho_v$ and
innovation s.d. $\sigma_v$, and a natural-rate demand shock $d_t$ with $\rho_d$,
$\sigma_d$. In Klein form $A\,\mathbb{E}_t s_{t+1}=B\,s_t$ with state
$s=(v,d,y,\pi)$ and $n_\text{predetermined}=2$,

$$
A=\underbrace{\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 1/\sigma\\
0 & 0 & 0 & \beta
\end{bmatrix}}_{\text{coefficients on } \mathbb{E}_t s_{t+1}},\qquad
B=\underbrace{\begin{bmatrix}
\rho_v & 0 & 0 & 0\\
0 & \rho_d & 0 & 0\\
1/\sigma & -1 & 1+\phi_y/\sigma & \phi_\pi/\sigma\\
0 & 0 & -\kappa & 1
\end{bmatrix}}_{\text{coefficients on } s_t}.
$$

Klein QZ returns the policy matrices $(F, P)$ in $x_{t+1}=F\,x_t$ and $y_t=P\,x_t$
with $x=(v,d)$, $y=(y_t,\pi_t)$. Adding the Taylor rule yields observables
$(y_t, \pi_t, i_t)$ and a linear Gaussian state-space form. The Kalman filter
gives the log marginal likelihood

$$
\log p(Y\mid\theta) = \sum_{t=1}^{T}\log \mathcal{N}(y_t;\,H\hat x_{t\mid t-1},\,H\Sigma_{t\mid t-1}H^\top+S),
$$

evaluated with the predict-update recursion. The posterior is
$p(\theta\mid Y)\propto p(\theta)\,p(Y\mid\theta)$.

## Model Setup

| Symbol | Role | Prior | True | Prior mean |
|---|---|---|---:|---:|
| $\sigma$ | Inverse EIS in the IS curve | Gamma(2, 0.5) | 1.0 | 1.0 |
| $\phi_\pi$ | Taylor-rule response to inflation | Gamma(4, 0.375) | 1.5 | 1.5 |
| $\phi_y$ | Taylor-rule response to the output gap | Gamma(2, 0.0625) | 0.125 | 0.125 |
| $\kappa$ | Phillips slope | Beta(3, 7) | 0.3 | 0.3 |
| $\sigma_v$ | Monetary innovation s.d. | Gamma(2, 0.005) | 0.01 | 0.01 |
| $\rho_v$ | Monetary persistence | Beta(2, 2) | 0.5 | 0.5 |
| $\sigma_d$ | Demand innovation s.d. | Gamma(2, 0.005) | 0.01 | 0.01 |
| $\rho_d$ | Demand persistence | Beta(2, 2) | 0.8 | 0.5 |
| $\beta$ | Quarterly discount factor | fixed at 0.99 | 0.99 | - |
| $T$ | Series length | observations | 200 | - |

Priors live on the constrained (economic) space. NUTS samples in an unconstrained latent $z$ via $\theta=\exp(z)$ for positive parameters and $\theta=\mathrm{sigmoid}(z)$ for bounded parameters. The implied log-Jacobian is added to the target density.

## Solution Method

The pipeline composes three JAX building blocks. Each one has a single responsibility.

**Klein QZ (`lib/perturbation_jax.py`).** Builds the four-by-four pencil above, reduces it to the standard eigenproblem $M=A^{-1}B$, runs a complex Schur, and sorts the diagonal by $|\lambda|$ via Givens-rotation bubble sort. With Blanchard-Kahn satisfied the stable eigenvalues sit in the top-left block and $(F, P)$ come out of $Z_{11}T_{11}Z_{11}^{-1}$ and $Z_{21}Z_{11}^{-1}$. JAX 0.9.x has no autodiff rule for `schur`, so a `custom_jvp` solves the implicit Klein equations for the tangents instead of differentiating through the Schur path.

**Kalman filter (`lib/kalman_jax.py`).** Predict-update recursion in `jax.lax.scan`. Stationary initial covariance from the discrete Lyapunov solver. Symmetric Cholesky of the innovation covariance keeps the log-determinant cheap and the gain numerically stable. Plain autodiff; no custom rule.

**BlackJAX NUTS.** `blackjax.window_adaptation` tunes the step size and mass matrix during warm-up. Four chains run independently from near-zero unconstrained starting points, one after another in a sequential loop. A random-walk Metropolis with the same total draw count runs as the baseline.

### A small 2-by-2 worked example

Before scaling to the 4-by-4 NK pencil, here is the smallest system that exercises Klein, the policy formula, and the gradient. The shock is predetermined and the forward-looking variable solves a simple Euler-style condition:

$$v_{t+1}=\rho\,v_t,\qquad y_t=a\,\mathbb{E}_t y_{t+1}+b\,v_t.$$

With state $s=(v,y)$ and $n_\text{predetermined}=1$, the Klein pencil is

$$A=\begin{bmatrix}1 & 0\\ 0 & a\end{bmatrix},\qquadB=\begin{bmatrix}\rho & 0\\ -b & 1\end{bmatrix}.$$

Pick $\rho=0.5$, $a=0.5$, $b=1$. The closed-form guess $y_t=\psi\,v_t$ gives $\psi\,(1-a\rho)=b$, so $\psi=b/(1-a\rho)=4/3$.

Now the Schur path. Reducing the pencil gives

$$M=A^{-1}B=\begin{bmatrix}0.5 & 0\\ -2 & 2\end{bmatrix},\quad\text{eigenvalues }\{0.5,\,2\}.$$

Only the eigenvalue $\rho=0.5$ is stable, matching the one predetermined variable, so Blanchard-Kahn holds. Its eigenvector $(1,\,4/3)^\top$ partitions as $Z_{11}=1$, $Z_{21}=4/3$. The Klein formulas then deliver $F=Z_{11}\,T_{11}\,Z_{11}^{-1}=0.5$ and $P=Z_{21}\,Z_{11}^{-1}=4/3$, the same numbers as the closed form.

Gradients close the loop. Differentiating $\psi=b/(1-a\rho)$ by hand gives $\partial\psi/\partial a=b\rho/(1-a\rho)^2=8/9$, $\partial\psi/\partial b=1/(1-a\rho)=4/3$, and $\partial\psi/\partial\rho=ab/(1-a\rho)^2=8/9$. The implicit-IFT JVP in `lib/perturbation_jax.py` returns those three numbers from `jax.grad` at machine precision. Drop the toy and the same machinery handles the 4-by-4 NK system unchanged.

## Results

The three observables span 200 quarters. Output and inflation carry both shocks through the policy function; the policy rate carries them through the Taylor rule on top of $v_t$.

<img src="/Users/pranjal/Code/computational-economics/structural-econometrics/bayesian-dsge-hmc/figures/observations.png" alt="Simulated observations." width="80%">

NUTS ran 4 chains of 1000 warm-up plus 2000 kept draws in 728.9 seconds wall time. Random-walk Metropolis ran a single chain of 8000 draws in 3.2 seconds. NUTS reached an average acceptance of 0.91; RW-MH at the chosen step size reached 0.02.

Each panel overlays the prior (dashed black), the posterior histogram (light blue), and the data-generating value (red). The posteriors concentrate on the truth wherever the parameter is well identified from the three observables; remaining width is real posterior uncertainty given $T=200$.

<img src="/Users/pranjal/Code/computational-economics/structural-econometrics/bayesian-dsge-hmc/figures/posterior-densities.png" alt="Posterior densities with priors and ground truth." width="80%">

The top row shows the response of $(y,\pi,i)$ to a one-standard-deviation monetary wedge $v_t$. The bottom row shows the response to a demand shock $d_t$. The blue line and band are the posterior median and the 90 percent credible band over draws; the red dashed line is the IRF at the data-generating $\theta$.

<img src="/Users/pranjal/Code/computational-economics/structural-econometrics/bayesian-dsge-hmc/figures/posterior-irfs.png" alt="Posterior impulse responses with 90 percent bands." width="80%">

**Posterior summary and effective sample size comparison.**

| parameter   |   true |   post mean |   post median |   5 percent |   95 percent |    R hat |   NUTS ESS |   RW MH ESS |   ESS ratio |
|:------------|-------:|------------:|--------------:|------------:|-------------:|---------:|-----------:|------------:|------------:|
| sigma       |  1     |   0.877019  |     0.843089  |  0.473794   |    1.39662   | 1.0011   |    2192.42 |     3.792   |    578.17   |
| phi_pi      |  1.5   |   1.61392   |     1.61603   |  1.43668    |    1.78474   | 1.0007   |    3000.15 |    52.614   |     57.0219 |
| phi_y       |  0.125 |   0.127129  |     0.106443  |  0.0234303  |    0.297344  | 0.999865 |    3659.94 |    10.1277  |    361.378  |
| kappa       |  0.3   |   0.323172  |     0.324824  |  0.259208   |    0.382229  | 1.00087  |    2100.14 |    29.8694  |     70.3109 |
| sigma_v     |  0.01  |   0.0102313 |     0.0101611 |  0.00904731 |    0.0116557 | 0.999906 |    3853.68 |    42.2221  |     91.2715 |
| rho_v       |  0.5   |   0.470964  |     0.472242  |  0.382225   |    0.55625   | 0.999857 |    3632.94 |     4.27264 |    850.28   |
| sigma_d     |  0.01  |   0.0136829 |     0.0127763 |  0.00748374 |    0.0227336 | 1.00105  |    2493.58 |    41.4234  |     60.1973 |
| rho_d       |  0.8   |   0.799726  |     0.799844  |  0.761747   |    0.836667  | 1.00013  |    2689.61 |    18.1934  |    147.834  |

The figure y-axis uses a log scale. NUTS exploits gradients of the log posterior with respect to all eight estimated parameters; RW-MH spends most of its budget rejecting proposals or accepting highly autocorrelated ones.

<img src="/Users/pranjal/Code/computational-economics/structural-econometrics/bayesian-dsge-hmc/figures/ess-comparison.png" alt="ESS per parameter: NUTS vs. random-walk Metropolis." width="80%">

## Takeaway

Three pieces compose. Klein QZ in JAX gives a differentiable policy function. A Kalman filter in JAX gives a differentiable log likelihood. BlackJAX NUTS turns the differentiable posterior into samples. The recovery experiment shows the posterior concentrates on the data-generating parameters at $T=200$, and gradient-based sampling delivers one to several orders of magnitude more effective draws per raw sample than the random-walk baseline at the same total draw count. That per-sample mixing gain does not carry over to a per-wall-clock-second comparison: NUTS pays a large warm-up and JIT-compilation cost, so on this run RW-MH produces more effective draws per second on several parameters. The gradient-based advantage is in samples drawn, not in wall time.

## References

- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn Sampler. *Journal of Machine Learning Research*, 15, 1593-1623.
- Cabezas, A. et al. (2024). BlackJAX: Composable Bayesian inference in JAX.
- Smets, F. and Wouters, R. (2007). Shocks and Frictions in US Business Cycles. *American Economic Review*, 97(3), 586-606.
- Farkas, M. (2020). Bayesian estimation of DSGE models in Stan. IMFS Working Paper 145.
- Herbst, E. and Schorfheide, F. (2016). *Bayesian Estimation of DSGE Models*. Princeton University Press.
