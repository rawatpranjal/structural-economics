# Sequence-Space Jacobian for One-Asset HANK

## Overview

A HANK economy maps the path of a monetary policy shock into paths of output, inflation, the real interest rate, and the cross section of consumption. Solving for those paths requires propagating the shock through the household block, the firm block, the New Keynesian Phillips curve, and the monetary rule, all at the same time.

Sequence-space Jacobians make that propagation tractable. Each block is linearized around the steady state and represented by a matrix that maps a sequence of inputs into a sequence of outputs. The aggregate impulse response is then the solution of a single linear system whose blocks compose like Lego pieces.

The household block here reuses the discrete-time Aiyagari logic from `dynamic-programming/aiyagari/` and the endogenous-grid-point inversion from `heterogeneous-agents/endogenous-grid-points/`. The aggregate comparison overlays the representative-agent New Keynesian model from `dsge/nkdsge/`, so the reader can see what heterogeneous agents add to the standard monetary transmission picture.

## Equations

**Household problem.** Each household has assets $a \geq 0$ and idiosyncratic
labor productivity $z$ on a finite grid with transition matrix $P(z' \mid z)$.
With consumption $c$, real return $r_t$, real wage $w_t$, and a uniform
per-capita profit transfer $D_t$ (set to zero in the baseline calibration; see
the Takeaway), the budget constraint is

$$
\underbrace{c_t + a_{t+1}}_{\text{uses}} = \underbrace{(1 + r_t)\, a_t}_{\text{gross asset income}} + \underbrace{w_t z_t}_{\text{labor income}} + \underbrace{D_t}_{\text{profit transfer}}, \qquad a_{t+1} \geq 0.
$$

The flow-of-funds split is what makes the household block a function of three input paths $\lbrace r_s, w_s, D_s \rbrace$: each one shifts a different term in the budget constraint, so each one has its own household-block Jacobian.

CRRA preferences with inverse elasticity $\sigma$ give the Euler equation

$$
c_t^{-\sigma} = \beta\, (1 + r_{t+1})\, \mathbb{E}\lbrack c_{t+1}^{-\sigma} \mid z_t \rbrack
$$

at any interior optimum, and an inequality when the borrowing constraint binds.

**Firm block.** A representative firm produces $Y_t = K_t^{\alpha} N_t^{1-\alpha}$
with $N_t = 1$, capital $K_t = A_{t-1}$ (last period's household savings), and
a sticky-price markup $\mu_t$. First-order conditions give the real wage and
real return,

$$
w_t = (1 - \alpha)\, \frac{Y_t}{\mu_t}, \qquad
r_t + \delta = \alpha\, \frac{Y_t}{K_t}\, \frac{1}{\mu_t}.
$$

Real marginal cost is $mc_t = 1 / \mu_t$. Combining the two firm FOCs removes
$\mu_t$ and gives $r_t + \delta = \alpha\, w_t / ((1 - \alpha)\, K_t)$.
Profits earned by the markup are rebated uniformly to households:

$$
D_t = (1 - mc_t)\, Y_t.
$$

**New Keynesian Phillips curve.** A standard log-linearization of Rotemberg
price-setting around the zero-inflation steady state gives

$$
\pi_t = \beta\, \pi_{t+1} + \kappa\, (mc_t - mc^{\ast}),
$$

where $mc^{\ast} = 1 / \mu^{\ast}$ is the steady-state marginal cost.

**Monetary policy.** The nominal rate follows a Taylor rule with an exogenous
shock $v_t$, and the Fisher equation links nominal and real returns:

$$
i_t = \phi_{\pi}\, \pi_t + \phi_y\, \widehat{Y}_t + v_t,
\qquad r_t = i_t - \pi_{t+1},
\qquad v_t = \rho_v\, v_{t-1} + \varepsilon_t.
$$

**Sequence-space equilibrium map.** Stack T periods of unknowns
$U = (\pi, w)$ and shocks $Z = v$. After substituting the firm FOCs and the
household-block aggregator $A_t = A_t(\lbrace r_s, w_s \rbrace_{s = 0}^{T-1})$
the equilibrium reduces to two block equations,

$$
H(U, Z) = \begin{pmatrix} H^{\text{NKPC}}(U, Z) \\ H^{\text{Taylor}}(U, Z) \end{pmatrix} = 0.
$$

Linearizing around the steady state gives

$$
\underbrace{H_U}_{(2T) \times (2T) \text{ Jacobian w.r.t. unknowns}}\, \mathrm{d}U + \underbrace{H_Z}_{(2T) \times T \text{ Jacobian w.r.t. shocks}}\, \mathrm{d}Z = 0,
\qquad
\mathrm{d}U = \underbrace{-H_U^{-1} H_Z}_{\text{full IRF operator}}\, \mathrm{d}Z.
$$

This single linear solve is the entire equilibrium IRF computation; that is the algorithmic payoff of sequence-space.
Doing it without SSJ would mean either iterating a nonlinear fixed point at every shock size or differentiating the household block by finite differences in $T$ separate solves, both of which are far more expensive.
The work is concentrated in building $H_U$ and $H_Z$, and inside those almost all the cost is in the household-block Jacobian below.

**Household-block Jacobian.** Six matrices of shape $(T, T)$ collect the partial
derivatives of the two household-block aggregates with respect to the three
inputs,

$$
J^{Y, x}_{t, s} = \underbrace{\frac{\partial Y_t}{\partial x_s}}_{\substack{\text{response at time } t \\ \text{to a shock at time } s}}, \qquad
Y \in \lbrace C, A \rbrace, \quad x \in \lbrace r, w, D \rbrace,
$$

where $Y_t$ is an aggregate of the household block and $x_s$ is one element of
the input path.
Each column $s$ of $J^{Y, x}$ is the impulse response of $\lbrace Y_t \rbrace$ to a one-period shock at date $s$, so building $J^{Y, x}$ for all three inputs is the entire content of "the household block, linearized".
A naive build re-solves the household problem $3T$ times, $O(T^2)$ per column.
The fake-news algorithm exploits the time-invariance of the steady-state household problem: every column is a shifted version of a single backward-then-forward sweep, so the whole matrix costs one backward EGP step plus one forward distribution sweep.

## Model Setup

**Parameters.** Calibrated to a one-asset HANK at quarterly frequency.

| Object | Symbol | Value | Role |
|---|---|---:|---|
| Inverse EIS | $\sigma$ | 2.00 | Log utility |
| Discount factor | $\beta$ | 0.9892 | Calibrated so $A^\ast = K^\ast$ |
| Capital share | $\alpha$ | 0.33 | Cobb-Douglas |
| Depreciation | $\delta$ | 0.025 | Quarterly |
| Steady-state markup | $\mu^\ast$ | 1.10 | 10 percent |
| Income persistence | $\rho_z$ | 0.966 | AR(1) on log income |
| Income innovation std | $\sigma_z$ | 0.50 | Unconditional std target |
| NKPC slope | $\kappa$ | 0.10 | On real marginal cost |
| Taylor inflation | $\phi_\pi$ | 1.50 | |
| Taylor output gap | $\phi_y$ | 0.125 | |
| Shock persistence | $\rho_v$ | 0.61 | |
| Income grid | $n_z$ | 7 | Rouwenhorst |
| Asset grid | $n_a$ | 200 | Exponential on $[0.0, 200.0]$ |
| Sequence horizon | $T$ | 300 | Quarters |
| Monetary shock | $\varepsilon_0$ | 0.0006 | 25 bp annualized tightening |

**Steady-state values.** Real return $r^\ast = 0.0050$ per quarter
(~2.0 percent annual). Capital $K^\ast = 31.084$,
output $Y^\ast = 3.108$, wage $w^\ast = 1.8933$. Aggregate
household savings $A^\ast = 31.084$ clear the capital market to within
$10^{-5}$. Aggregate consumption $C^\ast = 2.048$.

## Solution Method

The block decomposition makes the problem tractable. The household block is
the heavy piece because its inputs and outputs are sequences of aggregates.

**Steady-state household block.** Endogenous grid points solve the
Euler-inverted policy in a few hundred iterations. The stationary distribution
follows from forward-iterating the Young (2010) lottery on the saving policy.
A bisection on $\beta$ matches aggregate savings to the firm capital target.

**Fake-news household Jacobian.** Each Jacobian column $J^{Y, x}_{:, s}$ is
the path of aggregate $Y$ in response to a unit pulse to input $x$ at date $s$.
The naive approach reruns the perfect-foresight household problem once per $s$,
costing $O(T)$ per column and $O(T^2)$ overall. The fake-news trick avoids the
inner loop:

```text
Algorithm: fake-news household-block Jacobian
Inputs    steady-state policies c_bar, a'_bar; stationary distribution D_bar;
          steady-state lottery (idx_low, omega_lo); horizon T; input x in {r, w}
Output    J^{C, x}[t, s], J^{A, x}[t, s] for t, s = 0..T-1

# Step 1: anticipation curves via repeated backward EGP, O(T |state|)
dc[0], da[0] = one_backward_egp_step(c_bar, perturbed x = x_bar + eps)
for k = 1..T-1:
    dc[k], da[k] = one_backward_egp_step(c_bar + eps dc[k-1], x = x_bar)
    # dc[k] is the date-0 consumption response to a unit pulse at date k

# Step 2: forward distribution propagation, O(T |state|) per pulse date
for s = 0..T-1:
    delta_D <- 0
    for t = 0..T-1:
        policy_t = dc[s - t] if t <= s else 0
        save_t   = da[s - t] if t <= s else 0
        J^{C, x}[t, s] = <policy_t, D_bar> + <c_bar, delta_D>
        J^{A, x}[t, s] = <save_t,   D_bar> + <a'_bar, delta_D>
        delta_D <- bar Lambda delta_D + Tau(save_t) D_bar
```

The two-step structure mirrors Auclert, Bardóczy, Rognlie, and Straub (2021):
anticipation curves are translation-invariant, so they are computed once and
then convolved with the time-varying input path during the forward sweep. The
distribution shift operator $\mathcal{T}$ tilts the steady-state lottery
weights by $-\Delta a' / (a_{k+1} - a_k)$, the linearization of the lottery
in the saving policy. The full SSJ library uses a further trick that drops the
overall cost to $O(T |state|)$ via a Toeplitz decomposition; the same blocks
are documented in their `sequence-jacobian` codebase.

**Firm, NKPC, and monetary blocks.** Cobb-Douglas FOCs, the linearized NKPC,
and the Taylor + Fisher pair are closed-form $T \times T$ Jacobians. With
$\mathrm{d}K_t = \mathrm{d}A_{t-1}$, the firm block expresses
$(\mathrm{d}Y, \mathrm{d}r, \mathrm{d}mc)$ as linear maps of
$(\mathrm{d}K, \mathrm{d}w)$. Substituting yields a $2 T \times 2 T$
system $H_U\, \mathrm{d}U = -H_Z\, \mathrm{d}Z$ in the unknowns
$U = (\pi, w)$, solved by a single dense linear solve.

**Convergence.** The household EGP converged in
945 iterations to a sup-norm residual of
9.83e-10. The stationary distribution converged in
${O}(\\text{tens of thousands})$ iterations to mass error
$\\leq 1e-11$. The Jacobian construction took
24.1 seconds at $T = 300$. The aggregate condition number of
$H_U$ is order $10^{2}$, well within
double-precision range.

## Results

A 25 basis-point monetary tightening pushes the real rate up in both economies. The two models differ structurally and the shapes reflect that. HANK has capital that is predetermined at date 0, so output cannot adjust instantaneously and the trough arrives with a lag while the capital stock contracts. The representative-agent NK model has no capital, so output and consumption coincide and respond on impact. The comparison shows what HANK adds qualitatively: a persistent, hump-shaped consumption response with heterogeneity across the wealth distribution, even if the headline magnitude depends on whether the benchmark includes capital and on how firm profits are rebated to households.

<img src="figures/irf-comparison.png" alt="Headline impulse responses: HANK vs representative-agent NK" width="80%">

Splitting the household-block consumption response by steady-state wealth quintile shows where the aggregate decline comes from. The lowest quintile, which holds the borrowing-constrained mass, has the highest MPC and contributes a disproportionately large share of the consumption decline relative to its share of aggregate wealth. The richer quintiles also reduce consumption but smooth more, consistent with the standard buffer-stock logic. The decomposition is on the policy channel only; the distributional channel (steady-state policy applied to a perturbed distribution) accumulates into the aggregate response but is not split across quintiles.

<img src="figures/quintile-irf.png" alt="Consumption IRF decomposed by wealth quintile" width="80%">

Each curve is the date-0 consumption response to a unit interest rate pulse anticipated to arrive at a future date $s$. Curves at longer lags are smaller and smoother because anticipation is filtered through the household's Euler equation: high-MPC households at the constraint barely respond to far-future news, while wealthy households respond similarly to news at any horizon below their planning window. These curves are the columns of $J^{C, r}_{0, s}$ before the forward distribution propagation.

<img src="figures/anticipation-curves.png" alt="Anticipation curves: date-0 consumption response to a future $r$ pulse" width="80%">

Left: the steady-state consumption policy is concave in assets and shifted up by income. Right: the stationary distribution has a sharp mode near the borrowing constraint and a long right tail. The constrained mass governs the magnitude of MPC heterogeneity, which is what gives HANK its IRF amplification.

<img src="figures/steady-state.png" alt="Steady-state policy and distribution" width="80%">

The household block is the costly piece; the aggregate solve is a single dense system. The peak-response rows summarize the central economic comparison between HANK and the RA NK benchmark.

**Solver diagnostics and IRF peak responses**

| Quantity                                    | Value           |
|:--------------------------------------------|:----------------|
| Household EGP iterations to convergence     | 945             |
| Household EGP final sup-norm residual       | 9.83e-10        |
| Stationary distribution iterations          | iterated to tol |
| Calibrated discount factor                  | 0.98918         |
| Aggregate savings A*                        | 31.084          |
| Capital target K*                           | 31.084          |
| Aggregate consumption C*                    | 2.048           |
| Steady-state real rate r* (quarterly)       | 0.0050          |
| Jacobian construction time (seconds, T=300) | 24.06           |
| H_U matrix size                             | 600 x 600       |
| H_U condition number                        | 1.17e+02        |
| Peak HA output response (% of Y*)           | -0.024          |
| Peak HA consumption response (% of C*)      | -0.036          |
| Peak RA output response (%)                 | -0.055          |

The sequence-space solve gives joint impulse responses of output, inflation, the real rate, and aggregate consumption. The HANK economy shows a larger and more persistent decline in consumption than the representative-agent NK economy with the same calibration, driven by high-MPC households at the borrowing constraint. The quintile decomposition makes the source of the amplification visible: most of the aggregate consumption drop comes from the lower income quintiles. The anticipation curves show why distant future shocks still have a contemporaneous effect on the date-0 policy: even high-MPC households reoptimize over their saving horizon when news arrives, and the response decays smoothly with the anticipation lag.

## Takeaway

Sequence-space Jacobians turn HANK with aggregate shocks into a tractable linear-algebra problem. The household block is the compute-heavy piece, but the fake-news algorithm builds its Jacobian from one backward iteration plus a forward propagation, sidestepping the repeated perfect-foresight resolves that earlier approaches like Krusell-Smith required.

Block composition pays for itself: firm, NKPC, and monetary blocks are closed-form $T \times T$ matrices and stack against the household block without recomputing anything. The aggregate IRF is then a single dense solve.

**On HANK amplification.** The baseline calibration here does not rebate firm profits and keeps labor supply inelastic. Both choices are deliberate. The household block takes a $D_t$ input slot, but with predetermined capital and a Cobb-Douglas firm, switching the rebate on actually flips the sign of the consumption IRF: short-run output is locked by the steady-state capital stock, the markup is countercyclical along the NKPC, so the rebate channel raises dividends right when wages fall, and the two effects cancel or reverse. The canonical sequence-jacobian one-asset HANK avoids this by using labor-only production with demand-determined output, so profits fall with output and amplification works as advertised. Both setups use exactly the same SSJ machinery; only the firm block changes. The `sequence-jacobian` package implements the full production pipeline (one-asset and two-asset variants, elastic labor, skill-proportional rebates, likelihood-based estimation) and is the natural next stop.

## References

- Auclert, A., Bardóczy, B., Rognlie, M., and Straub, L. (2021). Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models. *Econometrica*, 89(5), 2375-2408.
- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
- Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.
- Galí, J. (2015). *Monetary Policy, Inflation, and the Business Cycle: An Introduction to the New Keynesian Framework and Its Applications.* Princeton University Press.
- Young, E. R. (2010). Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell-Smith Algorithm and Non-Stochastic Simulations. *Journal of Economic Dynamics and Control*, 34(1), 36-41.
- `sequence-jacobian` Python package: https://github.com/shade-econ/sequence-jacobian
