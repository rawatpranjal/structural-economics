# DMP Search, Vacancies, and Unemployment

> Free entry, Nash bargaining, and the Shimer amplification puzzle.

## Overview

An unemployed worker meets a posted vacancy through a matching technology, the resulting match generates surplus $z_t-b$ which is split by Nash bargaining, and free entry of vacancies pins down how many firms find it profitable to post. The single equilibrium price is labor-market tightness $\theta_t=v_t/u_t$: high tightness means firms are competing for workers, raising the worker's job-finding rate and lowering the firm's vacancy-filling rate.

The mechanism is direct. A productivity shock changes the present value $J_t$ of a filled job, which through the free-entry condition $k=\beta q(\theta_t)\mathbb{E}_t[J_{t+1}]$ pulls $\theta_t$ in the same direction. Higher tightness then raises $f(\theta_t)$, drains unemployment through the matching rate, and produces the downward-sloping Beveridge curve in $(u,v)$-space.

Quantitatively the model has a problem. With Shimer's (2005) calibration the surplus $z-b=0.60$ is large relative to the vacancy cost $k=0.211$, so a small productivity innovation barely moves the present-value calculus that drives entry. The simulated standard deviation ratio of tightness to productivity is roughly $1.7$ here, while in U.S. data it is closer to $19$. Hagedorn and Manovskii (2008) trace the amplification gap to this single calibration choice: with $b$ near $\bar z$ the surplus is small, the elasticity coefficient $C$ explodes, and the model can match observed labor-market volatility.

Two solvers compute the equilibrium and confirm that the puzzle is *economic*, not numerical. The log-linear local rule writes $\hat\theta_t=C\hat z_t$ with $C$ derived analytically from the linearized free-entry condition. The nonlinear rule discretizes $\hat z$ on a Rouwenhorst grid and iterates the free-entry fixed point at every node. Both produce nearly the same equilibrium because shocks are small and the relevant nonlinearity sits in $\theta\to q(\theta)$, which is nearly log-linear over the ergodic range.

Two cross-references frame this tutorial. The [McCall search tutorial](../job-search-mccall/) keeps only the worker side: a known wage offer distribution and an optimal-stopping reservation rule. Here vacancy posting is endogenous and the wage is bargained, so the worker's outside option matters in equilibrium. The [RBC tutorial](../rbc/) drives a representative-agent business cycle with the same persistent productivity shock; in DMP that shock propagates through match formation rather than capital accumulation.

## Equations

**Matching technology.** Let $u_t$ be the unemployment rate, $v_t$ the vacancy
rate, and $\theta_t=v_t/u_t$ tightness. Constant-returns matching gives

$$m(u_t,v_t)=\chi u_t^{1-\eta}v_t^\eta,\qquad
f(\theta_t)=\chi\theta_t^{\eta},\qquad
q(\theta_t)=\chi\theta_t^{\eta-1},$$

where $f$ is the worker's job-finding rate and $q$ is the firm's
vacancy-filling rate. Both depend on $\theta_t$ only.

**Productivity.** Aggregate productivity is a stationary AR(1) in logs,

$$\hat z_{t+1}=\rho\hat z_t+\epsilon_{t+1},\quad
\epsilon_{t+1}\sim\mathcal{N}(0,\sigma_\epsilon^2),\quad
z_t=\bar z\exp(\hat z_t).$$

**Wage rule.** Nash bargaining with worker weight $\gamma$ splits joint
surplus and yields the equilibrium wage

$$w_t=\gamma(z_t+k\theta_t)+(1-\gamma)b,$$

where $b$ is the flow value of unemployment (benefits, leisure, home
production) and $k$ is the per-period cost of an open vacancy. The term
$k\theta_t$ enters because tighter markets are more costly in expectation
for the worker to walk away from.

**Job value and free entry.** A filled job satisfies the recursive identity

$$J_t=z_t-w_t+\beta(1-\sigma)\,\mathbb{E}_t[J_{t+1}],$$

with $\sigma$ the exogenous separation rate. Free entry of vacancies equates
the expected discounted job value with the cost of one open vacancy,

$$k=\beta\,q(\theta_t)\,\mathbb{E}_t[J_{t+1}].$$

This is the equilibrium condition that pins down $\theta_t$.

**Stock dynamics.** Once $\theta_t$ is determined, unemployment evolves
mechanically and vacancies follow as a residual:

$$u_{t+1}=\sigma(1-u_t)+(1-f(\theta_t))u_t,\qquad
v_t=\theta_t u_t.$$

The deterministic steady state has $u_{ss}=\sigma/(\sigma+f(\theta_{ss}))$,
the textbook Beveridge relation.

**Local linearization.** Writing $\hat\theta_t=\log\theta_t-\log\theta_{ss}$ and
linearizing the free-entry condition at $\theta_{ss}=1$ delivers a closed-form
elasticity rule $\hat\theta_t=C\hat z_t$ with

$$C=\frac{\rho}{A-B\rho},\qquad
A=\frac{\eta k}{(1-\gamma)\beta\chi},\qquad
B=\beta A(1-\sigma)-\frac{\gamma k}{1-\gamma}.$$

At the baseline calibration $A=1.1098$ and $B=0.5262$, so a one-percent
productivity innovation raises tightness by $C=1.55$ percent.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Discount factor $\beta$ | 0.996 | Monthly time preference |
| Productivity persistence $\rho$ | 0.949 | AR(1) coefficient on $\hat z_t$ |
| Innovation s.d. $\sigma_\epsilon$ | 0.0065 | Monthly productivity shock |
| Separation rate $\sigma$ | 0.034 | Exogenous job destruction |
| Matching efficiency $\chi$ | 0.49 | Level of $m(u,v)$ |
| Matching elasticity $\eta$ | 0.72 | Vacancy elasticity in $m$ |
| Worker bargaining weight $\gamma$ | 0.72 | Nash share |
| Flow value of unemployment $b$ | 0.40 | Outside option |
| Vacancy cost $k$ | 0.2106 | Calibrated for $\theta_{ss}=1$ |
| Steady-state unemployment $u_{ss}$ | 0.0649 | $\sigma/(\sigma+f(\theta_{ss}))$ |
| Steady-state wage $w_{ss}$ | 0.9837 | Nash wage at $z=\bar z$ |
| Surplus $\bar z-b$ | 0.60 | Match surplus before vacancy costs |
| Coarse grid $N_z$ | 41 | Rouwenhorst nodes (tutorial run) |
| Fine-grid benchmark $N_z$ | 121 | Discretization audit |
| Simulation length | 4500 months | Post-burn-in moments |

## Solution Method

Two solvers run in parallel.

**Log-linear local rule.** Linearizing the free-entry condition and the productivity AR(1) around the deterministic steady state delivers the closed-form elasticity $C=\rho/(A-B\rho)$ above. The mapping $\theta_t=\exp(C\hat z_t)$ is exact to first order in $\hat z$ and carries no discretization error. Its weakness is that it forces a single elasticity at every productivity level, which would matter if the underlying $\theta(z)$ were strongly curved over the ergodic set.

**Nonlinear free-entry fixed point.** Discretize $\hat z_t$ on a Rouwenhorst grid with $N_z=41$ nodes and transition matrix $P_{ij}=\Pr(\hat z_{t+1}=\hat z_j\mid\hat z_t=\hat z_i)$. Substitute the free-entry condition for $\theta_i$ inside the job-value Bellman to get a recursion in $J_i=J(z_i)$ alone:

$$J_i=(1-\gamma)(z_i-b)-\gamma k\theta_i+\beta(1-\sigma)\sum_j P_{ij}J_j,\qquad \theta_i=(\tfrac{\beta\chi}{k}\sum_j P_{ij}J_j)^{1/(1-\eta)}.$$

The right-hand side defines a contraction in the sup norm with modulus $\beta(1-\sigma)=0.9621$, so iterates converge geometrically at that rate independent of $N_z$.

```text
Algorithm 1: Log-linear local rule
Inputs    primitives (β, σ, χ, η, γ, b, z̄, ρ); shock series {ẑ_t}
Outputs   elasticity C; tightness {θ_t}; unemployment {u_t}, vacancies {v_t}

1. Calibrate k from θ_ss = 1 at z = z̄
2. Compute  A = η k / [(1−γ) β χ],  B = β A (1−σ) − γ k / (1−γ)
3. Set C = ρ / (A − B ρ)
4. For each ẑ_t in the simulation:
       θ_t  ← exp(C · ẑ_t)
       f_t  ← χ θ_t^η
       u_{t+1} ← σ (1 − u_t) + (1 − f_t) u_t
       v_t  ← θ_t · u_t
```

```text
Algorithm 2: Nonlinear finite-state free-entry fixed point
Inputs    primitives; Rouwenhorst grid {ẑ_i}_{i=1..N_z};
          transition matrix P; calibrated k; tolerance ε
Outputs   job value J_i and tightness θ_i at each productivity state z_i

Initialise   J_i ← k / (β χ)               # value if free entry binds today
repeat n = 0, 1, 2, ...:
    EJ_i  ← Σ_j P_{ij} J_j                 # one mat-vec multiply
    θ_i   ← (β χ EJ_i / k)^{1/(1−η)}       # invert free entry
    J_i^new ← (1−γ)(z_i − b) − γ k θ_i + β(1−σ) EJ_i
    err   ← max_i |J_i^new − J_i|
    J_i   ← J_i^new
until err < ε
```

**Discretization audit.** The same nonlinear solver is rerun with $N_z=121$ nodes on a wider Rouwenhorst grid that brackets the coarse one. Interpolating the fine solution onto the coarse $\hat z$ grid gives a max relative gap in $\theta(z)$ of **3.97e-04%**, so the $N_z=41$ tutorial run is essentially the discretization-free answer.

At baseline calibration: log-linear $C=1.554$; nonlinear $N_z=41$ converged in **26 iterations** to sup-norm error **5.32e-12**; nonlinear $N_z=121$ converged in **31 iterations** to **7.99e-12**. The maximum relative gap between the nonlinear coarse solution and the log-linear rule is **3.23%** across the productivity grid.

## Results

The nonlinear free-entry rule (black) and the log-linear local rule (red dashed) trace nearly the same path through $(z,\theta)$-space, and the fine-grid benchmark (green dotted) sits indistinguishably on top of the coarse-grid solution. The blue cloud is where the simulated economy actually spends time. Reading off the cloud, $\theta_t$ moves about 1.72 times more than $z_t$ in log deviations — well below the order-of-magnitude amplification that Shimer (2005) measures in U.S. labor-market data. The puzzle is economic, not numerical: switching from the local rule to the global solver leaves the volatility ratio essentially unchanged.

<img src="figures/productivity-tightness.png" alt="Tightness as a function of productivity: log-linear, nonlinear coarse, and nonlinear fine-grid benchmark, with simulated months overlaid." width="80%">

Once tightness is determined, the rest is mechanics. Vacancy posting jumps because free entry makes $v_t$ the firm's contemporaneous choice variable; unemployment moves with a lag because it is a stock — today's hires only show up as a smaller pool of searchers tomorrow. The result is a co-movement of low-frequency $u$ and high-frequency $v$ that produces the cyclical asymmetry the Beveridge curve below traces out.

<img src="figures/unemployment-vacancies.png" alt="Simulated unemployment and vacancy paths under the nonlinear tightness rule." width="80%">

Productivity shocks move the economy along the curve; what makes the locus downward-sloping is the matching technology, which trades unemployment for vacancies through the constant-returns matching function. The cluster is one-dimensional because $z_t$ is the only shock — separations and matching efficiency are held fixed. Reallocation shocks to $\sigma$ or $\chi$ would shift this curve outward or inward instead of moving along it, which is the standard empirical decomposition of Beveridge-curve movements.

<img src="figures/beveridge-curve.png" alt="Simulated unemployment and vacancy pairs trace a downward-sloping Beveridge curve around the steady state." width="80%">

The signs are right: tightness and vacancies are strongly procyclical, unemployment is strongly countercyclical, and the log-linear and nonlinear simulations agree closely. The amplification ratios are wrong: Shimer (2005) reports an empirical $\sigma_\theta/\sigma_z\approx 19$, this calibration delivers about $1.7$. Switching solvers does not move that number.

**Simulated business-cycle moments**

| Variable                    |   Mean |   Std. log dev. |   Std./Std. z |   Corr. with z |
|:----------------------------|-------:|----------------:|--------------:|---------------:|
| Productivity z              | 0.9976 |          0.0216 |          1    |          1     |
| Unemployment u              | 0.0651 |          0.0243 |          1.13 |         -0.94  |
| Vacancies v                 | 0.0648 |          0.0166 |          0.77 |          0.866 |
| Tightness theta             | 0.9959 |          0.0372 |          1.72 |          1     |
| Tightness theta, log-linear | 0.9965 |          0.0336 |          1.55 |          1     |

Where the surplus comes from is the entire computational story. Holding the matching technology and bargaining weight fixed and raising $b$ from Shimer's $0.40$ to Hagedorn-Manovskii's $0.95$ shrinks the surplus $\bar z-b$ by an order of magnitude and drives the elasticity $C$ from about $1.55$ to roughly $19$ — the empirical volatility ratio Shimer measures. A small-surplus economy amplifies productivity shocks because the vacancy cost $k$ is then a substantial fraction of expected match value, so a small move in $z$ is a large move in the ratio $\beta\chi\,\mathbb{E}[J']/k$ that drives free entry.

**Tightness elasticity by flow value of unemployment**

|   Flow value b |   Surplus z-b |   Vacancy cost k |   Tightness elasticity C |
|---------------:|--------------:|-----------------:|-------------------------:|
|           0.4  |          0.6  |           0.2106 |                     1.55 |
|           0.55 |          0.45 |           0.158  |                     2.07 |
|           0.71 |          0.29 |           0.1018 |                     3.22 |
|           0.85 |          0.15 |           0.0527 |                     6.22 |
|           0.95 |          0.05 |           0.0176 |                    18.65 |

## Takeaway

DMP gives an equilibrium account of the Beveridge curve. Productivity raises match surplus, vacancy posting expands through free entry, and unemployment falls as the job-finding rate rises. Switching from a log-linear local rule to a global nonlinear free-entry fixed point leaves the volatility ratios essentially unchanged, so the Shimer (2005) amplification puzzle is *not* a numerical artefact: with $b=0.40$ the surplus $\bar z-b=0.60$ is large enough that modest productivity shocks are barely transmitted through the firm's free-entry decision. The sensitivity table makes the lever explicit: amplification depends mechanically on the size of the surplus, exactly the channel Hagedorn and Manovskii (2008) exploit. Two natural extensions: replacing exogenous separations by an endogenous match-destruction margin (Mortensen-Pissarides 1994) gives the Beveridge curve a second source of variation; embedding the same search block inside the heterogeneous-agent saving problem in [Aiyagari](../aiyagari/) lets idiosyncratic income risk and frictional unemployment interact.

## References

- Diamond, P. (1982). "Aggregate Demand Management in Search Equilibrium." *Journal of Political Economy*, 90(5), 881-894.
- Mortensen, D. and Pissarides, C. (1994). "Job Creation and Job Destruction in the Theory of Unemployment." *Review of Economic Studies*, 61(3), 397-415.
- Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.
- Shimer, R. (2005). "The Cyclical Behavior of Equilibrium Unemployment and Vacancies." *American Economic Review*, 95(1), 25-49.
- Hagedorn, M. and Manovskii, I. (2008). "The Cyclical Behavior of Equilibrium Unemployment and Vacancies Revisited." *American Economic Review*, 98(4), 1692-1706.
