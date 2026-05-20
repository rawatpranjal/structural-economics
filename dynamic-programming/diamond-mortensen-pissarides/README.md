# DMP Search, Vacancies, and Unemployment

## Overview

Unemployed workers and posted vacancies meet through a matching technology. A formed match produces surplus $z_t-b$, and Nash bargaining splits it.

The equilibrium object is labor-market tightness $\theta_t=v_t/u_t$. Free entry pins down tightness because firms post vacancies until expected job value covers vacancy cost.

The code compares a log-linear rule with a finite-state free-entry fixed point. This asks whether the Shimer amplification puzzle comes from the solver or from surplus calibration.

## Equations

**Matching technology.** Let $u_t$ be unemployment, $v_t$ vacancies, and
$\theta_t=v_t/u_t$ tightness. Constant-returns matching gives

$$m(u_t,v_t)=\chi u_t^{1-\eta}v_t^\eta,\qquad
f(\theta_t)=\chi\theta_t^{\eta},\qquad
q(\theta_t)=\chi\theta_t^{\eta-1},$$

Here $f$ is the worker job-finding rate. The term $q$ is the firm
vacancy-filling rate.

**Productivity.** Aggregate productivity is a stationary AR(1) in logs,

$$\hat z_{t+1}=\rho\hat z_t+\epsilon_{t+1},\quad
\epsilon_{t+1}\sim\mathcal{N}(0,\sigma_\epsilon^2),\quad
z_t=\bar z\exp(\hat z_t).$$

**Wage rule.** Nash bargaining with worker weight $\gamma$ splits joint
surplus and yields the equilibrium wage

$$w_t=\gamma(z_t+k\theta_t)+(1-\gamma)b,$$

Here $b$ is the flow value of unemployment. The parameter $k$ is the
per-period cost of an open vacancy.

**Job value and free entry.** A filled job has value

$$J_t=z_t-w_t+\beta(1-\sigma)\,\mathbb{E}_t[J_{t+1}],$$

where $\sigma$ is the exogenous separation rate. Free entry equates expected
discounted job value with vacancy cost:

$$k=\beta\,q(\theta_t)\,\mathbb{E}_t[J_{t+1}].$$

This condition pins down $\theta_t$.

**Stock dynamics.** Once $\theta_t$ is known, unemployment follows

$$u_{t+1}=\sigma(1-u_t)+(1-f(\theta_t))u_t,\qquad
v_t=\theta_t u_t.$$

The deterministic steady state has $u_{ss}=\sigma/(\sigma+f(\theta_{ss}))$.

**Local linearization.** Write
$\hat\theta_t=\log\theta_t-\log\theta_{ss}$. Linearizing free entry at
$\theta_{ss}=1$ gives $\hat\theta_t=C\hat z_t$, with

$$C=\frac{\rho}{A-B\rho},\qquad
A=\frac{\eta k}{(1-\gamma)\beta\chi},\qquad
B=\beta A(1-\sigma)-\frac{\gamma k}{1-\gamma}.$$

At baseline, $A=1.1098$ and $B=0.5262$. A one-percent productivity
innovation raises tightness by $C=1.55$ percent.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Discount factor $\beta$ | 0.996 | Monthly time preference |
| Productivity persistence $\rho$ | 0.949 | AR(1) coefficient on $\hat z_t$ |
| Innovation s.d. $\sigma_\epsilon$ | 0.0065 | Monthly productivity shock |
| Mean productivity $\bar z$ | 1.00 | Normalization |
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

Two solvers compute the same tightness rule.

**Log-linear local rule.** The local rule linearizes free entry and the AR(1) around the deterministic steady state. It gives $C=\rho/(A-B\rho)$ and sets $\theta_t=\exp(C\hat z_t)$.

**Nonlinear free-entry fixed point.** The nonlinear solver discretizes $\hat z_t$ on a Rouwenhorst grid with $N_z=41$ nodes. It substitutes free entry inside the job-value Bellman:

$$J_i=(1-\gamma)(z_i-b)-\gamma k\theta_i+\beta(1-\sigma)\sum_j P_{ij}J_j,\qquad \theta_i=(\frac{\beta\chi}{k}\sum_j P_{ij}J_j)^{1/(1-\eta)}.$$

The operator is a contraction. Its linear term $\beta(1-\sigma)E[J']$ alone has modulus $\beta(1-\sigma)=0.9621$, but the substituted free-entry term $\theta(E[J'])$ inside the Bellman adds a negative correction. The total derivative of the update with respect to $E[J']$ at the steady state gives an effective modulus of about $0.293$, well below the linear-term bound. The tight effective modulus is why the fixed point converges in a few dozen iterations rather than the several hundred the linear-term modulus would imply.

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

**Discretization audit.** The same nonlinear solver is rerun with $N_z=121$ nodes. The interpolated gap in $\theta(z)$ is **3.97e-04%**.

At baseline, the log-linear elasticity is $C=1.554$. The $N_z=41$ fixed point converges in **26 iterations**. The maximum policy gap between the nonlinear and log-linear rules is **3.23%**.

## Results

The nonlinear rule, fine-grid rule, and local rule are close over the simulated productivity range. Tightness moves about 1.72 times as much as productivity, far below Shimer's value near 19. The mismatch remains after switching solvers.

<img src="figures/productivity-tightness.png" alt="Tightness as a function of productivity: log-linear, nonlinear coarse, and nonlinear fine-grid benchmark, with simulated months overlaid." width="80%">

Given tightness, unemployment follows the stock law and vacancies equal $\theta_t u_t$. Vacancies jump with entry. Unemployment falls more slowly because hires reduce tomorrow's search pool.

<img src="figures/unemployment-vacancies.png" alt="Simulated unemployment and vacancy paths under the nonlinear tightness rule." width="80%">

The simulated pairs trace a Beveridge curve. Productivity shocks move the economy along that curve because separations and matching efficiency stay fixed.

<img src="figures/beveridge-curve.png" alt="Simulated unemployment and vacancy pairs trace a downward-sloping Beveridge curve around the steady state." width="80%">

The signs match the model logic. Tightness and vacancies are procyclical, unemployment is countercyclical, and both solvers give similar volatility.

**Simulated business-cycle moments**

| Variable                    |   Mean |   Std. log dev. |   Std./Std. z |   Corr. with z |
|:----------------------------|-------:|----------------:|--------------:|---------------:|
| Productivity z              | 0.9976 |          0.0216 |          1    |          1     |
| Unemployment u              | 0.0651 |          0.0243 |          1.13 |         -0.94  |
| Vacancies v                 | 0.0648 |          0.0166 |          0.77 |          0.866 |
| Tightness theta             | 0.9959 |          0.0372 |          1.72 |          1     |
| Tightness theta, log-linear | 0.9965 |          0.0336 |          1.55 |          1     |

Raising $b$ shrinks surplus and raises elasticity $C$. Moving from $b=0.40$ to $b=0.95$ takes $C$ from 1.55 to 18.65. The surplus calibration drives amplification.

**Tightness elasticity by flow value of unemployment**

|   Flow value b |   Surplus z-b |   Vacancy cost k |   Tightness elasticity C |
|---------------:|--------------:|-----------------:|-------------------------:|
|           0.4  |          0.6  |           0.2106 |                     1.55 |
|           0.55 |          0.45 |           0.158  |                     2.07 |
|           0.71 |          0.29 |           0.1018 |                     3.22 |
|           0.85 |          0.15 |           0.0527 |                     6.22 |
|           0.95 |          0.05 |           0.0176 |                    18.65 |

The policy gap, interpolation gap, and iteration counts are persisted here so the convergence claims in the Solution Method section can be cross-checked against a committed artifact.

**Nonlinear fixed-point solver diagnostics**

| Quantity                                    | policy_gap_pct   | grid_gap_pct   | iterations   |
|:--------------------------------------------|:-----------------|:---------------|:-------------|
| Coarse-grid policy gap vs. log-linear       | 3.2287           |                |              |
| Coarse-grid interpolation gap vs. fine grid |                  | 0.000397       |              |
| Coarse-grid fixed-point iterations          |                  |                | 26           |
| Fine-grid fixed-point iterations            |                  |                | 31           |

## Takeaway

DMP links productivity to vacancies and unemployment through free entry. The local rule and nonlinear fixed point give almost the same volatility. The Shimer puzzle therefore comes from the large baseline surplus, not from the numerical method. The sensitivity table shows how a smaller surplus raises tightness amplification.

## References

- Diamond, P. (1982). "Aggregate Demand Management in Search Equilibrium." *Journal of Political Economy*, 90(5), 881-894.
- Mortensen, D. and Pissarides, C. (1994). "Job Creation and Job Destruction in the Theory of Unemployment." *Review of Economic Studies*, 61(3), 397-415.
- Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.
- Shimer, R. (2005). "The Cyclical Behavior of Equilibrium Unemployment and Vacancies." *American Economic Review*, 95(1), 25-49.
- Hagedorn, M. and Manovskii, I. (2008). "The Cyclical Behavior of Equilibrium Unemployment and Vacancies Revisited." *American Economic Review*, 98(4), 1692-1706.
