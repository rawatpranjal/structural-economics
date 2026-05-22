# Kolmogorov Forward Equation and the Stationary Wealth Distribution

## Overview

Continuous-time heterogeneous-agent macroeconomics needs two ingredients. The household side picks consumption from a Hamilton-Jacobi-Bellman equation. The cross section follows a Kolmogorov forward equation, also called the Fokker-Planck equation, that propagates the population density of households along the policy-implied drift. The stationary version of that forward equation is what closes a Huggett or Aiyagari steady state: it tells the modeller the long-run distribution of assets and income that the household policy induces.

The economic content is mass conservation. Households do not appear or disappear, so the rate of change of mass in any asset interval equals the inflow at one boundary minus the outflow at the other, plus the net Poisson income switching. Writing that bookkeeping as a partial differential equation, then discretising it on the same grid as the HJB, gives a sparse linear system whose stationary solution is the cross section.

The numerical payoff is the same operator twice. The upwind generator built in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/) acts on values from the right. Its transpose acts on densities from the left. The same matrix entries, just transposed, solve both directions. The single linear-algebra object discretises the entire stationary equilibrium of the dense Huggett tutorial in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/) and the dense continuous-time Aiyagari tutorial in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/). The discrete-time analog is the Young (2010) lottery iteration used in the steady-state block of [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/).

The reader takes away three things. The derivation of the forward equation from mass conservation. The matrix-algebra duality that turns the upwind HJB generator into a forward operator for the density. The recipe for solving the stationary linear system with one row replaced by a normalisation constraint so the singular generator has a unique probability solution.

## Preliminary readings

- [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/)

## Equations

Let $`x`$ be a one-dimensional state on a bounded interval $`[\underline x, \overline x]`$ and let $`g(x, t)`$ be the density of agents at state $`x`$ at time $`t`$. Each agent's state drifts at rate $`s(x)`$ and is buffeted by a diffusion of constant volatility $`\sigma \geq 0`$. Mass conservation requires that the rate of change of mass in any region equals the inflow at the left boundary minus the outflow at the right. Differentiating the integral balance gives the continuity equation,

```math
\frac{\partial g}{\partial t}(x, t) =
-\frac{\partial}{\partial x}\big[s(x)  g(x, t)\big]
+ \frac{\sigma^2}{2}  \frac{\partial^2 g}{\partial x^2}(x, t) .
```

The first term is the divergence of the deterministic flux $`s(x)  g(x, t)`$. The second term is the diffusion correction from a Wiener noise; it drops to zero when the process is purely convective. In a steady state the left-hand side is zero and the density satisfies the elliptic equation $`-\partial_x[s g] + (\sigma^2/2)  \partial_{xx} g = 0`$ together with the normalisation $`\int g  dx = 1`$.

Discretise $`x`$ on a uniform grid $`x_1 < x_2 < \cdots < x_n`$ with spacing $`\Delta x`$ and let $`g_i = g(x_i)`$. The same upwind operator that discretises the HJB on this grid carries the drift block of the forward equation. Let $`A`$ denote that operator, built in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/). Its rows sum to zero and its off-diagonals are non-negative, so $`A`$ is the generator of a continuous-time Markov chain on the grid. Discrete mass conservation then becomes a single linear system,

```math
A^{\top}  g = 0,
\qquad
\sum_i g_i  \Delta x = 1 .
```

The system is singular because $`A`$ has zero row sums, so $`A^{\top}`$ has the constant vector in its left null space and the density vector $`g`$ in its right null space. The numerical solve pins the scale by replacing one row of $`A^{\top}`$ with the normalisation constraint, solving the resulting non-singular system, and rescaling the solution to integrate to one. This is the stationary KFE solve that the dense Huggett and Aiyagari tutorials invoke.

When the state has more than one component, the upwind generator on the joint state space is built blockwise. For a two-state Poisson income chain with states $`j \in \lbrace L, H \rbrace`$, jump rate $`\lambda_{LH}`$ from $`L`$ to $`H`$, and jump rate $`\lambda_{HL}`$ from $`H`$ to $`L`$, the income generator is

```math
Q =
\begin{pmatrix}
-\lambda_{LH} & \lambda_{LH} \\
 \lambda_{HL} & -\lambda_{HL}
\end{pmatrix} .
```

The off-diagonal entries are Poisson jump rates and each row sums to zero, so $`Q`$ is a continuous-time Markov generator on the discrete income axis. The joint generator on the product state space is

```math
A_{\mathrm{joint}}
= \mathrm{diag}(A_{L}, A_{H}) + Q \otimes I_n ,
```

where $`A_j`$ is the asset-axis generator for income state $`j`$ and $`I_n`$ is the $`n \times n`$ identity. The first term advances the asset distribution within each income state. The second term shuffles probability mass across income states at every asset level. The stationary joint density solves $`A_{\mathrm{joint}}^{\top} g = 0`$ by the same single-row-replacement trick. This is the generalisation that [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/) uses with an $`N`$-state Rouwenhorst-derived chain.

## Model Setup

The prelim runs three small examples that share the same upwind generator. Symbols already used by dense tutorials carry their original meaning so a reader walking from this prelim to Huggett or Aiyagari-HACT sees no notational drift.

| Symbol | Value or set | Role |
|---|---|---|
| $`x`$ | grid node | generic 1D state [from `optimal-control/upwind-finite-differences/`] |
| $`a`$ | (Huggett asset grid) | asset, instance of $`x`$ [from `heterogeneous-agents/huggett-incomplete-markets/`] |
| $`k`$ | $`[0.5, 15.0]`$ | capital, instance of $`x`$ [from `optimal-control/hjb-growth/`] |
| $`g(x)`$ | grid array | cross-sectional density [from `huggett-incomplete-markets/` and `aiyagari-hact/`] |
| $`s(x)`$ | grid array | signed drift, sign chooses the upwind side [from `huggett-incomplete-markets/`] |
| $`A`$ | sparse $`n \times n`$ | upwind generator on the 1D state [from `optimal-control/upwind-finite-differences/`] |
| $`A^{\top}`$ | sparse $`n \times n`$ | forward operator for the density [prelim introduces the explicit duality name; dense tutorials adopt it on cut] |
| $`Q`$ | $`2 \times 2`$ generator | continuous-time Markov generator for two-state Poisson income [from `aiyagari-hact/`] |
| $`\lambda_{LH}, \lambda_{HL}`$ | $`(1.2, 1.2)`$ | Poisson jump rates between income states [prelim adds the directional double subscript; `huggett-incomplete-markets/` uses $`\lambda_L, \lambda_H`$ for the same two rates] |
| $`\Lambda`$ | discrete-time lottery weights | Young (2010) lottery transition matrix [prelim names this for the SSJ cross-reference; `sequence-space-jacobian-hank/` currently refers to it as `bar Lambda` in pseudocode only] |
| $`i^{\ast}`$ | row index | normalisation row replacing one redundant equation [prelim introduces this naming] |
| OU parameters $`(\kappa, \mu, \sigma)`$ | $`(1.0, 0.0, 0.5)`$ | mean reversion, long-run mean, volatility of the Ornstein-Uhlenbeck example |
| OU grid | 401 points on $`[-3, 3]`$ | uniform 1D grid for the OU stationary solve |
| Ramsey grid | 200 points on $`[0.5, 15.0]`$ | uniform 1D capital grid reused from the upwind prelim |
| Joint grid | 200 asset points $`\times`$ 2 income | uniform 1D asset grid with the two-state income chain |

The symbol $`A`$ collides with the lead matrix of a rational-expectations system in the linearised DSGE block of the catalog. Each scope labels its $`A`$ on first use; the dense Huggett and Aiyagari-HACT tutorials use $`A`$ in the same continuous-time meaning as this prelim.

## Solution Method

### Stationary solve by row replacement

The discretised stationary KFE is $`A^{\top} g = 0`$ with $`\sum_i g_i  \Delta x = 1`$. The matrix $`A`$ has zero row sums by construction, so $`A^{\top}`$ has a non-trivial right null space spanned by the stationary density. The trick is to fold the normalisation into the linear system by replacing one row of $`A^{\top}`$ with the normalisation constraint. Concretely: pick a row index $`i^{\ast}`$, set row $`i^{\ast}`$ of $`A^{\top}`$ to the unit row $`e_{i^{\ast}}^{\top}`$, set the right-hand side to $`e_{i^{\ast}}`$ with all other entries zero, and solve the resulting non-singular sparse linear system by LU. Rescale the solution by $`(\sum_i g_i  \Delta x)^{-1}`$ so the density integrates to one. The helper `lib.finite_differences.stationary_distribution` packages this recipe.

```text
Algorithm: stationary distribution by row replacement
Inputs    sparse generator A (zero row sums), grid spacing dx, fix-row index i*
Output    probability vector g with sum_i g_i * dx = 1

form AT as the transpose of A
overwrite row i_star of AT with the unit row that has a 1 in column i_star and 0 elsewhere
form right-hand side b as the unit vector with a 1 in entry i_star and 0 elsewhere
solve AT g = b by sparse LU
clip negative entries to zero (rounding cleanup)
rescale g by dividing through by the integral sum_i g_i times dx
return g
```

### Operator duality across the HJB and the KFE

The same sparse matrix carries information in two directions. The HJB step inverts $`(\rho I - A)`$ to step the value function backward in pseudo-time. The KFE step inverts (a modified) $`A^{\top}`$ to find the stationary density. The Results section visualises this side by side: the spy pattern of $`A`$ and its transpose share the same nonzero structure, and the entry at $`(i, j)`$ of $`A`$ becomes the entry at $`(j, i)`$ of $`A^{\top}`$. This is the elegance of the continuous-time framework noted by Achdou et al. (2022) and used throughout the dense Huggett and Aiyagari-HACT tutorials.

### Multi-state generalisation via Kronecker blocks

The 1D solve generalises by block assembly. For an income chain with generator $`Q \in \mathbb{R}^{N \times N}`$ and asset-axis generators $`A_1, \dots, A_N`$ at each income state, the joint generator is $`\mathrm{diag}(A_1, \dots, A_N) + Q \otimes I_n`$. The block-diagonal piece advances assets within each income state. The Kronecker piece shuffles income across states at every asset level. The same single-row replacement pins the joint stationary scale.

## Results

### Ornstein-Uhlenbeck stationary density against the analytic Gaussian

The first example builds $`A`$ for the mean-reverting OU drift $`s(x) = -\kappa  (x - \mu)`$ on a uniform 401-point grid over $`[-3, 3]`$, adds a centered second-difference diffusion block scaled by $`\sigma^2 / 2`$, and solves $`A^{\top} g = 0`$. The analytic stationary density of an OU process on the real line is Gaussian with mean $`\mu`$ and variance $`\sigma^2 / (2 \kappa)`$. The numerical density matches the analytic Gaussian (renormalised to the bounded interval) in shape, location, and spread. The sup-norm gap is $`6.2 \times 10^{-3}`$ on the working grid. The first moment matches to four decimals. The second moment is $`0.127`$ against an analytic target of $`0.125`$. The small residual gap is the boundary truncation: the analytic Gaussian has unbounded support and the discrete solve restricts the density to $`[-3, 3]`$.

<img src="figures/stationary-density-ou.png" alt="OU stationary density from A^T g = 0 vs analytic Gaussian" width="80%">

### Sparse-matrix pattern of A and its transpose

The OU example produces a sparse $`401 \times 401`$ generator with at most three nonzeros per row: the diagonal and the two neighbours. The spy pattern of $`A`$ has its mass on the main diagonal, one super-diagonal, and one sub-diagonal. The spy pattern of $`A^{\top}`$ has the same shape because transposing a tridiagonal matrix preserves tridiagonality. The entries differ between the two: every super-diagonal entry of $`A`$ becomes a sub-diagonal entry of $`A^{\top}`$ at the mirrored index, and vice versa. The structural shape is what is shared.

<img src="figures/sparse-A-pattern.png" alt="Spy pattern of A and A^T" width="90%">

### Ramsey toy: stationary distribution from the upwind HJB generator

The second example feeds the upwind generator from prelim #1's toy Ramsey HJB into the same stationary solve. The Ramsey drift is deterministic and points toward the steady state from both sides, so the analytic stationary distribution is a point mass at $`k^{\ast} = (\alpha/(\rho+\delta))^{1/(1-\alpha)} \approx 7.40`$. On a discrete grid the stationary density concentrates almost entirely on a single node next to the steady-state location, with two adjacent nodes carrying small residual mass picked out by the row-replacement normalisation. The drift panel below confirms the sign change at the analytic steady state: drift is positive to the left and negative to the right. The figure makes the operator-duality concrete on the same generator that drove the HJB solve in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/): one matrix, one set of upwind branches, two equations.

<img src="figures/ramsey-stationary.png" alt="Ramsey toy stationary density and drift" width="80%">

### Operator duality on a six-node grid

The annotated 6-node visualisation shows the same matrix entries on the left as $`A`$ for the HJB direction and on the right as $`A^{\top}`$ for the KFE direction. Where $`A_{ij}`$ is the rate at which mass flows from $`i`$ to $`j`$, the transpose entry $`A^{\top}_{ji}`$ is the rate at which value at $`j`$ depends on value at $`i`$ in the backward direction. The same numerical entries appear in both panels, mirrored across the diagonal. This is the matrix-algebra version of the statement that the forward operator for the density and the backward operator for the value function are adjoints under one transposition.

<img src="figures/operator-duality.png" alt="Operator duality: A for HJB vs A^T for KFE on a small grid" width="95%">

### Joint asset-income stationary density

The third example assembles a joint $`400 \times 400`$ generator over 200 asset nodes and 2 income states. Each asset block uses a stylised mean-reverting drift toward an income-specific target plus a small diffusion to keep the joint density smooth. The Kronecker income block $`Q \otimes I_n`$ shuffles mass between the two income states at every asset level at Poisson rates $`\lambda_{LH} = \lambda_{HL} = 1.2`$. The joint stationary solve recovers two unimodal densities, each concentrated near its income-specific target, with marginal income masses $`p_L = p_H = 0.5`$ matching the symmetric jump-rate calibration. This is the block-matrix construction that [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/) uses with an $`N`$-state chain.

<img src="figures/joint-density.png" alt="Joint asset-income stationary density from A_joint^T g = 0" width="80%">

## Takeaway

Mass conservation in continuous time gives one matrix that serves two equations. The same sparse upwind generator that the HJB step inverts also defines the forward operator for the cross-sectional density. Transposing it and replacing one row with a normalisation constraint turns the singular system into a sparse linear solve whose answer is the stationary distribution.

This is the workhorse used by the dense Huggett solve in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/) and the multi-state Aiyagari-HACT solve in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/). The same logic in discrete time becomes the Young (2010) lottery iteration that the steady-state block of [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/) uses.

## References

1. Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86. The canonical economic reference for the $`A`$ vs $`A^{\top}`$ duality in computational heterogeneous-agent macroeconomics.
2. Pavliotis, G. A. (2014). *Stochastic Processes and Applications: Diffusion Processes, Fokker-Planck and Langevin Equations.* Springer. Chapters 2-3 derive the forward equation from the Chapman-Kolmogorov identity.
3. Gardiner, C. (2004). *Handbook of Stochastic Methods*, 3rd ed. Springer. Section 3.2 derives the Fokker-Planck equation from mass balance with physical intuition.
4. Young, E. R. (2010). "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell-Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34(1), 36-41. The lottery iteration that is the discrete-time analog of the continuous-time stationary solve developed here.
- **See also.** The upwind generator $`A`$ assembled here is built in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/). The single-income-state $`A^{\top} g = 0`$ solve appears in equilibrium form in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/). The multi-state generator construction with $`Q`$ on a richer Rouwenhorst chain appears in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/). The discrete-time lottery iteration that is the analog of this stationary solve appears in the steady-state block of [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/).
