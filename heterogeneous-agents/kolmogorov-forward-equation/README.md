# Kolmogorov Forward Equation and the Stationary Wealth Distribution

## Overview

The Kolmogorov forward equation is the partial differential equation that tracks how a density of agents evolves over time, as each agent's state drifts according to its own dynamics. Its stationary form is the density that no longer changes. That density is what closes a Huggett or Aiyagari steady state. It is the long-run cross section of assets and income.

Mass conservation drives the derivation. No household appears or vanishes, so the integral of the density over the full state space stays at one. The density at any point changes only because agents arrive at that point or depart from it. Discretising this bookkeeping on the same grid as the HJB yields a sparse linear system. Its stationary solution is the cross section.

The same sparse matrix serves both halves of the equilibrium. The upwind generator built in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/) discretises the HJB step that solves for the household value function. Its transpose discretises the KFE step that solves for the household density. One matrix, two equations, two solves. This single object discretises the stationary equilibrium of the dense Huggett tutorial in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/) and the continuous-time Aiyagari tutorial in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/). The discrete-time analog is the Young (2010) lottery iteration in the steady-state block of [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/).

## Preliminary readings

- [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/)

## Equations

We want the law of motion for the cross-sectional density. Let $`t`$ index time. Let $`x`$ be a one-dimensional state on a bounded interval $`[\underline x, \overline x]`$. Let $`g(x, t)`$ be the density of agents at state $`x`$ at time $`t`$. Each agent's state evolves with drift $`s(x)`$ and diffusion of volatility $`\sigma \geq 0`$. Mass conservation says the total number of agents in any sub-interval changes only by inflow at one boundary and outflow at the other. Taking the derivative of that balance with respect to time and length gives the continuity equation,

```math
\frac{\partial g}{\partial t}(x, t) =
-\frac{\partial}{\partial x}\big[s(x)  g(x, t)\big]
+ \frac{\sigma^2}{2}  \frac{\partial^2 g}{\partial x^2}(x, t) .
```

The first term reads as follows. The product $`s(x)  g(x, t)`$ is the flux: the rate at which agents cross point $`x`$. Its spatial derivative measures whether more agents arrive at $`x`$ than leave it. The second term is the diffusion correction: it appears whenever each agent's state has a noise component, and it vanishes when $`\sigma = 0`$. In steady state the density no longer changes, so $`\partial_t g = 0`$. The stationary density then satisfies $`-\partial_x[s g] + (\sigma^2/2)  \partial_{xx} g = 0`$ with normalisation $`\int g  dx = 1`$.

We need a finite version of the stationary equation that a computer can solve. Discretise $`x`$ on a uniform grid $`x_1 < x_2 < \cdots < x_n`$ with spacing $`\Delta x`$. Let $`g_i = g(x_i)`$ be the density value at node $`i`$. The upwind generator built for the HJB step is the same matrix that drives the forward equation here. Call that matrix $`A`$. It is defined in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/). Its rows sum to zero, and its off-diagonal entries are non-negative. A matrix with those two properties is called a continuous-time Markov generator: multiplied by a probability vector, it returns the time derivative of that vector. Discrete mass conservation becomes a single linear system,

```math
A^{\top}  g = 0,
\qquad
\sum_i g_i  \Delta x = 1 .
```

This system is singular by construction. Zero row sums of $`A`$ mean $`A^{\top}`$ has the constant vector in its left null space and the stationary density $`g`$ in its right null space. Any scalar multiple of $`g`$ also solves $`A^{\top} g = 0`$. To pick out the one with unit mass, replace one row of $`A^{\top}`$ with the normalisation constraint. Solve the modified system. Rescale the answer to integrate to one. The dense Huggett and Aiyagari tutorials invoke this stationary KFE solve.

The same idea extends to a state with more than one component. The asset axis already has a generator $`A`$. We add an income axis with its own generator, and assemble a joint generator from the two pieces. Take a two-state Poisson income chain $`j \in \lbrace L, H \rbrace`$. Let $`\lambda_{LH}`$ be the jump rate from low income to high, and $`\lambda_{HL}`$ the rate from high to low. The income generator is

```math
Q =
\begin{pmatrix}
-\lambda_{LH} & \lambda_{LH} \\
 \lambda_{HL} & -\lambda_{HL}
\end{pmatrix} .
```

Off-diagonals are Poisson jump rates. Each row sums to zero, so $`Q`$ is a Markov generator in the same sense as $`A`$: it gives the time derivative of any income probability vector. To combine the asset and income generators on the joint state space, we need a block-matrix construction. The result is

```math
A_{\mathrm{joint}}
= \mathrm{diag}(A_{L}, A_{H}) + Q \otimes I_n ,
```

where $`A_j`$ is the asset-axis generator for income state $`j`$, and $`I_n`$ is the $`n \times n`$ identity. Reading the two terms: the block-diagonal piece $`\mathrm{diag}(A_L, A_H)`$ advances assets within each income state, leaving income alone. The Kronecker product $`Q \otimes I_n`$ is a block matrix that places each entry of $`Q`$ as a scaled $`n \times n`$ block; it shuffles agents across income states at every asset level, leaving the asset position alone. The stationary joint density solves $`A_{\mathrm{joint}}^{\top} g = 0`$ by the same single-row-replacement trick used in the 1D case. [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/) generalises this construction to an $`N`$-state Rouwenhorst-derived chain.

## Worked Numerical Example

To see the row-replacement trick at small scale, solve the stationary KFE for the two-state Poisson income chain with $`\lambda_{LH} = 1`$ and $`\lambda_{HL} = 2`$. The continuous-state asset generator is replaced by a $`2 \times 2`$ matrix, so the whole computation fits on one page.

The income generator from the Equations section becomes

```math
Q = \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix}.
```

Let $`\pi = (\pi_L, \pi_H)`$ be the stationary probability vector. The stationary KFE is $`Q^{\top} \pi = 0`$ with $`\pi_L + \pi_H = 1`$:

```math
Q^{\top} \pi
= \begin{pmatrix} -1 & 2 \\ 1 & -2 \end{pmatrix}
  \begin{pmatrix} \pi_L \\ \pi_H \end{pmatrix}
= \begin{pmatrix} 0 \\ 0 \end{pmatrix}.
```

Both rows give the same equation $`-\pi_L + 2 \pi_H = 0`$, confirming the system is singular: $`Q^{\top}`$ has the stationary vector in its right null space. Pick the normalisation row $`i^{\ast} = 2`$ and overwrite the second row of $`Q^{\top}`$ with $`(1, 1)`$, with the right-hand side set to 1 in that entry. The modified system is

```math
\begin{pmatrix} -1 & 2 \\ 1 & 1 \end{pmatrix}
\begin{pmatrix} \pi_L \\ \pi_H \end{pmatrix}
= \begin{pmatrix} 0 \\ 1 \end{pmatrix}.
```

Row 1 gives $`\pi_L = 2 \pi_H`$. Row 2 gives $`\pi_L + \pi_H = 1`$. Substituting, $`3 \pi_H = 1`$, so

```math
\boxed{\pi = \left(\tfrac{2}{3}, \tfrac{1}{3}\right)}.
```

The high-income state has rate 2 out and rate 1 in, so it holds half the mass of the low-income state. The single-row replacement turned a singular homogeneous system into a non-singular linear solve and produced a probability vector in one step. The discretised continuous-state case in Solution Method runs the same recipe on the sparse $`n \times n`$ generator $`A^{\top}`$.

## Model Setup

The prelim runs three examples that share one upwind generator. Symbols match the dense tutorials, so a reader walking to Huggett or Aiyagari-HACT sees no notational drift.

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

The symbol $`A`$ collides with the lead matrix of rational-expectations systems in the linearised DSGE block. Each scope labels its $`A`$ on first use. The dense Huggett and Aiyagari-HACT tutorials use $`A`$ in the continuous-time meaning of this prelim.

## Solution Method

### Stationary solve by row replacement

The discretised stationary KFE is $`A^{\top} g = 0`$ with $`\sum_i g_i  \Delta x = 1`$. Zero row sums of $`A`$ give $`A^{\top}`$ a non-trivial right null space spanned by the stationary density. Any scalar multiple of $`g`$ solves the homogeneous equation, so the system needs one extra equation to pick out the unit-mass solution. Fold the normalisation into the system by replacing one row of $`A^{\top}`$ with the constraint. Pick a row index $`i^{\ast}`$. Set row $`i^{\ast}`$ of $`A^{\top}`$ to the unit row $`e_{i^{\ast}}^{\top}`$. Set the right-hand side to $`e_{i^{\ast}}`$ with zeros elsewhere. Solve the non-singular sparse system by LU. Rescale by $`(\sum_i g_i  \Delta x)^{-1}`$ to integrate to one. The helper `lib.finite_differences.stationary_distribution` packages this recipe.

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

One sparse matrix serves two equations. The HJB step inverts $`(\rho I - A)`$ to step the value function backward in pseudo-time. The KFE step inverts a modified $`A^{\top}`$ to recover the stationary density. Results visualises both sides on the same grid. The spy patterns of $`A`$ and $`A^{\top}`$ share the same nonzero locations. Entry $`(i, j)`$ of $`A`$ moves to entry $`(j, i)`$ of $`A^{\top}`$. Achdou et al. (2022) emphasise this duality. It runs through the dense Huggett and Aiyagari-HACT tutorials.

### Multi-state generalisation via Kronecker blocks

The 1D solve extends to multi-component states by block assembly. Take an income chain with generator $`Q \in \mathbb{R}^{N \times N}`$ and asset-axis generators $`A_1, \dots, A_N`$, one per income state. The joint generator is $`\mathrm{diag}(A_1, \dots, A_N) + Q \otimes I_n`$. The block-diagonal piece advances assets inside each income state. The Kronecker piece moves agents between income states at every asset level. Single-row replacement on $`A_{\mathrm{joint}}^{\top}`$ pins the joint stationary scale.

## Results

### Ornstein-Uhlenbeck stationary density against the analytic Gaussian

The first example builds $`A`$ for the OU drift $`s(x) = -\kappa  (x - \mu)`$ on a 401-point grid over $`[-3, 3]`$. A centered second-difference diffusion block scaled by $`\sigma^2 / 2`$ is added, then $`A^{\top} g = 0`$ is solved. The analytic stationary density on the real line is Gaussian with mean $`\mu`$ and variance $`\sigma^2 / (2 \kappa)`$. The numerical density matches it (renormalised to the bounded interval) in shape, location, and spread. The sup-norm gap is $`6.2 \times 10^{-3}`$. The first moment matches to four decimals. The second moment is $`0.127`$ against an analytic target of $`0.125`$. The residual gap is boundary truncation: the Gaussian has unbounded support; the discrete solve restricts to $`[-3, 3]`$.

<img src="figures/stationary-density-ou.png" alt="OU stationary density from A^T g = 0 vs analytic Gaussian" width="80%">

### Sparse-matrix pattern of A and its transpose

The OU example produces a sparse $`401 \times 401`$ generator with at most three nonzeros per row: the diagonal and two neighbours. The spy pattern of $`A`$ has mass on the main diagonal, one super-diagonal, one sub-diagonal. Transposing a tridiagonal matrix preserves tridiagonality, so $`A^{\top}`$ has the same shape. Entries differ: every super-diagonal entry of $`A`$ becomes a sub-diagonal entry of $`A^{\top}`$ at the mirrored index. Only the structural shape is shared.

<img src="figures/sparse-A-pattern.png" alt="Spy pattern of A and A^T" width="90%">

### Ramsey toy: stationary distribution from the upwind HJB generator

The second example feeds the upwind generator from prelim #1's toy Ramsey HJB into the stationary solve. The Ramsey drift is deterministic, pointing toward the steady state from both sides. The analytic stationary distribution is a point mass at $`k^{\ast} = (\alpha/(\rho+\delta))^{1/(1-\alpha)} \approx 7.40`$. The numerical density concentrates on a single node next to $`k^{\ast}`$, with two neighbours carrying residual mass from the row-replacement normalisation. The drift panel confirms the sign change at $`k^{\ast}`$: positive on the left, negative on the right. The same generator drove the HJB solve in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/); one matrix, two equations.

<img src="figures/ramsey-stationary.png" alt="Ramsey toy stationary density and drift" width="80%">

### Operator duality on a six-node grid

The 6-node visualisation shows the same matrix entries as $`A`$ on the left (HJB direction) and as $`A^{\top}`$ on the right (KFE direction). $`A_{ij}`$ is the rate at which mass flows from $`i`$ to $`j`$. The transpose entry $`A^{\top}_{ji}`$ is the rate at which value at $`j`$ depends on value at $`i`$ backward. Both panels show the same numbers, mirrored across the diagonal. The forward operator for the density and the backward operator for the value function are adjoints under one transposition.

<img src="figures/operator-duality.png" alt="Operator duality: A for HJB vs A^T for KFE on a small grid" width="95%">

### Joint asset-income stationary density

The third example assembles a joint $`400 \times 400`$ generator over 200 asset nodes and 2 income states. Each asset block uses a stylised mean-reverting drift toward an income-specific target, plus a diffusion to keep the density smooth. The Kronecker income block $`Q \otimes I_n`$ shuffles mass between the income states at every asset level, at Poisson rates $`\lambda_{LH} = \lambda_{HL} = 1.2`$. The solve recovers two unimodal densities, each concentrated near its income-specific target, with marginal masses $`p_L = p_H = 0.5`$ matching the symmetric calibration. [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/) uses this block-matrix construction with an $`N`$-state chain.

<img src="figures/joint-density.png" alt="Joint asset-income stationary density from A_joint^T g = 0" width="80%">

## Takeaway

Mass conservation in continuous time yields one matrix that serves two equations. The upwind generator inverted by the HJB step also defines the forward operator for the cross-sectional density. Transposing it, then replacing one row with the normalisation constraint, turns a singular system into a sparse linear solve. The answer is the stationary distribution.

The dense Huggett solve in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/) and the multi-state Aiyagari-HACT solve in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/) use this workhorse. The discrete-time analog is the Young (2010) lottery iteration in the steady-state block of [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/).

## References

1. Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86. The canonical economic reference for the $`A`$ vs $`A^{\top}`$ duality in computational heterogeneous-agent macroeconomics.
2. Pavliotis, G. A. (2014). *Stochastic Processes and Applications: Diffusion Processes, Fokker-Planck and Langevin Equations.* Springer. Chapters 2-3 derive the forward equation from the Chapman-Kolmogorov identity.
3. Gardiner, C. (2004). *Handbook of Stochastic Methods*, 3rd ed. Springer. Section 3.2 derives the Fokker-Planck equation from mass balance with physical intuition.
4. Young, E. R. (2010). "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell-Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34(1), 36-41. The lottery iteration that is the discrete-time analog of the continuous-time stationary solve developed here.
- **See also.** The upwind generator $`A`$ assembled here is built in [`optimal-control/upwind-finite-differences/`](../../optimal-control/upwind-finite-differences/). The single-income-state $`A^{\top} g = 0`$ solve appears in equilibrium form in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/). The multi-state generator construction with $`Q`$ on a richer Rouwenhorst chain appears in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/). The discrete-time lottery iteration that is the analog of this stationary solve appears in the steady-state block of [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/).
