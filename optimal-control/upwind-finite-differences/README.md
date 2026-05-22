# Upwind Finite Differences for First-Order PDEs and the HJB State Constraint

## Overview

A continuous-time control problem on a bounded interval reduces to a first-order Hamilton-Jacobi-Bellman partial differential equation. The state has a policy-implied drift whose sign can change across the interval. The numerical scheme that solves the HJB has to honour that sign, because information in a first-order PDE flows along the drift.

The economic question is sharp. A solver that treats both neighbours symmetrically is unstable: value iterates pick up checkerboard oscillations that grow with iteration count. A solver that always uses the same neighbour is biased: it propagates information against the natural flow at one end of the interval. The upwind rule resolves both pathologies by choosing the one-sided difference whose neighbour the state is moving toward.

A second question appears at the boundary. When the state is bounded below, the Kuhn-Tucker condition replaces the policy that would push the state through the floor with the policy that holds it there. The numerical scheme must enforce the same clip or the resulting policy violates the constraint.

This tutorial builds the generic operator on a 1D state grid and demonstrates both fixes on a toy HJB. The same machinery appears in the Ramsey HJB tutorial in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) and in the Huggett incomplete-markets equilibrium in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/). Those tutorials apply the operator to specific economic settings. They now cross-reference this prelim for the derivation.

The reader takes away three things. A working definition of forward, backward, and upwind one-sided differences. The construction of the sparse generator that the implicit HJB solve inverts. The state-constraint clip that turns the lower-boundary policy into one that respects the floor.

## Preliminary readings

- [`dynamic-programming/optimal-growth/`](../../dynamic-programming/optimal-growth/)

## Equations

Place a uniform grid $`x_1 < x_2 < \cdots < x_n`$ on the bounded interval $`[\underline x, \overline x]`$ with spacing $`\Delta x`$. Let $`v_i = v(x_i)`$ denote the value at node $`i`$ and let $`s_i = s(x_i)`$ denote the policy-implied drift at the same node. The drift carries the sign of state motion, positive when the state moves right and negative when it moves left.

The forward and backward one-sided slopes at an interior node are

```math
D^{+} v_i = \frac{v_{i+1} - v_i}{\Delta x},
\qquad
D^{-} v_i = \frac{v_i - v_{i-1}}{\Delta x}.
```

Each slope uses one neighbour. The central slope $`(v_{i+1} - v_{i-1}) / (2 \Delta x)`$ uses both with equal weight and is unstable in this setting because information in the HJB propagates along the drift, not against it.

The upwind rule selects the slope whose neighbour the state is moving toward,

```math
D v_i =
\begin{cases}
D^{+} v_i & \text{if } s_i > 0,\\
D^{-} v_i & \text{if } s_i < 0,\\
0 & \text{if } s_i = 0.
\end{cases}
```

The selector is computed from candidate drifts. Solve the first-order condition with each one-sided slope, evaluate the implied drift, and keep the side whose drift has the matching sign. Where neither sign holds, the node sits at a local steady state and the policy is the zero-drift fallback. The Results section colours each node by which branch the rule selected.

Once the upwind slope is fixed at every node, split the drift into its positive and negative parts $`s^{+}_i = \max(s_i, 0)`$ and $`s^{-}_i = \min(s_i, 0)`$. The sparse generator $`A`$ has super-diagonal entry $`s^{+}_i / \Delta x`$, sub-diagonal entry $`-s^{-}_i / \Delta x`$, and main-diagonal entry chosen so each row sums to zero. With one state, $`A`$ is tridiagonal.

```math
A_{i, i-1} = -\frac{s^{-}_i}{\Delta x},
\qquad
A_{i, i+1} = \frac{s^{+}_i}{\Delta x},
\qquad
A_{i, i} = -A_{i, i-1} - A_{i, i+1}.
```

This is the generator of a continuous-time Markov chain on the grid: zero row sums, non-positive diagonal, off-diagonals non-negative. The implicit HJB step inverts $`(\rho I - A)`$ at every iteration, which is a sparse tridiagonal solve. The same matrix transposes into the forward operator for the Kolmogorov forward equation, a duality this tutorial states and the KFE prelim develops.

At the lower boundary $`x_1 = \underline x`$, the backward slope is undefined because there is no node $`x_0`$. The boundary forcing rule replaces $`D v_1`$ with $`D^{+} v_1`$ when the resulting drift is non-negative. When the policy implied by $`D^{+} v_1`$ would have negative drift, the state would leave the interval and the constraint binds. The Kuhn-Tucker condition is

```math
s_1 \geq 0
\quad\Longleftrightarrow\quad
v'(\underline x) \geq u'(c_{\mathrm{floor}}),
```

where $`c_{\mathrm{floor}}`$ is the consumption that makes the drift zero at the boundary, defined implicitly by $`s(c_{\mathrm{floor}}) = 0`$ given $`x = \underline x`$.

Equality holds when the constraint is slack. Strict inequality holds when the household would prefer to dissave further. The numerical scheme enforces the condition by overriding the lower-boundary policy with $`c_{\mathrm{floor}}`$ whenever the unconstrained forward drift would be negative. An analogous one-sided rule applies at $`x_n`$.

A note on what the upwind rule is approximating. The HJB is a first-order PDE whose classical solution may not exist where the value function has kinks. The viscosity-solution framework of Crandall, Evans, and Lions (1984) is the standard relaxation: it admits non-differentiable value functions and selects the economically meaningful root. The upwind scheme is a *monotone* discretisation in their sense, so its fixed point converges to the viscosity solution as the grid is refined. The centred scheme is non-monotone and has no such convergence guarantee, which is the underlying reason the failure-mode comparison below comes out as it does.

## Model Setup

The illustrative model is a Ramsey HJB on a single capital state with log utility, Cobb-Douglas production, and depreciation. The point of the calibration is to expose the upwind branches and the boundary forcing rule, not to make economic claims that the dense Ramsey tutorial already covers.

| Symbol | Value or set | Role |
|---|---|---|
| $`x`$ | grid node | generic 1D state [prelim introduces this; dense tutorials adopt it on cut] |
| $`k`$ | $`[0.5, 15.0]`$ | capital, instance of $`x`$ [from `optimal-control/hjb-growth/`] |
| $`a`$ | (Huggett asset grid) | asset, instance of $`x`$ [from `heterogeneous-agents/huggett-incomplete-markets/`] |
| $`\underline a`$ | left endpoint of asset grid | lower state-constraint boundary [from `huggett-incomplete-markets/`] |
| $`\underline x, \underline k`$ | left endpoint of generic grid | lower grid endpoint when no economic constraint binds [prelim introduces this for the generic exposition] |
| $`v(x)`$ | grid array | value function [from both dense tutorials] |
| $`s(x)`$ | grid array | policy-implied drift, signed [from `huggett-incomplete-markets/`, which uses $`s_i(a)`$ with an income index] |
| $`D^{+}, D^{-}`$ | linear operators | forward and backward one-sided differences [prelim introduces this; dense tutorials adopt it on cut] |
| $`A`$ | sparse $`n \times n`$ | upwind generator on the 1D grid [prelim introduces this name; `huggett-incomplete-markets/` uses $`\mathbf{A}^n`$ with an iteration index for the same object; `hjb-growth/` currently uses $`G^n`$ and the rename to $`A`$ is scheduled with the next pass on that tutorial] |
| $`\rho`$ | 0.05 | continuous-time discount rate [from `hjb-growth/`] |
| $`\alpha`$ | 0.36 | capital share in Cobb-Douglas production [from `hjb-growth/`] |
| $`\delta`$ | 0.05 | depreciation rate [from `hjb-growth/`] |
| $`\Delta`$ | 1000 | implicit pseudo-time step [from `hjb-growth/`] |
| Working grid | 200 points | uniform on $`[0.5, 15.0]`$ |
| Reference grid | 4000 points | uniform on the same interval, used as a refinement benchmark |
| HJB tolerance | $`10^{-7}`$ | sup-norm on successive value iterates |

The collision between the symbol $`A`$ here (upwind generator) and $`A`$ in the linearised DSGE prelim (lead matrix in a rational-expectations system) is unavoidable. Each scope labels the matrix on first use.

## Solution Method

The HJB is solved by implicit upwind iteration. At each pseudo-time step the solver forms the forward and backward slopes at every node, computes the implied drift for each side, and picks the side whose drift carries the matching sign. The implied consumption then defines the upwind generator $`A`$ and the next value iterate satisfies a sparse linear system.

The implicit step has two reasons to recommend it. The matrix $`(\rho I + I / \Delta - A)`$ is strictly diagonally dominant for any positive pseudo-time step, so the linear solve is unconditionally stable regardless of $`\Delta`$. Taking $`\Delta`$ large drives the update toward a Newton step on the HJB residual with the policy frozen. The same machinery is used in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) for the Ramsey example, with the implementation details collected in the dense tutorial.

```text
Algorithm: implicit upwind HJB iteration on a 1D grid
Inputs    grid {x_i}, primitives (rho, ...), pseudo-time step Delta, tolerance eps
Output    value v, policy c, drift s, selector (forward / backward / zero)

initialise v from a myopic flow value
repeat
    for each interior node i:
        compute D+ v_i and D- v_i
        derive candidate consumptions and candidate drifts from each slope
        if forward drift positive: use D+
        elif backward drift negative: use D-
        else: use zero-drift fallback
    at the left endpoint: force forward, then apply the KT clip if needed
    at the right endpoint: force backward
    build the sparse generator A from upwind drifts
    solve [(1 / Delta + rho) I - A] v_new = u(c) + v / Delta
    if max abs(v_new - v) below eps: stop
    v = v_new
```

The KT clip at the left endpoint enforces the state constraint when one is present. When the policy implied by the forward slope at $`\underline x`$ would push the state left, the override sets consumption to the zero-drift value. The `lib.finite_differences.kt_state_constraint_clip` helper exposes the same operation for downstream tutorials. In the Ramsey calibration used below the steady state is interior, so the unconstrained forward drift at the grid floor is positive and the clip does not bind. The clip is exercised on a synthetic input in `run.py` to verify the helper, and it binds actively in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/) at the borrowing limit, where households would prefer to dissave below $`\underline a`$ if unconstrained.

A note on stability. The implicit scheme above is unconditionally stable. The same operator under an explicit pseudo-time step obeys a CFL-like condition $`\Delta t \lesssim \Delta x / \max_i |s_i|`$. The failure-mode figure compares the explicit centred update against the explicit upwind update at the same step size and shows the centred iterates diverging while the upwind iterates remain bounded.

## Results

The HJB converges in seven implicit iterations to a sup-norm change of about $`9.5 \times 10^{-10}`$ on the working grid, well below the $`10^{-7}`$ stopping criterion declared in the Model Setup. The same calibration on the reference grid also converges in seven iterations. This is typical of the implicit scheme: doubling the grid does not roughly double the iteration count because each step is close to a Newton step.

<img src="figures/value-and-drift.png" alt="Value function and drift sign coloured by upwind branch" width="80%">

The value function is increasing in capital. It is also concave. The drift is positive below the steady state and negative above it. The upwind selector colours each node green when the forward difference was used and red when the backward difference was used. The zero-drift fallback fires at exactly one node near the steady state, where neither one-sided drift carries the matching sign. The transition between branches is monotone, with no flickering.

<img src="figures/error-vs-reference.png" alt="Sup-norm policy error vs grid size" width="70%">

Refinement is the second diagnostic. The sup-norm error of the policy on a sequence of coarsenings against the fine-grid reference solve decays at approximately the first-order rate $`1/n`$ on average, with some non-monotonicity across grids because the reference grid is itself discretised. First-order convergence is the expected behaviour for a one-sided difference on a smooth value function. The diagnostic confirms that the upwind selection rule does not introduce a non-convergent constant. Higher-order schemes can improve the rate. The tutorial flags them as out of scope.

<img src="figures/failure-naive-central.png" alt="Naive explicit central differences blow up while explicit upwind stays bounded" width="100%">

The failure-mode panel makes the case for upwinding directly. Starting from the same myopic guess and taking explicit pseudo-time steps at the same step size, the centred slope generates iterates that develop large spikes and hit the y-axis clip within a few dozen steps. The upwind slope produces iterates that march toward the converged value function and stay bounded. The contrast is the practical reason monotone upwind schemes are the standard for first-order HJB equations: the discretisation respects the direction in which information flows.

## Takeaway

The upwind rule turns a numerically unstable discretisation of a first-order PDE into a monotone scheme with a sparse generator. The state-constraint clip at the lower boundary turns a constraint that the policy might violate into one that the policy respects exactly. Both ideas come from one operator, so the dense HJB and KFE tutorials can build on a shared discretisation rather than re-deriving it.

The same generator transposed gives the forward operator for the cross-sectional density, a duality the next prelim in [`heterogeneous-agents/kolmogorov-forward-equation/`](../../heterogeneous-agents/kolmogorov-forward-equation/) develops.

## References

1. Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86. Primary reference for upwinding HJB equations in continuous-time heterogeneous-agent macroeconomics.
2. Achdou, Y., and Capuzzo-Dolcetta, I. (2010). "Mean Field Games: Numerical Methods." *SIAM Journal on Numerical Analysis* 48(4), 1136-1162. Stability and consistency of upwind schemes for first-order HJB equations.
3. Crandall, M. G., Evans, L. C., and Lions, P.-L. (1984). "Some Properties of Viscosity Solutions of Hamilton-Jacobi Equations." *Transactions of the American Mathematical Society* 282(2), 487-502. Monotone schemes converging to viscosity solutions of first-order HJB equations.
4. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press. Chapter 4 develops upwind methods for advection equations as a pedagogical anchor.
- **See also.** The same upwind generator and KT clip drive the Ramsey HJB solve in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) and the Huggett borrowing-limit clip and asset-grid generator in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/).
