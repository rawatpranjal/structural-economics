# Upwind Finite Differences for First-Order PDEs and the HJB State Constraint

## Overview

A continuous-time control problem on a bounded interval reduces to a first-order Hamilton-Jacobi-Bellman partial differential equation. Information flows along the policy-implied drift, whose sign can change across the interval. The numerical scheme must honour that sign.

A symmetric solver picks up checkerboard oscillations that grow with iteration count. A fixed one-sided solver propagates information against the natural flow at one end. The upwind rule picks the one-sided difference whose neighbour the state is moving toward.

At a lower-bounded state, the Kuhn-Tucker condition swaps the policy that would push the state through the floor for the policy that holds it there. The numerical scheme must enforce the same clip.

The generic operator drives the Ramsey HJB in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) and the Huggett incomplete-markets equilibrium in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/). The payoff here is threefold: forward, backward, and upwind one-sided differences; the sparse generator that the implicit HJB solve inverts; the state-constraint clip that turns the lower-boundary policy into one that respects the floor.

## Preliminary readings

- [`dynamic-programming/optimal-growth/`](../../dynamic-programming/optimal-growth/)

## Equations

Place a uniform grid $`x_1 < x_2 < \cdots < x_n`$ on $`[\underline x, \overline x]`$ with spacing $`\Delta x`$. Let $`v_i = v(x_i)`$ be the value at node $`i`$, and $`s_i = s(x_i)`$ the policy-implied drift, positive when the state moves right.

The forward and backward one-sided slopes at an interior node are

```math
D^{+} v_i = \frac{v_{i+1} - v_i}{\Delta x},
\qquad
D^{-} v_i = \frac{v_i - v_{i-1}}{\Delta x}.
```

Each slope uses one neighbour. The central slope $`(v_{i+1} - v_{i-1}) / (2 \Delta x)`$ weights both equally. It is unstable here because HJB information propagates along the drift, not against it.

The upwind rule selects the slope whose neighbour the state is moving toward,

```math
D v_i =
\begin{cases}
D^{+} v_i & \text{if } s_i > 0,\\
D^{-} v_i & \text{if } s_i < 0,\\
0 & \text{if } s_i = 0.
\end{cases}
```

To compute the selector, solve the first-order condition with each one-sided slope, evaluate the implied drift, and keep the side whose drift has the matching sign. Where neither sign holds, the node sits at a local steady state with zero-drift policy. Results colours each node by branch.

Split the drift into positive and negative parts $`s^{+}_i = \max(s_i, 0)`$, $`s^{-}_i = \min(s_i, 0)`$. The sparse generator $`A`$ has super-diagonal $`s^{+}_i / \Delta x`$, sub-diagonal $`-s^{-}_i / \Delta x`$, with main-diagonal chosen so each row sums to zero. With one state, $`A`$ is tridiagonal.

```math
A_{i, i-1} = -\frac{s^{-}_i}{\Delta x},
\qquad
A_{i, i+1} = \frac{s^{+}_i}{\Delta x},
\qquad
A_{i, i} = -A_{i, i-1} - A_{i, i+1}.
```

This is a continuous-time Markov-chain generator on the grid: zero row sums, non-positive diagonal, off-diagonals non-negative. The implicit HJB step inverts $`(\rho I - A)`$ as a sparse tridiagonal solve. Its transpose is the Kolmogorov forward operator, a duality the KFE prelim develops.

At $`x_1 = \underline x`$, the backward slope is undefined. The boundary forcing rule uses $`D^{+} v_1`$ when the resulting drift is non-negative. Otherwise the constraint binds. The Kuhn-Tucker condition is

```math
s_1 \geq 0
\quad\Longleftrightarrow\quad
v'(\underline x) \geq u'(c_{\mathrm{floor}}),
```

where $`c_{\mathrm{floor}}`$ is the consumption that zeroes the drift at the boundary, defined implicitly by $`s(c_{\mathrm{floor}}) = 0`$ given $`x = \underline x`$.

Equality holds when the constraint is slack. Strict inequality holds when the household would dissave further. The scheme overrides the lower-boundary policy with $`c_{\mathrm{floor}}`$ whenever the unconstrained forward drift would be negative. An analogous rule applies at $`x_n`$.

The HJB is a first-order PDE whose classical solution may not exist where the value function has kinks. The viscosity-solution framework of Crandall, Evans, and Lions (1984) admits non-differentiable value functions. It selects the economically meaningful root. The upwind scheme is *monotone* in their sense, so its fixed point converges to the viscosity solution under refinement. The centred scheme is non-monotone, which explains the failure-mode comparison below.

## Model Setup

The illustrative model is a Ramsey HJB on a single capital state with log utility, Cobb-Douglas production, and depreciation. The calibration exposes the upwind branches and the boundary forcing rule. Economic claims belong in the dense Ramsey tutorial.

| Symbol | Value or set | Role |
|---|---|---|
| $`x`$ | grid node | generic 1D state (prelim notation, adopted by dense tutorials) |
| $`k`$ | $`[0.5, 15.0]`$ | capital, instance of $`x`$ (from `hjb-growth/`) |
| $`a`$ | (Huggett asset grid) | asset, instance of $`x`$ (from `huggett-incomplete-markets/`) |
| $`\underline a`$ | left endpoint of asset grid | lower state-constraint boundary (from `huggett-incomplete-markets/`) |
| $`\underline x, \underline k`$ | left endpoint of generic grid | lower grid endpoint when no constraint binds |
| $`v(x)`$ | grid array | value function |
| $`s(x)`$ | grid array | policy-implied drift, signed (Huggett writes $`s_i(a)`$ with an income index) |
| $`D^{+}, D^{-}`$ | linear operators | forward and backward one-sided differences |
| $`A`$ | sparse $`n \times n`$ | upwind generator on the 1D grid (Huggett: $`\mathbf{A}^n`$; `hjb-growth/`: $`G^n`$, rename to $`A`$ pending) |
| $`\rho`$ | 0.05 | continuous-time discount rate |
| $`\alpha`$ | 0.36 | capital share in Cobb-Douglas production |
| $`\delta`$ | 0.05 | depreciation rate |
| $`\Delta`$ | 1000 | implicit pseudo-time step |
| Working grid | 200 points | uniform on $`[0.5, 15.0]`$ |
| Reference grid | 4000 points | uniform on the same interval, refinement benchmark |
| HJB tolerance | $`10^{-7}`$ | sup-norm on successive value iterates |

The symbol $`A`$ here (upwind generator) collides with $`A`$ in the linearised DSGE prelim (lead matrix in a rational-expectations system). Each scope labels the matrix on first use.

## Solution Method

Implicit upwind iteration. At each pseudo-time step the solver forms forward and backward slopes at every node, computes the implied drift for each side, and picks the side whose drift carries the matching sign. The implied consumption defines $`A`$, and the next value iterate satisfies a sparse linear system.

The matrix $`(\rho I + I / \Delta - A)`$ is strictly diagonally dominant for any positive pseudo-time step, so the solve is unconditionally stable regardless of $`\Delta`$. Large $`\Delta`$ drives the update toward a Newton step with the policy frozen. The Ramsey application in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) collects the implementation details.

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

The KT clip at the left endpoint enforces a state constraint when present. When the forward-slope policy at $`\underline x`$ would push the state left, the override sets consumption to the zero-drift value. The `lib.finite_differences.kt_state_constraint_clip` helper exposes this operation. In the Ramsey calibration below the steady state is interior, so the clip does not bind. `run.py` exercises it on a synthetic input to verify the helper. The clip binds at the borrowing limit in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/), where households would dissave below $`\underline a`$ if unconstrained.

The implicit scheme is unconditionally stable. Under an explicit pseudo-time step the same operator obeys a CFL-like condition $`\Delta t \lesssim \Delta x / \max_i |s_i|`$. The failure-mode figure compares explicit centred against explicit upwind at the same step size. The centred iterates diverge; the upwind iterates remain bounded.

## Results

The HJB converges in seven implicit iterations on the working grid. The sup-norm change at stop is $`9.5 \times 10^{-10}`$, well below the $`10^{-7}`$ tolerance. The reference grid also converges in seven iterations. Doubling the grid does not double the iteration count because each step is close to a Newton step.

<img src="figures/value-and-drift.png" alt="Value function and drift sign coloured by upwind branch" width="80%">

The value function is increasing and concave in capital. The drift is positive below the steady state, negative above. The selector colours each node green for forward, red for backward. The zero-drift fallback fires at exactly one node near the steady state. The branch transition is monotone, no flickering.

<img src="figures/error-vs-reference.png" alt="Sup-norm policy error vs grid size" width="70%">

The second diagnostic is refinement. Sup-norm policy error against the fine-grid reference decays at the first-order rate $`1/n`$ on average, with some non-monotonicity because the reference grid is itself discretised. First-order convergence is the expected rate for a one-sided difference on a smooth value function. The diagnostic confirms that the upwind selection rule introduces no non-convergent constant.

<img src="figures/failure-naive-central.png" alt="Naive explicit central differences blow up while explicit upwind stays bounded" width="100%">

The failure-mode panel makes the case for upwinding directly. From the same myopic guess at the same explicit step size, the centred slope develops large spikes and hits the y-axis clip within a few dozen steps. The upwind slope marches toward the converged value function and stays bounded. The discretisation respects the direction of information flow.

## Takeaway

The upwind rule turns an unstable discretisation of a first-order PDE into a monotone scheme with a sparse generator. The state-constraint clip turns a constraint that the policy might violate into one the policy respects exactly. Both ideas come from one operator that the dense HJB and KFE tutorials reuse.

The transpose of that generator is the forward operator for the cross-sectional density. The next prelim, [`heterogeneous-agents/kolmogorov-forward-equation/`](../../heterogeneous-agents/kolmogorov-forward-equation/), develops the duality.

## References

1. Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86. Primary reference for upwinding HJB equations in continuous-time heterogeneous-agent macroeconomics.
2. Achdou, Y., and Capuzzo-Dolcetta, I. (2010). "Mean Field Games: Numerical Methods." *SIAM Journal on Numerical Analysis* 48(4), 1136-1162. Stability and consistency of upwind schemes for first-order HJB equations.
3. Crandall, M. G., Evans, L. C., and Lions, P.-L. (1984). "Some Properties of Viscosity Solutions of Hamilton-Jacobi Equations." *Transactions of the American Mathematical Society* 282(2), 487-502. Monotone schemes converging to viscosity solutions of first-order HJB equations.
4. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press. Chapter 4 develops upwind methods for advection equations as a pedagogical anchor.
- **See also.** The same upwind generator and KT clip drive the Ramsey HJB solve in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) and the Huggett borrowing-limit clip and asset-grid generator in [`heterogeneous-agents/huggett-incomplete-markets/`](../../heterogeneous-agents/huggett-incomplete-markets/).
