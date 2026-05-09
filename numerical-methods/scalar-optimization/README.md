# One-Dimensional Optimization for Bellman Inner Steps

> Golden section search and Newton-on-FOC find the per-state cake-eating maximum, with a log-utility closed-form check.

## Overview

The inner step in cake-eating value function iteration is a one-dimensional maximization on a bounded interval.

$$\max_{c \in [0, W]} u(c) + \beta\, V(W - c).$$

Under log utility the closed-form inner optimum is $c^{\ast} = (1 - \beta) W$.

Golden section search contracts a bracket $[a, b]$ around the maximum. Newton on the first-order condition extrapolates a quadratic surrogate at the current iterate.

Golden section needs only unimodality and is globally safe. Newton is locally fast and depends on the starting point.

## Equations

Let $u(c) = \log c$ and let $V$ be the closed-form cake-eating value
function

$$V(W) = \frac{\log((1-\beta) W)}{1-\beta} + \frac{\beta \log \beta}{(1-\beta)^2}.$$

The inner Bellman objective is

$$g(c) = \log c + \beta\, V(W - c),$$

with derivatives

$$g'(c) = \frac{1}{c} - \frac{\beta}{1 - \beta}\, \frac{1}{W - c},
\qquad
g''(c) = -\frac{1}{c^2} - \frac{\beta}{1 - \beta}\, \frac{1}{(W - c)^2} < 0.$$

The objective is strictly concave on $(0, W)$, so the maximum is unique.
Setting $g'(c) = 0$ gives the closed-form policy

$$c^{\ast} = (1 - \beta) W.$$

Golden section search shrinks a bracket $[a, b]$ around the maximum using
$\phi = (\sqrt{5} - 1)/2 \approx 0.618$:

$$c_n = b_n - \phi (b_n - a_n),
\qquad d_n = a_n + \phi (b_n - a_n).$$

Here $c_n$ and $d_n$ are the left and right probe points at iteration $n$, distinct from the control variable $c$.

The next bracket keeps whichever endpoint is closer to the larger of
$g(c_n)$ and $g(d_n)$, contracting the width by a factor $\phi$ each step.

Newton on the FOC follows the tangent of $g'$ at the current iterate:

$$x_{n+1} = x_n - \frac{g'(x_n)}{g''(x_n)}.$$

Equivalently, Newton maximizes a parabolic surrogate
$q_n(c) = g(x_n) + g'(x_n)(c - x_n) + \tfrac{1}{2} g''(x_n)(c - x_n)^2$.

## Model Setup

| Symbol | Value | Role |
|--------|-------|------|
| $\beta$ | 0.9 | Discount factor; closed-form inner share is $1 - \beta$ |
| $W$ | 1.0 | Wealth at the inner state being optimized |
| $c^{\ast}$ | 0.1000 | Closed-form inner argmax $(1 - \beta) W$ |
| Bracket $[a_0, b_0]$ | $[1e-03,\, 0.9990]$ | Initial unimodal bracket for golden section |
| Newton start $x_0$ | 0.05 | Starting iterate for Newton-on-FOC |
| Tolerance $\varepsilon$ | 1e-10 | Stopping rule on bracket width and on $g'(x_n)$ |

## Solution Method

Both methods solve the same maximization. Golden section needs only that $g$ is unimodal on the bracket. Newton on the FOC needs $g'$ and $g''$.

**Golden section search.** Contract a bracket using the golden ratio so one interior point is reused each step.

```text
Algorithm: Golden section search
Input : a, b with g unimodal on [a, b]; tolerance eps
Output: c_n
  phi <- (sqrt(5) - 1) / 2
  c   <- b - phi (b - a)
  d   <- a + phi (b - a)
  for n = 1, 2, ... :
      if g(c) > g(d): b <- d
      else          : a <- c
      recompute c, d
      stop when (b - a) < eps
```

**Newton on the FOC.** Step along the tangent of $g'$ at the current iterate. Equivalently, jump to the argmax of a parabolic surrogate of $g$.

```text
Algorithm: Newton on FOC
Input : x_0; tolerance eps; g', g''
Output: x_n
  for n = 0, 1, ... :
      x_{n+1} <- x_n - g'(x_n) / g''(x_n)
      stop when |g'(x_n)| < eps
```

Starting from the bracket $[1e-03,\, 0.9990]$, golden section converges in **48 iterations** with residual $|g'(c)| =$ **1.77e-07**. From $x_0 = 0.05$ Newton converges in **6 iterations** with residual **0.00e+00**.

## Results

The inner objective $g(c)$ peaks at $c^{\ast} = 0.100$.

The first six golden-section brackets sit below the curve. Each step contracts the bracket by a factor $\phi \approx 0.618$ while keeping the maximum inside.

<img src="figures/golden-section-trace.png" alt="Cake-eating inner objective $g(c)$ with the first golden-section brackets contracting around $c^{\ast}$" width="80%">

Each Newton iterate defines a parabolic surrogate $q_n$ that matches $g$ in value, slope, and curvature.

Newton jumps to the argmax of $q_n$. From $x_0 = 0.05$ the iterates close in on $c^{\ast}$ within a handful of steps.

<img src="figures/newton-step.png" alt="Newton parabolic surrogates at the first iterates of the inner Bellman maximization" width="80%">

Golden section needs **48 bracket halvings** and Newton needs only **6 steps** from the same calibration.

The right panel shows the trade-off. Golden-section counts are flat across starting points: the bracket is the same. Newton counts depend on $x_0$, and **3 of 9** starts overshoot outside $(0, W)$ and diverge (hatched bars marked DNC).

<img src="figures/convergence.png" alt="Distance from the closed-form inner argmax (left) and Newton sensitivity to starting point (right)" width="80%">

The table summarises both solves on the same calibration. Both land within the chosen tolerance of the closed-form inner argmax.

**Golden section vs Newton-on-FOC on the cake-eating inner step**

| Method         | Start          |   Iterations |   Final residual |   Error in c | Convergence rate   |
|:---------------|:---------------|-------------:|-----------------:|-------------:|:-------------------|
| Golden section | [1e-03, 0.999] |           48 |         1.77e-07 |     1.59e-09 | linear (phi)       |
| Newton on FOC  | x_0 = 0.05     |            6 |         0        |     0        | quadratic          |

## Takeaway

Golden section is the safe default for a one-state Bellman inner step: it only needs unimodality and a bracket, and contracts at a fixed factor regardless of where the optimum sits.

Newton on the FOC is much faster when $g'$ and $g''$ are available and $x_0$ is inside the basin of attraction. A far-off start makes the parabolic extrapolation overshoot outside the feasible interval.

Cake-eating and consumption-savings VFI use golden section for this reason. Smooth problems with cheap derivatives can graduate to Newton.

## References

- Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 10.
- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 4.
