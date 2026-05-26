# Numerical Quadrature: Gauss-Hermite Nodes for Conditional Expectations

## Overview

The classical approach to Gaussian-weighted expectations is to evaluate the integrand on a uniform grid and sum with equal weights. Gauss (1814) showed that a far better strategy exists: choose both the nodes and weights to annihilate the integration error exactly for polynomials up to degree $`2N-1`$. The nodes are the roots of the $`N`$-th Hermite polynomial. The weights are determined by the same polynomials through the Christoffel-Darboux formula.

*Gauss-Hermite quadrature* places nodes where the Gaussian density concentrates mass and assigns weights that reflect that concentration. For smooth integrands, error decays geometrically in the node count $`N`$. Three nodes already integrate a degree-four polynomial exactly. Ten nodes reach machine precision for smooth Gaussian-decay integrands. For kinked or indicator functions the convergence slows, and equal-weight rules can outperform.

The same node-weight pair recurs throughout computational economics. Tauchen and Hussey (1991) use Gauss-Hermite nodes directly to approximate AR(1) transition probabilities. Smolyak sparse grids take tensor products of Gauss-Hermite rules for multi-dimensional integration. Bayesian optimisation computes expected improvement with a Gauss-Hermite sum over the posterior predictive. The node computation is the shared primitive across all three.

## Read before

- [Shock discretization: Tauchen and Rouwenhorst](../shock-discretization/README.md)
- [Smolyak sparse grids](../../computational-methods/smolyak-sparse-grids/README.md)

## Equations

The Gauss-Hermite identity covers integrals against the physicists' weight $`e^{-x^2}`$. For any function $`f`$ that grows no faster than the Gaussian weight decays,

```math
\int_{-\infty}^{\infty} f(x)\, e^{-x^2}\, dx \approx \sum_{n=1}^{N} w_n\, f(\xi_n),
```

where $`\{\xi_n\}_{n=1}^{N}`$ are the roots of the $`N`$-th physicists' Hermite polynomial $`H_N`$ and $`\{w_n\}`$ are the associated weights, given by $`w_n = 2^{N-1} N! \sqrt{\pi} / (N^2 H_{N-1}(\xi_n)^2)`$. The approximation is exact whenever $`f`$ is a polynomial of degree at most $`2N - 1`$. The error for smooth $`f`$ decays geometrically in $`N`$.

The *standard-normal form* integrates against the density $`\phi`$ rather than the physicists' weight. The change of variables $`x = z / \sqrt{2}`$ in the Gauss-Hermite identity gives

```math
\mathbb{E}_{Z \sim \phi}[g(Z)] = \frac{1}{\sqrt{\pi}} \sum_{n=1}^{N} w_n\, g(\sqrt{2}\,\xi_n).
```

The factor $`1/\sqrt{\pi}`$ normalises the physicists' weight to the standard-normal density. The transformed nodes $`\sqrt{2}\,\xi_n`$ concentrate around zero, with weights largest at the origin and falling away from it.

The AR(1) conditional expectation follows directly. Suppose $`z_{t+1} = \rho\, z_t + \sigma_\varepsilon\, \eta_{t+1}`$ with $`\eta_{t+1} \sim N(0,1)`$. The conditional expectation $`\mathbb{E}[f(z_{t+1}) \mid z_t]`$ is an integral against the standard-normal density of the shifted-and-scaled argument. Substituting $`\eta = \sqrt{2}\,\xi`$ into the AR(1) recursion gives the Gauss-Hermite estimator

```math
\mathbb{E}[f(z_{t+1}) \mid z_t] \approx \frac{1}{\sqrt{\pi}} \sum_{n=1}^{N} w_n\, f\!\left(\rho\, z_t + \sigma_\varepsilon \sqrt{2}\,\xi_n\right).
```

This converts the AR(1) conditional expectation into a weighted sum of $`f`$ evaluated at $`N`$ shifted, scaled nodes. The Tauchen-Hussey approximation of the AR(1) process discretises the state space with exactly these nodes and assigns transition probabilities from the same weights.

Stroud and Secrest (1966) establish that for an integrand in $`C^{2k}`$, the Gauss-Hermite error decays at rate $`O(N^{-2k})`$, which is faster than any polynomial in $`N`$ when $`f`$ is smooth. For non-smooth integrands (kinked functions, indicators) the error decays slowly and Simpson's rule on a finite truncation can outperform Gauss-Hermite.

## Worked Numerical Example

Take $`f(z) = z^2`$ and $`Z \sim N(0,1)`$. The exact expectation is $`\mathbb{E}[Z^2] = 1`$.

The standard 3-node Gauss-Hermite rule gives physicists' nodes $`x = (-1.2247,\, 0,\, 1.2247)`$ and weights that absorb the $`e^{-x^2}`$ factor. To integrate against the standard-normal density $`\phi`$ we apply the change of variables from Equations: transformed nodes $`\xi_i = \sqrt{2}\, x_i`$ and rescaled weights $`\tilde{w}_i = w_i / \sqrt{\pi}`$,

```math
\xi = (-1.7321,\; 0,\; 1.7321), \qquad \tilde{w} = (0.1667,\; 0.6667,\; 0.1667).
```

The *3-node Gauss-Hermite sum* is

```math
\hat{I}_{\mathrm{GH}} = \sum_{i=1}^{3} \tilde{w}_i\, f(\xi_i) = 0.1667 \cdot (-1.7321)^2 + 0.6667 \cdot 0^2 + 0.1667 \cdot (1.7321)^2.
```

Since $`(-1.7321)^2 = (1.7321)^2 = 3`$,

```math
\hat{I}_{\mathrm{GH}} = 0.1667 \cdot 3 + 0 + 0.1667 \cdot 3 = 0.500 + 0.500 = \boxed{1.000}.
```

Three nodes reproduce the exact answer because $`f(z) = z^2`$ is a polynomial of degree 2, which is below the $`2N - 1 = 5`$ exactness threshold for $`N = 3`$.

Now compare against a 3-point trapezoid rule on $`[-3, 3]`$ with nodes $`(-3, 0, 3)`$ and step $`h = 3`$. The trapezoid approximates $`\int f(z)\phi(z)\,dz`$ as

```math
\hat{I}_{\mathrm{trap}} = h \left[\frac{f(-3)\,\phi(-3)}{2} + f(0)\,\phi(0) + \frac{f(3)\,\phi(3)}{2}\right].
```

Substituting $`\phi(-3) = \phi(3) \approx 0.0044`$ and $`\phi(0) \approx 0.3989`$,

```math
\hat{I}_{\mathrm{trap}} = 3 \left[\frac{9 \cdot 0.0044}{2} + 0 \cdot 0.3989 + \frac{9 \cdot 0.0044}{2}\right] = \boxed{0.119}.
```

Gauss-Hermite gives the exact value. The trapezoid gives 0.119. The trapezoid places its outer nodes at $`\pm 3`$ where the Gaussian density is tiny, so the mass between nodes is poorly captured. Gauss-Hermite places its outer nodes at $`\pm 1.73`$, precisely where the $`z^2 \phi(z)`$ integrand peaks. Node placement determined by the Hermite polynomials, not by a uniform grid, is the source of the gain.

## Model Setup

The table records all objects and parameter values needed to reproduce the figures and tables in Results. The *Hermite roots* $`\xi_n`$ and weights $`w_n`$ are computed by `scipy.special.roots_hermite` and are not tuned by hand.

| Object | Value | Object | Value |
|---|---:|---|---:|
| Node counts $`N`$ in sweep | 3, 5, 10, 30 | AR(1) persistence $`\rho`$ | 0.9 |
| Hermite roots $`\xi_n`$ | `roots_hermite(N)` | AR(1) innovation s.d. $`\sigma_\varepsilon`$ | 0.5 |
| Test integrand 1 $`f_1(z) = z^6 - 3z^4 + 2`$ | exact at $`N \geq 4`$ | Test integrand 2 $`f_2(z) = e^{-z^2/4}`$ | smooth, non-polynomial |
| Test integrand 3 $`f_3(z) = \lvert z \rvert`$ | non-smooth (kink at 0) | AR(1) test function $`f(z') = e^{z'}`$ | MGF closed form |

## Solution Method

The computation has two stages: build the Gauss-Hermite rule once from the Hermite polynomial roots, then apply it to each integrand or AR(1) conditional expectation. The AR(1) stage is a *change of variables* that shifts and scales the nodes before evaluating the integrand.

```
   f, N nodes                    f, rho, sigma, z_t
        |                                |
        v                                v
  +-- GH quadrature --+         +-- AR(1) GH step --+
  | [ Hermite roots ] |         | [ shifted nodes ] |
  |        +          |         |  [ weighted sum ] |
  | [ weighted sum ]  |         +-------------------+
  +-------------------+                 |
        |                          E[f(z')|z_t]
    E_phi[f]
```

Real Python for the standard-normal expectation and the AR(1) conditional:

```python
from scipy.special import roots_hermite
import numpy as np

def gh_expectation(f, N):
    xi, w = roots_hermite(N)           # nodes and weights for exp(-x^2) weight
    z_nodes = np.sqrt(2.0) * xi        # map to N(0,1) standard deviation scale
    return float(np.dot(w, f(z_nodes)) / np.sqrt(np.pi))

def gh_ar1_conditional(z_grid, rho, sigma_eps, f, N):
    xi, w = roots_hermite(N)
    # z_prime[i, j] = rho * z_grid[i] + sigma_eps * sqrt(2) * xi[j]
    z_prime = rho * z_grid[:, None] + sigma_eps * np.sqrt(2.0) * xi[None, :]
    return np.dot(f(z_prime), w) / np.sqrt(np.pi)   # shape (len(z_grid),)
```

Smoothness governs convergence speed. When $`f`$ is differentiable to high order, the error decays geometrically. When $`f`$ has a kink, Simpson's rule on a finite truncation can be more accurate at the same evaluation count. The change of variables above relies on the AR(1) innovation being Gaussian. Non-Gaussian innovations require a different node family.

## Results

The error-versus-nodes curve compares Gauss-Hermite against Simpson's rule and Monte Carlo on the three test integrands. The *smoothness gap* (the difference in convergence rate between polynomial and non-smooth integrands) is the central diagnostic.

![Log-log plot of absolute error versus node count N for Gauss-Hermite, Simpson, and Monte Carlo on three integrands](figures/error-vs-nodes.png)

For the polynomial $`f_1`$, Gauss-Hermite is exact to machine precision at $`N = 5`$. Simpson's rule still has error of 1.4 at $`N = 5`$ and only catches up at $`N = 30`$. For the smooth non-polynomial $`f_2`$, Gauss-Hermite reaches machine precision at $`N = 30`$, with Simpson within a factor of two. For the non-smooth $`f_3`$, Gauss-Hermite's convergence slows substantially. Its error sits near $`10^{-2}`$ at $`N = 30`$, while Simpson's rule achieves near $`10^{-5}`$ at the same evaluation count, a three-order-of-magnitude advantage on this integrand. The mainline takeaway is that Gauss-Hermite's advantage is smoothness-dependent.

The node and weight visualisation makes the geometry of the rule concrete.

![Gauss-Hermite nodes and weights at N=10 with the standard normal density overlaid](figures/gh-nodes-weights.png)

The transformed nodes $`\sqrt{2}\,\xi_n`$ concentrate around the origin where the standard-normal density is largest. The weights, scaled by $`1/\sqrt{\pi}`$ to integrate against $`\phi`$, are also largest near the origin. At $`N = 10`$ the nodes span roughly $`\pm 4`$ standard deviations, which is enough to integrate against the full Gaussian density with negligible truncation error.

The AR(1) conditional expectation example uses $`f(z') = \exp(z')`$ and compares the Gauss-Hermite estimate against a Monte Carlo benchmark.

![AR(1) conditional expectation E[exp(z prime) given z]: GH estimate, MC estimate, and closed form, with relative error panel](figures/ar1-conditional-error.png)

At $`N = 10`$, Gauss-Hermite recovers the closed-form expectation $`\exp(\rho\, z + \sigma_\varepsilon^2 / 2)`$ to machine precision across the entire $`z`$ grid. Monte Carlo with $`10^4`$ draws sits at relative error $`10^{-3}`$. For dynamic-programming problems that compute conditional expectations many times inside an outer loop, the gain from replacing a large Monte Carlo sample with a handful of Gauss-Hermite nodes is several orders of magnitude per evaluation.

### Method comparison

| Integrand | GH error at $`N=10`$ | Simpson error at $`N=10`$ | MC error at $`N=10`$ |
|:---|---:|---:|---:|
| Polynomial $`f_1`$ | $`< 10^{-15}`$ (exact) | $`\approx 10^{-1}`$ | $`\approx 10^{0}`$ |
| Smooth $`f_2`$ | $`\approx 10^{-9}`$ | $`\approx 10^{-5}`$ | $`\approx 10^{-1}`$ |
| Non-smooth $`f_3`$ | $`\approx 10^{-2}`$ | $`\approx 10^{-3}`$ | $`\approx 10^{-1}`$ |

## Takeaway

*Gauss-Hermite quadrature* converts a Gaussian-weighted integral into a short weighted sum at the roots of the Hermite polynomials. For smooth integrands the convergence is geometric in the node count. For kinked or indicator integrands it slows toward the rates that Simpson's rule and Monte Carlo achieve. The quantitative gap between a large Monte Carlo sample and a handful of Gauss-Hermite nodes, striking for smooth functions, collapses for non-smooth ones. The change-of-variables identity for AR(1) conditional expectations is the downstream primitive that makes sparse-grid solvers, Tauchen-Hussey discretisation, and Bayesian acquisition integrals tractable.

## See also

- [Aiyagari saving and capital-market clearing](../../dynamic-programming/aiyagari/README.md)
- [Shock discretization: Tauchen and Rouwenhorst](../shock-discretization/README.md)
- [Smolyak sparse grids](../../computational-methods/smolyak-sparse-grids/README.md)

## References

- Gauss, C. F. (1814). *Methodus nova integralium valores per approximationem inveniendi*. Commentationes Societatis Regiae Scientiarum Gottingensis Recentiores. Original derivation of the Gaussian quadrature rule and the node-weight relationship via orthogonal polynomials.
- Hammersley, J. M. and Handscomb, D. C. (1964). *Monte Carlo Methods*. Methuen, London. Foundation text for Monte Carlo integration, establishing the $`O(N^{-1/2})`$ convergence rate that Gauss-Hermite dominates for smooth integrands.
- Stroud, A. H. and Secrest, D. (1966). *Gaussian Quadrature Formulas*. Prentice-Hall. Tabulated nodes and weights for the Hermite weight; convergence theory for smooth integrands.
- Tauchen, G. and Hussey, R. (1991). Quadrature-Based Methods for Obtaining Approximate Solutions to Nonlinear Asset Pricing Models. *Econometrica*, 59(2), 371-396. GH nodes for AR(1) Markov-chain approximation.
- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Chapter 7. Gaussian quadrature for economic applications.
- Heer, B. and Maußner, A. (2009). *Dynamic General Equilibrium Modeling*, 2nd edition. Springer, Chapter 6. Gauss-Hermite inside parametric-expectations and projection methods.
