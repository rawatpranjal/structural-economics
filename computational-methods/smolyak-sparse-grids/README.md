# High-Dimensional Growth Policy by Smolyak Sparse Grids

## Overview

A planner manages four sectoral capital stocks under one common productivity shock. Output flows from each sector and feeds back into consumption and next-period capitals.

The state vector has five continuous coordinates. Tensor Chebyshev grids would charge $m^5$ nodes for $m$ points per dimension, which becomes wasteful long before the policy is captured.

Smolyak sparse grids keep only the tensor blocks that carry new interaction information at a chosen accuracy level. The node count then grows polynomially in dimension, and Chebyshev collocation still recovers a smooth saving rule.

## Equations

The planner chooses consumption $c$ and the next-period sectoral capitals
$k_i'$ to maximize expected log utility under a common productivity shock $z$:

$$
V(k, z) = \max_{c, k'} \left[ \log c + \beta \mathbb{E}[V(k', z') \mid z] \right],
\qquad
c + \sum_{i=1}^{N} k_i' = e^{z} \sum_{i=1}^{N} A_i k_i^{\alpha}.
$$

Productivity follows an AR(1) process $z' = \rho_z z + \sigma_z \varepsilon'$
with $\varepsilon' \sim \mathcal{N}(0, 1)$. The first-order conditions across
sectors give one Euler equation per capital:

$$
\frac{1}{c} = \beta \alpha A_i \mathbb{E}\left[ \frac{e^{z'} (k_i')^{\alpha - 1}}{c'} \,\middle|\, z \right],
\qquad i = 1, \dots, N.
$$

Taking the ratio of two sector FOCs eliminates the expectation. Because the
shock $z$ is common, the ratio of marginal products of capital must equal the
ratio of saving choices raised to $\alpha - 1$:

$$
\frac{A_i (k_i')^{\alpha - 1}}{A_j (k_j')^{\alpha - 1}} = 1,
\qquad
k_i' = \omega_i\, S,
\qquad
\omega_i = \frac{A_i^{1 / (1 - \alpha)}}{\sum_{j=1}^{N} A_j^{1 / (1 - \alpha)}}.
$$

Here $S \equiv \sum_i k_i'$ is total savings and $\omega_i$ is the sector
share. With $Z \equiv \sum_j A_j^{1 / (1 - \alpha)}$ the Euler equation
collapses to a single condition on $S$:

$$
\frac{1}{c} = \beta \alpha Z^{1 - \alpha} S^{\alpha - 1} \mathbb{E}\left[ \frac{e^{z'}}{c'} \,\middle|\, z \right].
$$

For this calibration the closed-form policy is $S^{\ast}(k, z) = \alpha \beta Y(k, z)$, with
$Y(k, z) = e^{z} \sum_i A_i k_i^{\alpha}$. Each sector then receives $k_i' = \omega_i S$ and the
planner consumes $c = (1 - \alpha \beta) Y$.

### Method 1: nested Chebyshev extrema nodes

The 1D nested rule uses $m_1 = 1$ and $m_i = 2^{i-1} + 1$ for $i \ge 2$:

$$
X_i = \lbrace -\cos(\pi (j - 1) / (m_i - 1)) : j = 1, \dots, m_i \rbrace,
\qquad
X_1 = \lbrace 0 \rbrace.
$$

The doubling rule gives $X_i \subset X_{i+1}$ so refinement does not throw away
prior function evaluations.

### Method 2: Smolyak sparse grid in d dimensions

Given dimension $d$ and level $\mu \ge 0$, admissible level multi-indices
$\mathbf{i} = (i_1, \dots, i_d)$ satisfy $i_k \ge 1$ and $\mu + 1 \le |\mathbf{i}| \le \mu + d$.
The Smolyak grid is the union of admissible tensor products:

$$
H(d, \mu) = \bigcup_{\mu + 1 \le |\mathbf{i}| \le \mu + d} X_{i_1} \times \cdots \times X_{i_d}.
$$

Concrete enumerations in $d = 2$: at $\mu = 1$ the admissible blocks are
$(1, 1), (1, 2), (2, 1)$, whose union has $5$ unique points (the center plus
two axis endpoints in each direction). At $\mu = 2$ the admissible blocks add
$(2, 2), (1, 3), (3, 1)$, whose union has $13$ unique points. Asymptotically
$|H(d, \mu)| = O(2^{\mu} d^{\mu} / \mu!)$, polynomial in $d$ for fixed $\mu$.

### Method 3: Smolyak polynomial basis and collocation

For each 1D Chebyshev degree $a \ge 0$ define the polynomial level
$\ell(a) = 0$ if $a = 0$, $\ell(1) = 1$, and $\ell(a) = \lceil \log_2 a \rceil$
for $a \ge 2$. The Smolyak polynomial basis is the set of tensor products
whose summed polynomial level fits within $\mu$:

$$
\mathcal{B}(d, \mu) = \left\lbrace T_{a_1}(x_1) \cdots T_{a_d}(x_d) : \sum_{k=1}^{d} \ell(a_k) \le \mu \right\rbrace.
$$

The basis matrix $\Phi_{n,k} = \prod_j T_{a_{k,j}}(x_n^{(j)})$ is square and
invertible at the Smolyak nodes. Collocation sets the policy residual to zero
at each node.

### Method 4: time iteration with Smolyak collocation

Let $\theta$ be the coefficient vector for $\log S(x; \theta) = \Phi(x) \theta$.
Time iteration freezes $\theta^{\text{old}}$ on the right-hand side of the
Euler equation, solves a Smolyak collocation problem for $\theta^{\text{new}}$,
and iterates until $\|\theta^{\text{new}} - \theta^{\text{old}}\|_{\infty}$
falls below tolerance.

## Model Setup

| Symbol | Object | Value |
|--------|--------|-------|
| $\beta$ | Discount factor | 0.95 |
| $\alpha$ | Capital share | 0.36 |
| $N$ | Sectors | 4 |
| $d$ | State dimension | 5 |
| $A$ | Sector productivities | (1.00, 0.90, 1.10, 0.80) |
| $\omega$ | Sector shares | (0.269, 0.228, 0.312, 0.190) |
| $\rho_z$ | Productivity persistence | 0.95 |
| $\sigma_z$ | Productivity innovation SD | 0.010 |
| $k^{ss}$ | Sectoral steady states | (0.187, 0.159, 0.217, 0.132) |
| Quadrature | Gauss-Hermite nodes for $z'$ | 3 |
| Smolyak levels solved | $\mu$ values | (1, 2, 3) |

## Solution Method

The Smolyak approximator targets $\log S(x; \theta) = \Phi(x) \theta$ on $[-1, 1]^{d}$ after a linear rescaling of the state. Solution proceeds in four steps that mirror the equations above.

### Method 1: nested Chebyshev extrema nodes

Nested nodes let the algorithm reuse function evaluations across refinement levels. Without nesting, every increase in $\mu$ would resample the model from scratch.

```text
Algorithm: Nested 1D Chebyshev extrema
Input: level i in {1, 2, 3, ...}
Output: node set X_i in [-1, 1] with |X_i| = m_i
1. if i == 1: return {0}
2. m_i <- 2^(i-1) + 1
3. for j = 1, ..., m_i:
     x_j <- -cos(pi (j - 1) / (m_i - 1))
4. return {x_1, ..., x_m_i}
```

Failure mode: using Chebyshev roots breaks nesting because root sets at different orders share no points.

### Method 2: Smolyak sparse grid in d dimensions

Smolyak keeps only the tensor blocks whose total level fits in the band $\mu + 1 \le |\mathbf{i}| \le \mu + d$. The union is deduplicated by nesting so each unique node is counted once.

```text
Algorithm: Smolyak grid H(d, mu)
Input: dimension d, level mu
Output: sparse grid H in [-1, 1]^d
1. H <- empty set
2. for every multi-index i = (i_1, ..., i_d) with i_k >= 1
       and mu + 1 <= |i| <= mu + d:
     G_i <- X_{i_1} x ... x X_{i_d}
     H <- H union G_i
3. return H (deduplicated)
```

Failure mode: forgetting to deduplicate produces a multiset that double-counts shared nodes, breaking the square structure of the collocation system.

### Method 3: Smolyak polynomial basis and collocation

The basis multi-indices are paired with the grid via the same admissibility rule, expressed in degree space. The resulting basis matrix is square and invertible at the Smolyak nodes.

```text
Algorithm: Smolyak collocation residual
Input: grid H = {x^(n)}, basis multi-indices A, model primitives
Output: coefficient vector theta
1. Build basis matrix Phi with Phi[n, k] = prod_j T_{A[k, j]}(x^(n)_j)
2. Define policy log S(x; theta) = Phi(x) theta
3. At each node x^(n), assemble the Euler residual R_n(theta)
4. Solve R(theta) = 0 for the coefficient vector
```

Failure mode: pairing a hyperbolic-cross basis with a Smolyak grid leaves Phi non-square and the solver stalls.

### Method 4: time iteration with Smolyak collocation

Time iteration decouples the nonlinear problem across nodes. At each node the Euler equation reduces to a one-dimensional condition on $S_n$ because the sector allocation $k_i = \omega_i S$ is fixed by the FOC ratio.

```text
Algorithm: Time iteration with Smolyak collocation
Input: grid H, basis Phi, shock nodes (eps_q, w_q), tol = 1e-7
Output: converged coefficients theta_star
1. theta_old <- initial guess (constant saving fraction)
2. repeat:
     for n = 1, ..., N:
       compute E_n = sum_q w_q exp(z'_q) / c'_q using theta_old
       solve S_n in (0, Y_n) from log Euler equation
     theta_new <- Phi^{-1} log S
     delta <- max_k |theta_new[k] - theta_old[k]|
     theta_old <- theta_new
   until delta < tol
3. return theta_old
```

Failure mode: updating $\theta^{\text{new}}$ inside the expectation (full Euler iteration) couples every node and the Jacobian becomes ill-conditioned at higher $\mu$.

## Results

Tensor grids charge an exponential price for each new state dimension. Smolyak nodes grow polynomially in dimension at a fixed accuracy level.

<img src="figures/grid-size-scaling.png" alt="Tensor and Smolyak node counts vs state dimension" width="80%">

In two dimensions Smolyak keeps the axis directions resolved at higher level and skips the interior tensor points that contribute little new information.

<img src="figures/smolyak-grid-2d.png" alt="Tensor versus Smolyak nodes in two dimensions" width="80%">

Across five state dimensions the level-2 Smolyak grid uses 61 nodes where a tensor of 5 points per dimension would charge 3,125.

**Tensor versus Smolyak node counts across dimensions**

|   Dimension d |   Smolyak level mu | Tensor nodes   | Smolyak nodes   |   Tensor / Smolyak ratio |
|--------------:|-------------------:|:---------------|:----------------|-------------------------:|
|             2 |                  1 | 9              | 5               |              1.8         |
|             2 |                  2 | 25             | 13              |              1.92        |
|             2 |                  3 | 81             | 29              |              2.79        |
|             4 |                  1 | 81             | 9               |              9           |
|             4 |                  2 | 625            | 41              |             15.24        |
|             4 |                  3 | 6,561          | 137             |             47.89        |
|             5 |                  1 | 243            | 11              |             22.09        |
|             5 |                  2 | 3,125          | 61              |             51.23        |
|             5 |                  3 | 59,049         | 241             |            245.02        |
|             6 |                  1 | 729            | 13              |             56.08        |
|             6 |                  2 | 15,625         | 85              |            183.82        |
|             6 |                  3 | 531,441        | 389             |           1366.17        |
|             8 |                  1 | 6,561          | 17              |            385.94        |
|             8 |                  2 | 390,625        | 145             |           2693.97        |
|             8 |                  3 | 43,046,721     | 849             |          50702.8         |
|            10 |                  1 | 59,049         | 21              |           2811.86        |
|            10 |                  2 | 9,765,625      | 221             |          44188.3         |
|            10 |                  3 | 3,486,784,401  | 1,581           |              2.20543e+06 |

A k_1 slice with the other sectors at their steady-state values shows the Smolyak approximations converging onto the closed-form total savings rule as the level rises.

<img src="figures/policy-slice.png" alt="Closed-form total savings vs Smolyak approximations" width="80%">

Off-grid Euler residuals at 10,000 Sobol test states confirm that higher-level Smolyak grids drive the worst-case error down by orders of magnitude.

<img src="figures/euler-errors.png" alt="Empirical CDF of Euler errors on a Sobol test set" width="80%">

The accuracy table shows that adding a Smolyak level cuts the worst-case Euler error sharply while runtime grows only with the modest increase in node count.

**Accuracy and runtime by Smolyak level**

| Method       |   Nodes |   Max Euler error |   Median Euler error |   Max savings error |   Seconds |
|:-------------|--------:|------------------:|---------------------:|--------------------:|----------:|
| Smolyak mu=1 |      11 |          0.012    |             0.00226  |            0.00911  |      0    |
| Smolyak mu=2 |      61 |          0.00104  |             0.000191 |            0.000933 |      0.01 |
| Smolyak mu=3 |     241 |          0.000165 |             9.49e-06 |            0.000166 |      0.04 |

With Smolyak level $\mu = 3$ on $d = 5$ states the policy is stored in 241 coefficients. The worst-case absolute Euler error on the 10,000-point Sobol test set is 1.65e-04 and the median is 9.49e-06. The same accuracy under a tensor Chebyshev grid would charge 59,049 nodes.

## Takeaway

Smolyak sparse grids replace the tensor exponential with a polynomial node count at the cost of a slightly slower spectral convergence rate. For the smooth policies that arise in growth, RBC, and life-cycle models the trade is favorable up to about 20 continuous states. Beyond that, adaptive sparse grids that refine only in directions where the residual is large recover a further gain. Brumm and Scheidegger (2017) develop the adaptive extension.

## References

- [Judd, K. L., Maliar, L., Maliar, S., and Valero, R. (2014). Smolyak Method for Solving Dynamic Economic Models. *Journal of Economic Dynamics and Control*, 44, 92-123.](https://doi.org/10.1016/j.jedc.2014.03.003)
- [Brumm, J. and Scheidegger, S. (2017). Using Adaptive Sparse Grids to Solve High-Dimensional Dynamic Models. *Econometrica*, 85(5), 1575-1612.](https://doi.org/10.3982/ECTA12216)
- [Malin, B. A., Krueger, D., and Kubler, F. (2011). Solving the Multi-Country Real Business Cycle Model Using a Smolyak-Collocation Method. *Journal of Economic Dynamics and Control*, 35(2), 229-239.](https://doi.org/10.1016/j.jedc.2010.09.011)
- [Smolyak, S. A. (1963). Quadrature and Interpolation Formulas for Tensor Products of Certain Classes of Functions. *Doklady Akademii Nauk SSSR*, 148, 1042-1045.](https://www.mathnet.ru/eng/dan28167)
