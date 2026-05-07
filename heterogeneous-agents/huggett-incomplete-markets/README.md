# Huggett Equilibrium and the Risk-Free Rate

> A continuous-time pure-exchange economy where idiosyncratic income risk and a hard borrowing limit pin down the equilibrium bond return below the rate of time preference, $r^{\ast} < \rho$.

## Overview

The Huggett (1993) economy is the canonical pure-exchange model of an incomplete-markets
risk-free rate. A continuum of households face idiosyncratic income risk, can lend or
borrow in a single non-state-contingent bond at return $r$, and are bounded below by a
hard borrowing limit $a \geq \underline a$. The bond is in zero net supply: every unit lent
matches a unit borrowed, so the cross-sectional asset demand has to integrate to zero.
The price that makes that integral vanish is the equilibrium return $r^{\ast}$.

The interesting comparison is the complete-markets benchmark. Without idiosyncratic risk,
the household Euler equation pins $r$ at exactly the rate of time preference $\rho$.
Once insurance is incomplete, the marginal value of buffer wealth is strictly positive
at any household with little cash on hand, and aggregate desired bond holdings are
positive at $r = \rho$. Markets only clear at a strictly lower bond return; the size of
the wedge $\rho - r^{\ast}$ is the price the economy puts on the missing insurance
contract. In this calibration the wedge is $\rho - r^{\ast} = 0.0150$ (about $30\%$ of $\rho$) — a quantitatively non-trivial precautionary discount on the risk-free rate even with only two income states.

The continuous-time HJB/KFE representation of [Achdou et al. (2022)](https://benjaminmoll.com/lectures/)
is what `run.py` here implements. It carries two equilibrium objects in parallel: the
asset drift implied by household consumption decisions, $s_i(a) = z_i + ra - c_i(a)$, and
the stationary distribution that drift induces, $g_i(a)$. Bisection on $r$ closes the
loop. The neighbouring [Aiyagari tutorial](../../dynamic-programming/aiyagari/) runs
the same equilibrium logic in discrete time, but closes the model with a representative
firm so the asset in fixed supply is physical capital $K$ rather than a bond — a useful
distinction when comparing $r^{\ast}$ to $1/\beta - 1$ (the discrete-time mirror of $\rho$).
The household block in this tutorial is solved by an HJB upwind finite-difference scheme;
the Euler-equation analogues for discrete time, [EGP](../endogenous-grid-points/) and
[EEI](../envelope-equation-iteration/), live in the same section and target the same
buffer-stock policy.

## Equations

A continuum of households is indexed by current income state $i \in \{L, H\}$ with
endowment $z_i$ and Poisson switching intensity $\lambda_i$ into the other state $j$. A
household holds assets $a$, earns the bond return $r$, and consumes $c_i(a)$; the asset
position evolves deterministically between income jumps according to the drift

$$\dot a \;=\; s_i(a) \;=\; z_i + r\,a - c_i(a), \qquad a \geq \underline a.$$

The state space is the half-line $a \in [\underline a, \infty)$ together with the
two-point chain over $i$. With CRRA flow utility $u(c) = c^{1-\sigma}/(1-\sigma)$ and
discount rate $\rho > 0$, the Hamilton-Jacobi-Bellman equation is

$$\rho\,V_i(a) \;=\; \max_{c > 0}\,
[\,u(c) \;+\; V_i'(a)\,(z_i + r\,a - c) \;+\; \lambda_i\,(V_j(a) - V_i(a))\,].$$

The first two terms are the certainty-equivalent piece — current utility plus the
deterministic continuation value implied by $\dot a$ — and the last term is the
expected jump in the value function when income switches. The first-order condition
delivers the Euler/envelope identity

$$c_i(a) \;=\; [V_i'(a)]^{-1/\sigma}, \qquad
s_i(a) \;=\; z_i + r\,a - c_i(a).$$

The borrowing constraint enters as a *state constraint*: at $a = \underline a$ the asset
drift cannot point further left, so

$$s_i(\underline a) \;\geq\; 0
\quad\Longleftrightarrow\quad
V_i'(\underline a) \;\geq\; u'(z_i + r\,\underline a),$$

with equality whenever the constraint is slack and inequality when it binds. This is the
continuous-time counterpart to the Kuhn-Tucker margin in [EGP](../endogenous-grid-points/).

The cross-sectional density $g_i(a)$ on $(\underline a, \infty)$ satisfies the Kolmogorov
Forward Equation

$$0 \;=\; -\frac{\partial}{\partial a}[s_i(a)\,g_i(a)]
\;-\; \lambda_i\,g_i(a) \;+\; \lambda_j\,g_j(a),
\qquad \int g_L + g_H \;=\; 1,$$

with a delta-mass component at $\underline a$ for income states whose drift hits the
constraint with positive probability. Equilibrium in the bond market is the
zero-net-supply condition

$$S(r) \;\equiv\; \int_{\underline a}^{\bar a} a\,[g_L(a) + g_H(a)]\,da \;=\; 0.$$

In the deterministic mirror of the model, the household Euler equation reduces to
$\dot c / c = (r - \rho)/\sigma$, and a non-degenerate stationary equilibrium exists
only at $r = \rho$. Incomplete markets break that result: $S(\rho) > 0$ because households
want positive precautionary asset holdings, so equilibrium requires $r^{\ast} < \rho$.
Quantifying that wedge is the headline output here.

## Model Setup

The calibration is intentionally compact — two income states, symmetric switching, a
one-dimensional asset grid — so that the precautionary-saving mechanism is the only
source of action. Larger income chains (Tauchen or Rouwenhorst) plug into the same
solver but obscure the Huggett wedge with calibration noise.

| Object | Value | Role |
|---|---:|---|
| Discount rate $\rho$ | 0.05 | Continuous-time time preference; complete-markets benchmark for $r$ |
| CRRA $\sigma$ | 2.0 | Curvature; sets the precautionary motive and Euler curvature |
| Income endowments $(z_L, z_H)$ | (0.1, 0.2) | Two-state Poisson chain with stationary mean $\bar z = 0.1500$ |
| Switching intensities $(\lambda_L, \lambda_H)$ | (1.2, 1.2) | Symmetric jumps; expected duration in each state $1/\lambda_i \approx 0.83$ |
| Borrowing limit $\underline a$ | -0.15 | Hard lower bound; chosen so $z_L + r\underline a > 0$ at the equilibrium $r$ |
| Upper bound $\bar a$ | 5.0 | Set wide enough that the right tail of $g_i$ is numerically zero |
| Working asset grid | 2000 pts | Uniform on $[\underline a, \bar a]$; HJB upwind scheme |
| Reference asset grid | 6000 pts | Audit solve at the same calibration; defines the discretisation gap |
| Implicit step $\Delta$ | 1000 | Large step keeps the implicit HJB update close to a Newton step on $V$ |
| HJB tolerance | 1e-06 | Sup-norm on successive value functions |
| Bisection tolerance | $10^{-5}$ | On the bond-market residual $\lvert S(r)\rvert$ |

The two switching intensities are equal so the income chain has symmetric stationary
probabilities $p_L = p_H = 0.5$, and expected income is $\bar z = 0.1500$.
At the working solution the cross-sectional probabilities recover this prediction to
$|p_L - 0.5| = 1.11e-16$, a basic sanity check on the KFE solve.

## Solution Method

The household block at a candidate $r$ is solved by an implicit upwind
finite-difference scheme on the asset grid. The two delicate pieces are the choice of
derivative for $V_i'(a)$ at each grid point and the construction of the discrete generator
$A$ that approximates $\partial/\partial a[s_i(a)\,\cdot] + \lambda$-switching.

**Upwind derivative.** At each $(a_k, i)$ the algorithm computes both the forward and
backward finite-difference approximations of $V_i'(a_k)$, converts each into a candidate
consumption via $c = (V_i')^{-1/\sigma}$, and then picks the one whose implied drift
$s_i(a_k) = z_i + r a_k - c$ points *into* the grid (forward when $s>0$, backward when $s<0$).
This upwinding is what makes the discrete generator a sub-stochastic Markov matrix and
keeps the borrowing limit from being crossed numerically. Centred differences would
break that property and admit unphysical reflections off $\underline a$.

**Implicit step.** Stack $V$ over income states into a vector of length $2I$. The
implicit HJB update is

$$[(\Delta^{-1} + \rho)\,\mathbf I - A^{n}]\,V^{n+1} \;=\; u(c^{n}) + \Delta^{-1} V^{n},$$

where $A^{n}$ is the upwind transition generator built from the current drift and the
income-switching intensities $\lambda_i$. With a large step $\Delta = 1000$ this update
behaves like a Newton step on the steady-state HJB $\rho V = u(c) + AV$, so convergence
is essentially quadratic; in this calibration the inner loop terminates in single-digit
iterations.

**KFE.** Once $V$ converges, the same generator delivers the stationary distribution as
the left null space of $A$: solve $A^{\top} g = 0$ subject to $\int g = 1$. The system
is singular (the generator has a zero eigenvalue), so the algorithm pins one row of
$A^{\top}$ and rescales the solution to integrate to one.

**Equilibrium.** The bond-market excess demand $S(r) = \int a\,(g_L + g_H)\,da$ is
strictly increasing in $r$ on the relevant range — higher returns make saving more
attractive and discourage borrowing — so a single bisection on $[r_{\min}, r_{\max}]$
locates $r^{\ast}$ to any desired tolerance.

```text
Algorithm: Huggett equilibrium by HJB-KFE bisection
Inputs    asset grid {a_k}, income states (z_L, z_H), Poisson rates (lambda_L, lambda_H),
          primitives (rho, sigma, a_min), bisection bracket [r_lo, r_hi]
Output    equilibrium r*, value V_i(a), policies c_i(a), s_i(a), density g_i(a)

repeat (outer bisection)
    r = 0.5 * (r_lo + r_hi)

    # Inner HJB by implicit upwind finite differences
    initialise V_i(a) = u(z_i + r a) / rho                         # myopic guess
    repeat
        for each (a_k, i):
            dVf = (V_i(a_{k+1}) - V_i(a_k)) / da                   # forward
            dVb = (V_i(a_k) - V_i(a_{k-1})) / da                   # backward
            cf  = (dVf)^(-1/sigma);   sf = z_i + r a_k - cf
            cb  = (dVb)^(-1/sigma);   sb = z_i + r a_k - cb
            if   sf > 0: c_i(a_k) = cf;            drift = sf      # upwind forward
            elif sb < 0: c_i(a_k) = cb;            drift = sb      # upwind backward
            else        : c_i(a_k) = z_i + r a_k;  drift = 0       # local steady state

        build A from upwind drifts plus income-switching block
        solve [(1/Delta + rho) I - A] V_new = u(c) + V / Delta     # implicit step
        if max|V_new - V| < eps_HJB: break
        V <- V_new

    # KFE for the stationary distribution
    fix one row of A^T to pin scale; solve A^T g = e_fix; renormalise so int g = 1

    # Bond-market excess demand
    S(r) = sum_k a_k * (g_L(a_k) + g_H(a_k)) * da
    if |S(r)| < eps_S: return r, V, c, s, g
    if S(r) > 0: r_hi = r              # too much saving; lower r
    else       : r_lo = r              # too much borrowing; raise r
```

**Working solve.** The HJB inner loop converged in **9 iterations** (sup-norm change $7.91e-07$), and the outer bisection located $r^{\ast} = 0.03499$ with bond-market residual $5.43e-06$ on the 2000-point asset grid.

**Reference solve and audit.** The same equilibrium is recomputed on a $I_{\rm ref} = 6000$-point reference grid as a discretisation audit. The reference equilibrium is $r^{\ast}_{\rm ref} = 0.03572$, so the interest-rate gap is $|r^{\ast}_{2000} - r^{\ast}_{6000}| = 7.30e-04$. On the active asset range $a \in [\underline a, 1]$ the working savings policy
$s_i(a)$ lies within $1.57e-03$ of the interpolated reference policy in
sup norm; the value function gap is $2.23e-01$, or about $0.16\%$ relative to the value scale. The grid convergence of $r^{\ast}$ in this
calibration is genuinely slow — uniform-grid HJB is first-order accurate at the borrowing
limit, where the policy has a kink — so refining beyond $I = 6000$ would
shift $r^{\ast}$ further toward $\rho$ at a rate that scales like $1/I$. A non-uniform
asset grid concentrated near $\underline a$ (cf. Achdou et al. 2022, App. C) tightens this
quickly. For the qualitative wedge $\rho - r^{\ast} > 0$ and the cross-sectional shapes
the working grid is more than enough.

## Results

The first figure plots $V_L(a)$ and $V_H(a)$ at $r^{\ast}$ on the working grid (solid) and the reference equilibrium values on the finer grid (dashed). Both curves are increasing and concave in $a$, with $V_H > V_L$ uniformly because income enters cash on hand linearly. Near the borrowing limit $\underline a$ both curves steepen sharply: a marginal dollar of wealth there relaxes the state constraint $s_i(\underline a) \geq 0$ and buys insurance against staying in the low-income state. The reference and working curves are visually indistinguishable on the active range — the relative gap in $V$ is about $0.16\%$ — while the small vertical level shift at the right edge reflects the discretisation in the equilibrium price ($|r^{\ast}_{2000} - r^{\ast}_{6000}| = 7.30e-04$).

<img src="figures/value-function.png" alt="Value functions by income state at r*" width="80%">

The savings policy is the asset drift $\dot a = s_i(a)$ at the equilibrium price. The low-income household decumulates ($s_L < 0$) almost everywhere above the borrowing limit and is pushed onto the constraint by the state-constraint boundary condition; this is the visible kink of $s_L$ at $\underline a$. The high-income household saves ($s_H > 0$) at small $a$ to rebuild buffer wealth and crosses zero at the income-state-specific asset target where $z_H + r^{\ast} a = c_H(a)$. Income switching keeps the cross section moving across the two drift fields — a household never stays on a single curve. The reference overlay agrees to $1.57e-03$ in sup norm on $[\underline a, 1]$.

<img src="figures/savings-policy.png" alt="Savings drift by income state at r*" width="80%">

The KFE turns the drift fields above into a cross-sectional density. The low-income $g_L$ piles up at the borrowing limit because $s_L < 0$ pushes households toward $\underline a$ and the state constraint stops them there; the kink in the low-income drift translates into a sharp spike that becomes more concentrated as the grid is refined. The high-income $g_H$ is flatter and supported on a wider range because $s_H > 0$ near $\underline a$ moves households to the right. Together the two densities place $5.8\%$ of the population within $0.02$ of the borrowing limit, the visible signature of incomplete insurance. The reference density (dashed) shows a slightly taller and narrower spike at $\underline a$ — finer discretisation resolves the constraint mass more sharply — but the away-from-constraint shape is unchanged.

<img src="figures/wealth-distribution.png" alt="Stationary asset densities by income state at r*" width="80%">

The supply curve $S(r)$ is the equilibrium-pricing argument made visible. At any $r$, every household solves the HJB with that bond return as a primitive, and $S(r)$ aggregates their stationary asset positions. The curve is monotone in $r$ because higher returns simultaneously raise desired saving and discourage borrowing. The dashed horizontal line at $r = \rho$ is the complete-markets benchmark — the rate at which a representative household with no insurance demand would price the bond. The Huggett equilibrium sits strictly below it: even at $r$ as low as $r^{\ast} = 0.0350$ (red dot), aggregate asset demand only just clears zero, and the precautionary wedge is $\rho - r^{\ast} = 0.0150$. The reference equilibrium $r^{\ast}_{\rm ref} = 0.0357$ (black cross) lies on top of the working solution at this resolution.

<img src="figures/bond-market.png" alt="Aggregate asset demand against the interest rate" width="80%">

The table separates economic outputs from numerical diagnostics. The top block is the equilibrium price and its precautionary wedge; the middle block reports cross-sectional moments; the bottom block bounds the discretisation by comparing the working grid against the finer reference grid. Mean assets are numerically zero by construction — the bisection chose $r^{\ast}$ to enforce $S(r^{\ast}) = 0$ to the tolerance shown in the residual row.

**Equilibrium and Discretisation Summary**

| Statistic                             | Value    |
|:--------------------------------------|:---------|
| Discount rate rho                     | 0.0500   |
| Equilibrium r* (working grid)         | 0.03499  |
| Equilibrium r* (reference grid)       | 0.03572  |
| Precautionary wedge rho - r*          | 0.01501  |
| Mean wealth E[a]                      | 0.00000  |
| Mean income E[z]                      | 0.1500   |
| Mean consumption E[c]                 | 0.1500   |
| Mass within 0.02 of borrowing limit   | 0.0581   |
| Prob(z = z_low)                       | 0.5000   |
| Prob(z = z_high)                      | 0.5000   |
| Bond-market residual abs(S(r*))       | 5.43e-06 |
| r* gap, working vs reference          | 7.30e-04 |
| Sup-norm savings gap, a in [a_min, 1] | 1.57e-03 |
| Sup-norm value gap, a in [a_min, 1]   | 2.23e-01 |
| Relative value gap (% of value scale) | 0.155%   |
| HJB iterations (working)              | 9        |
| HJB sup-norm change (working)         | 7.91e-07 |

## Takeaway

The Huggett pricing mechanism is the lesson. With incomplete insurance and a hard
borrowing limit, every household's problem assigns strictly positive marginal value to
buffer wealth, so aggregate desired bond holdings at the deterministic-benchmark rate
$r = \rho$ are positive. Markets clear only at a strictly lower bond return, and the
wedge $\rho - r^{\ast}$ — $\,0.0150$ on the working grid, about $30\%$ of $\rho$ — is the price the economy charges for the missing state-contingent insurance
contract. That number tightens by about $7.3e-04$ when the grid is
refined from $I = 2000$ to $I = 6000$, but the qualitative wedge does not.

The continuous-time HJB/KFE machinery is what keeps the household decision and the
induced cross section in the same equilibrium loop. The state-constraint boundary
condition $V_i'(\underline a) \geq u'(z_i + r\underline a)$ is the natural analogue of the
discrete-time Kuhn-Tucker margin in [EGP](../endogenous-grid-points/) and
[EEI](../envelope-equation-iteration/), and the upwind finite-difference scheme is what
makes that boundary condition hold without spurious mass leakage across $\underline a$.
The discrete-time mirror is the [Aiyagari tutorial](../../dynamic-programming/aiyagari/),
where the asset in fixed supply is physical capital and the wedge becomes
$1/\beta - 1 - r^{\ast}$. The Euler-based household solvers from EGP and EEI would slot
directly into the inner step of either equilibrium computation, with the bisection on $r$
unchanged.

## References

- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control* 17(5-6), 953-969.
- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109(3), 659-684.
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86.
- Moll, B. "Lecture notes on continuous-time heterogeneous-agent models." https://benjaminmoll.com/lectures/
