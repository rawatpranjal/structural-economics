# Huggett Equilibrium and the Risk-Free Rate

## Overview

Households receive stochastic income and trade one risk-free bond. They can borrow only
down to $a \geq \underline a$. Since the bond is in zero net supply, aggregate asset
demand must equal zero.

The object is the equilibrium return $r^{\ast}$. It is the rate that makes stationary
bond demand $S(r^{\ast})$ equal zero. With incomplete insurance, households want buffer
wealth at $r = \rho$. Market clearing therefore requires $r^{\ast} < \rho$.

The computation links household policy to the cross section. The HJB gives consumption
and savings drift at a candidate $r$. The KFE turns that drift into a stationary density.
Bisection updates $r$ until aggregate bond demand clears.

## Equations

A household in income state $i \in \{L, H\}$ receives endowment $z_i$. Income jumps to
the other state $j$ with Poisson intensity $\lambda_i$. Assets move between jumps by

$$\dot a = s_i(a) = z_i + r\,a - c_i(a), \qquad a \geq \underline a.$$

The value function solves the HJB equation. With CRRA utility and discount rate $\rho$,
the equation is

$$\rho\,V_i(a) = \max_{c > 0}\,
[\,u(c) + V_i'(a)\,(z_i + r\,a - c) + \lambda_i\,(V_j(a) - V_i(a))\,].$$

The first-order condition links marginal value to consumption. It also defines the
savings drift used by the density equation:

$$c_i(a) = [V_i'(a)]^{-1/\sigma}, \qquad
s_i(a) = z_i + r\,a - c_i(a).$$

The borrowing limit is a state constraint. At $a = \underline a$, the drift cannot point
outside the grid:

$$s_i(\underline a) \geq 0
\quad\Longleftrightarrow\quad
V_i'(\underline a) \geq u'(z_i + r\,\underline a),$$

with equality only when the constraint is slack.

The stationary density $g_i(a)$ satisfies the KFE. It moves mass along the asset drift
and across income states:

$$0 = -\frac{\partial}{\partial a}[s_i(a)\,g_i(a)]
- \lambda_i\,g_i(a) + \lambda_j\,g_j(a),
\qquad \int g_L + g_H = 1,$$

The bond market clears when aggregate assets are zero:

$$S(r) \equiv \int_{\underline a}^{\bar a} a\,[g_L(a) + g_H(a)]\,da = 0.$$

The equilibrium return is the root of this function. In this run,
$r^{\ast} = 0.03499$ and the residual is $5.43e-06$.

## Model Setup

The calibration keeps only the ingredients needed for Huggett pricing. There are two
income states, symmetric switching, one bond, and a borrowing limit.

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

The symmetric income chain implies $p_L = p_H = 0.5$. Expected income is
$\bar z = 0.1500$. The KFE solution recovers
$|p_L - 0.5| = 1.11e-16$.

## Solution Method

At a candidate $r$, the code solves the household HJB on the asset grid. It uses an
upwind finite-difference scheme because the asset drift can point left or right.

**Upwind derivative.** The algorithm computes forward and backward derivatives of
$V_i(a_k)$. Each derivative implies a consumption choice and a drift. The update keeps
the derivative whose drift points into the grid.

**Implicit step.** The HJB update stacks both income states into one vector:

$$[(\Delta^{-1} + \rho)\,\mathbf I - A^{n}]\,V^{n+1} = u(c^{n}) + \Delta^{-1} V^{n},$$

where $A^n$ is the upwind generator. It combines asset drift and income switching.

**KFE.** Once $V$ converges, the same generator gives the stationary distribution. The
code solves $A^{\top} g = 0$ and rescales $g$ to integrate to one.

**Equilibrium.** The outer loop computes $S(r)$ and updates $r$ by bisection. Higher
returns raise saving and reduce borrowing, so the zero of $S(r)$ is well defined here.

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

**Working solve.** The HJB inner loop converged in **9 iterations**. The final sup-norm change was $7.91e-07$. Bisection found $r^{\ast} = 0.03499$ on the 2000-point grid. The bond-market residual is $5.43e-06$.

**Reference solve.** The reference grid repeats the solve with $I_{\rm ref} = 6000$ points. It gives $r^{\ast}_{\rm ref} = 0.03572$. The interest-rate gap is $7.30e-04$. On $a \in [\underline a, 1]$, the savings-policy gap is $1.57e-03$ in sup norm.

## Results

The value functions are increasing and concave in assets. $V_H(a)$ lies above $V_L(a)$ because high income raises cash on hand. Both curves steepen near the borrowing limit. The relative value gap against the reference grid is $0.16\%$.

<img src="figures/value-function.png" alt="Value functions by income state at r*" width="80%">

The savings policy is the asset drift at $r^{\ast}$. Low-income households decumulate above the borrowing limit. High-income households save near the limit to rebuild buffer wealth. Income switching moves households between the two drift fields. The reference gap is $1.57e-03$ in sup norm on $[\underline a, 1]$.

<img src="figures/savings-policy.png" alt="Savings drift by income state at r*" width="80%">

The KFE turns drift into a cross-sectional density. Low-income mass piles near the borrowing limit because negative drift pushes left. High-income density is flatter because positive drift moves households right. The population share within $0.02$ of the limit is $5.8\%$.

<img src="figures/wealth-distribution.png" alt="Stationary asset densities by income state at r*" width="80%">

The demand curve plots aggregate asset demand against $r$. Higher returns raise saving and reduce borrowing, so $S(r)$ rises with $r$. The complete-markets benchmark is $r = \rho$. The Huggett equilibrium is lower, at $r^{\ast} = 0.0350$. The precautionary wedge is $\rho - r^{\ast} = 0.0150$.

<img src="figures/bond-market.png" alt="Aggregate asset demand against the interest rate" width="80%">

The table reports prices, cross-sectional moments, and discretisation diagnostics. Mean assets are zero because bisection chose $r^{\ast}$ to satisfy $S(r^{\ast}) = 0$.

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

The Huggett price is a market-clearing return. Income risk and the borrowing limit make
households want buffer wealth at $r = \rho$. The bond market clears only at a lower
return. In this run the wedge is $\rho - r^{\ast} = 0.0150$.

The HJB/KFE loop ties the household policy to the stationary cross section. The upwind
HJB respects the borrowing limit. The KFE then measures aggregate asset demand. Bisection
on $r$ closes the zero-net-supply bond market.

## References

- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control* 17(5-6), 953-969.
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86.
- Moll, B. "Lecture notes on continuous-time heterogeneous-agent models." https://benjaminmoll.com/lectures/
