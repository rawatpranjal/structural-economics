# Envelope-Equation Iteration for Buffer-Stock Saving

> A partial-equilibrium income-risk household problem solved from marginal continuation values.

## Overview

The household problem is the IID version of the buffer-stock saving environment used in the [endogenous-grid](../endogenous-grid-points/) tutorial and the broader [income-risk savings](../../dynamic-programming/consumption-savings/) benchmark. An impatient household cannot borrow below $\underline a=0$, so assets are valuable because they insure consumption against bad income draws.

This tutorial changes the computational object. EEI does not update the value level state by state, and it does not build the endogenous current asset grid used by EGP. It updates $W_a(a)$, the marginal value of entering next period with one more unit of assets before the next IID income draw is realized. Given that marginal continuation value, the Euler equation chooses current consumption; given the consumption rule, the envelope condition updates $W_a(a)$.

The payoff from this example is conceptual as much as computational. Grid VFI, EGP, and EEI are three routes to the same buffer-stock saving policy. The run solves all three on the coarse grid and adds a fine-grid EGP reference so the plotted EEI policy can be checked against a more accurate Euler-equation solution.

## Equations

At the start of a period the household has assets $a \in A$ and observes income
$y_j$ drawn from an IID discrete distribution with probabilities $\pi_j$. Let
$V(a,y_j)$ be the value after the current income draw and let

$$
W(a)=\sum_{j=1}^{n_y}\pi_j V(a,y_j)
$$

be the income-integrated value used for continuation payoffs. The household
chooses next-period assets $a'=g(a,y_j)$:

$$
V(a,y_j)=
\max_{a' \geq \underline a}
\{u(Ra+y_j-a')+\beta W(a')\}.
$$

The budget identity is

$$
c(a,y_j)=Ra+y_j-g(a,y_j),
\qquad R=1+r.
$$

Preferences are CRRA,

$$
u(c)=\frac{c^{1-\gamma}-1}{1-\gamma},
\qquad
u'(c)=c^{-\gamma}.
$$

For an interior saving choice, the Euler equation is

$$
u'(c(a,y_j))=\beta W_a(g(a,y_j)).
$$

The envelope condition links the marginal continuation value to the policy:

$$
W_a(a)
=\sum_{j=1}^{n_y}\pi_j V_a(a,y_j)
=R\sum_{j=1}^{n_y}\pi_j u'(c(a,y_j)).
$$

At the borrowing limit, the Euler equality becomes the inequality
$u'(Ra+y_j-\underline a)\geq \beta W_a(\underline a)$. This is the same
liquidity constraint that gives low-asset households high marginal propensities
to consume.

## Model Setup

| Parameter | Value | Role |
|---|---:|---|
| $\gamma$ | 2 | CRRA risk aversion |
| $\beta$ | 0.95 | Discount factor |
| $r$ | 0.03 | Net risk-free return |
| $\beta R$ | 0.9785 | Patience-return product |
| $\mu_y$ | 1.0 | Mean labor income |
| $\sigma_y$ | 0.2 | Income standard deviation |
| $n_y$ | 5 | IID income states |
| $\underline{a}$ | 0.0 | Borrowing limit |
| $\bar a$ | 50 | Upper asset-grid bound |
| EEI asset grid | 50 points | Power spacing near $\underline a$ |
| Reference asset grid | 600 points | Fine-grid EGP policy check |
| Simulation | 50,000 households, 500 periods | Terminal cross section |

## Solution Method

EEI keeps the Euler equation visible. The current guess is a consumption policy. The envelope equation turns that policy into a one-dimensional marginal continuation value $W_a(a)$, and the next Euler step turns it back into a consumption policy.

```text
Input: asset grid A, income states y_j, probabilities pi_j,
       primitives beta, R, gamma, borrowing limit a_min
Initialize c_0(a_i,y_j), for example from current income plus interest
For n = 0, 1, 2, ...:
    Compute W_{a,n}(a_i) = R sum_j pi_j u'(c_n(a_i,y_j))
    For each current asset a_i and income y_j:
        Set cash = R a_i + y_j
        If u'(cash-a_min) >= beta W_{a,n}(a_min):
            Set g_{n+1}(a_i,y_j) = a_min
            Set c_{n+1}(a_i,y_j) = cash - a_min
        Otherwise find c in (0, cash-a_min) such that
            u'(c) = beta W_{a,n}(cash-c)
            and set g_{n+1}(a_i,y_j) = cash - c
    Stop when max_{i,j} |c_{n+1}(a_i,y_j)-c_n(a_i,y_j)| < epsilon
Output: consumption policy c, savings policy g, marginal value W_a
```

The coarse-grid EEI solve converged in **149 iterations** with a consumption sup-norm error below $10^{-6}$. The same grid took 151 EGP iterations and 203 grid-VFI iterations. Those iteration counts compare fixed-point objects, not optimized library implementations: this EEI code uses bisection at every state to make the Euler step transparent, while EGP avoids those one-dimensional solves.

A 600-point EGP solve provides the fine-grid reference. On the plotted asset range $a\leq 20$, the coarse EEI policy is within 1.09e-02 in consumption and 1.09e-02 in next assets. Those gaps are grid and interpolation errors, not a different economic mechanism.

## Results

The consumption policy has the usual buffer-stock shape. Low-asset households consume a large share of cash on hand, but they still save after middle and high income draws because tomorrow's draw may be bad. The dashed fine-grid reference nearly overlays the coarse EEI solution; the maximum consumption gap over $a\leq 20$ is 1.09e-02.

<img src="figures/consumption-policy.png" alt="EEI consumption policy with fine-grid EGP reference" width="80%">

This is the object EEI updates. $W_a(a)$ is steep near the borrowing limit because an extra dollar is most valuable when the household has little self-insurance. The lighter curves show the state-specific marginal utilities that the envelope condition averages across income states.

<img src="figures/value-derivative.png" alt="Marginal continuation value from EEI and fine-grid reference" width="80%">

The terminal cross section is right-skewed, but modest. This is still the IID income benchmark: households build buffer stocks, many remain close to the borrowing constraint, and the right tail mostly reflects long runs of favorable income draws rather than persistent earnings types.

<img src="figures/wealth-distribution.png" alt="Simulated terminal wealth distribution under the EEI policy" width="80%">

The convergence plot should be read by fixed-point object. VFI contracts in value levels. EGP and EEI work with Euler-equation objects, so their errors shrink in consumption-policy space. This transparent EEI implementation uses bisection at every state; the figure's main point is the different updating equation, not a timing race.

<img src="figures/convergence-comparison.png" alt="Convergence paths for EEI, EGP, and grid VFI" width="80%">

The table separates economic output from numerical diagnostics. The asset distribution and MPCs come from simulating the EEI policy. The fine-grid rows compare the coarse EEI policy with a denser Euler-equation reference.

**Solution and Simulation Summary**

| Statistic                                 | Value    |
|:------------------------------------------|:---------|
| EEI iterations                            | 149      |
| Same-grid EGP iterations                  | 151      |
| Same-grid VFI iterations                  | 203      |
| Fine-grid reference points                | 600      |
| Fine-grid reference iterations            | 140      |
| Max consumption gap vs reference, a <= 20 | 1.09e-02 |
| Max next-asset gap vs reference, a <= 20  | 1.09e-02 |
| Mean assets / mean income                 | 0.4118   |
| Fraction at borrowing limit               | 3.1%     |
| Mean MPC, 0.10 transfer                   | 0.2184   |
| Perfect-foresight MPC limit               | 0.0413   |
| 10th percentile wealth                    | 0.074    |
| 50th percentile wealth                    | 0.376    |
| 90th percentile wealth                    | 0.793    |

## Takeaway

EEI is not a new savings model. It is a different fixed point for the same incomplete-markets household problem. The economic policy is still the buffer-stock rule: low income draws push households toward the asset floor, good draws rebuild wealth, and low-wealth households have high MPCs.

The computational lesson is that the envelope condition can be used as an updating equation, not only as a theorem behind the Euler equation. By carrying $W_a(a)$ forward, EEI avoids value levels and keeps the marginal value of self-insurance in view. In this transparent implementation, EGP is faster, but EEI gives a clean way to see why the policy is governed by marginal continuation values rather than by the value function level itself.

## References

- Maliar, L. and Maliar, S. (2013). Envelope Condition Method with an Application to Default Risk Models. *Journal of Economic Dynamics and Control*, 37(7), 1439-1459.
- Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.
- Kaplan, G. (2017). Lecture Notes on Heterogeneous Agent Models. University of Chicago.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.
