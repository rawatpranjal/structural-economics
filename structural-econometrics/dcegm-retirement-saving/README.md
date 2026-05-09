# Retirement and Saving by Discrete-Continuous EGM

> Life-cycle retirement choice with branch-specific endogenous grids and an upper envelope.

## Overview

An older worker chooses whether to keep working or retire. The same household also chooses how much to save. Retirement is discrete and absorbing. Saving is continuous.

The economic object is the retirement boundary: at each age, which asset levels make the household leave work? The saving policy matters because assets insure retirement consumption.

A plain grid search treats every current asset and every next asset as a nested maximization. DC-EGM avoids that inner search. It solves the continuous saving problem separately for work and retirement, then keeps the upper envelope of the choice-specific value functions.

## Equations

At age $t$, the household enters with assets $a_t$ and retirement status
$m_t \in \{0,1\}$. Status $m_t=0$ means still active, and $m_t=1$ means already
retired. An active household can choose work or retire:

$$
d_t \in D(m_t), \qquad
D(0)=\{\mathrm{work},\mathrm{retire}\}, \qquad
D(1)=\{\mathrm{retire}\}.
$$

The next retirement status is absorbing:

$$
m'(d)=
\begin{cases}
0, & d=\mathrm{work}, \\
1, & d=\mathrm{retire}.
\end{cases}
$$

The budget constraint is

$$
c_t + a_{t+1} = R a_t + y_t(d_t),
\qquad a_{t+1} \geq \underline a .
$$

For any branch $d$, define the branch value

$$
V_t^d(a) =
\max_{a' \geq \underline a}
\{
\underbrace{u(Ra+y_t(d)-a')}_{\text{utility from consumption}} +
\underbrace{\psi_t(d)}_{\text{work cost or retirement amenity}} +
\underbrace{\beta V_{t+1}^{m'(d)}(a')}_{\text{continuation value under next status}}
\}.
$$

The active and retired value functions are then

$$
V_t^1(a) =
V_t^{\mathrm{retire}}(a),
\qquad
V_t^0(a)=
\max\{V_t^{\mathrm{work}}(a), V_t^{\mathrm{retire}}(a)\}.
$$

This upper envelope is the central DC-EGM object. It preserves the discrete
retirement kink instead of forcing the value function to be globally concave.

On a fixed branch, the continuous saving problem has the Euler equation

$$
u'(c_t^d(a')) =
\beta R
\frac{\partial V_{t+1}^{m'(d)}(a')}{\partial a'}.
$$

With CRRA utility, $u'(c)=c^{-\gamma}$, so EGM fixes the next-asset grid
$a'_i$, evaluates the next marginal value $\mu_{t+1}^{m'(d)}(a'_i)$, and
inverts the Euler equation:

$$
c_{t,i}^d =
(\beta R \mu_{t+1}^{m'(d)}(a'_i))^{-1/\gamma}.
$$

The endogenous current asset attached to that next-asset point is

$$
a_{t,i}^{\mathrm{endo},d} =
\frac{c_{t,i}^d + a'_i - y_t(d)}{R}.
$$

Each branch produces its own endogenous grid and value curve:

$$
\widetilde V_t^d(a_{t,i}^{\mathrm{endo},d}) =
u(c_{t,i}^d)+\psi_t(d)+\beta V_{t+1}^{m'(d)}(a'_i).
$$

DC-EGM interpolates the branch curves back to the common current-asset grid and
then takes the upper envelope across $d$.

## Model Setup

| Symbol | Calibration | Meaning |
|---|---:|---|
| $t$ | ages 55-70 | Finite-horizon retirement window |
| $a_t$ | grid on [0.0, 22.0] | Assets at the start of age $t$ |
| $m_t$ | $0$ active, $1$ retired | Absorbing retirement status |
| $d_t$ | $\mathrm{work}$ or $\mathrm{retire}$ | Discrete labor-supply choice |
| $c_t$ | residual from budget | Consumption after choosing next assets |
| $a'_i$ | 420 points | Exogenous next-asset grid used by DC-EGM |
| $a^{\mathrm{endo},d}_{t,i}$ | branch-specific | Current asset implied by Euler inversion on branch $d$ |
| $\beta$ | 0.96 | Discount factor |
| $R=1+r$ | 1.02 | Gross asset return |
| $\gamma$ | 2.0 | CRRA curvature |
| $y_t(\mathrm{retire})$ | 0.78 | Pension income after retirement |
| $\psi_t(\mathrm{retire})$ | 0.00 | Retirement amenity relative to work cost |
| $\underline a$ | 0.0 | Borrowing limit on next assets |
| Brute-force audit grid | 150 assets | Smaller benchmark grid for exhaustive search |
| Synthetic panel | 8,000 households | Simulated with initial assets centered at 2.8 |

## Solution Method

The continuous decision is solved on a next-asset grid. The discrete choice is
handled after each branch has produced its own value function.

For each branch $d$, DC-EGM constructs points

$$
(a_{t,i}^{\mathrm{endo},d}, c_{t,i}^d, a'_i, \widetilde V_t^d(a_{t,i}^{\mathrm{endo},d})).
$$

Interpolation converts those branch-specific points into functions on the
common current-asset grid. The active policy is then

$$
d_t^{\ast}(a)=
\begin{cases}
\mathrm{work}, & V_t^{\mathrm{work}}(a) \geq V_t^{\mathrm{retire}}(a), \\
\mathrm{retire}, & V_t^{\mathrm{retire}}(a) > V_t^{\mathrm{work}}(a).
\end{cases}
$$

The selected consumption and saving policies are copied from the winning branch.

```text
Algorithm: DC-EGM for retirement and saving
Input: current asset grid A, next asset grid A', ages t=0,...,T
Terminal: V_T^0(a) = V_T^1(a) = bequest(a)
for t = T-1, T-2, ..., 0:
    # retired status: only the retire branch is feasible
    for a'_i in A':
        mu_i = d V_{t+1}^1(a'_i) / d a'
        c_i^retire = (beta R mu_i)^(-1/gamma)
        a_i^{endo,retire} = (c_i^retire + a'_i - y_t(retire)) / R
        V_i^retire = u(c_i^retire) + psi_t(retire) + beta V_{t+1}^1(a'_i)
    interpolate retire branch onto A to get V_t^1(a)

    # active status: solve work and retire branches separately
    for d in {work, retire}:
        next status is m'(d)
        repeat the Euler inversion above using V_{t+1}^{m'(d)}
        interpolate branch values V_t^d(a), c_t^d(a), and g_t^d(a)

    V_t^0(a) = max{V_t^work(a), V_t^retire(a)}
    choose c_t^0(a) and g_t^0(a) from the branch attaining the max
```

The brute-force audit solves the same model on a smaller asset grid by checking
every feasible next asset at every current asset. It is slower and coarser, but
it provides a direct benchmark for the branch policies and the retirement
boundary.

## Results

The work and retirement branches solve ordinary continuous saving problems. The selected active policy follows the branch with the larger value. The switch point is where the discrete choice creates a kink.

<img src="figures/branch-consumption.png" alt="Work and retirement consumption branches" width="80%">

The threshold falls with age as work becomes less attractive and the horizon for earning labor income shrinks. At high ages, even lower-asset households prefer the retired branch.

<img src="figures/retirement-boundary.png" alt="Asset threshold for retirement by age" width="80%">

The simulated panel translates the policy functions into life-cycle moments. Retirement rises gradually because households start with different assets and the simulation smooths the deterministic boundary with small taste shocks.

<img src="figures/simulated-life-cycles.png" alt="Simulated retirement, assets, and consumption" width="80%">

The brute-force rule uses a smaller grid and searches over all feasible next assets. The comparison checks whether the upper envelope chooses the same retirement region.

<img src="figures/bruteforce-audit.png" alt="Retirement rule from DC-EGM and brute-force VFI" width="80%">

The policy gaps are measured after interpolating the DC-EGM policy onto the smaller brute-force grid. Agreement is the share of audit-grid assets with the same retire/work decision.

**DC-EGM versus brute-force audit**

|   Age |   Largest consumption-policy gap |   Largest next-asset-policy gap |   Retirement decision agreement |
|------:|---------------------------------:|--------------------------------:|--------------------------------:|
|    58 |                           0.1694 |                          0.6817 |                          0.98   |
|    62 |                           0.1471 |                          0.4089 |                          0.9933 |
|    66 |                           0.1276 |                          0.3149 |                          0.9933 |
|    70 |                           0.1259 |                          0.1259 |                          1      |

The runtime comparison is deliberately uneven: DC-EGM uses the larger main grid, while brute force uses the smaller audit grid. The point is the order of the computational bottleneck.

**Simulation and runtime moments**

| Moment                        |    Value |
|:------------------------------|---------:|
| Mean simulated retirement age |  62.21   |
| Share retired by age 62       |   0.5416 |
| Share retired by age 67       |   1      |
| Mean assets at age 55         |   3.0556 |
| Mean assets at age 70         |   0.6717 |
| DC-EGM runtime seconds        |   0.0031 |
| Brute-force runtime seconds   |   0.0481 |
| Brute-force asset points      | 150      |
| DC-EGM asset points           | 420      |

## Takeaway

DC-EGM is useful when a structural labor model combines a discrete margin with continuous saving. Each branch remains an Euler-equation problem, so EGM avoids the inner root search or grid search. The discrete retirement option then enters through the upper envelope. That envelope is the economic policy boundary and the numerical source of the kink.

## References

- [Iskhakov, F., Jorgensen, T. H., Rust, J., and Schjerning, B. (2017). The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with or without Taste Shocks. *Quantitative Economics*, 8(2), 317-365.](https://doi.org/10.3982/QE643)
- [Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.](https://doi.org/10.1016/j.econlet.2005.09.013)
