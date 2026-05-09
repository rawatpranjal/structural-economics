# Keane-Wolpin Career Choice by Emax Approximation

> Finite-horizon schooling and occupation choice with sampled continuation-value regression.

## Overview

A young worker chooses whether to stay in school, work blue collar, work white collar, or stay home. Schooling raises later white-collar wages. Work builds occupation-specific experience. The relevant state is the stock of human capital accumulated before the current age.

The economic object is the life-cycle policy: when does the worker leave school, which occupation does she enter, and how does early experience shape later wages?

The computational object is the Emax function. Exact backward induction is straightforward in this small version, but the number of states grows quickly. The Keane-Wolpin approximation computes exact Emax values at sampled states and predicts the rest with a regression.

## Equations

At age $t$, the state is

$$
s_t = (E_t, X^b_t, X^w_t),
$$

where $E_t$ is completed schooling and $X^b_t$, $X^w_t$ are occupation-specific
experience stocks. The action set is

$$
d_t \in D(s_t,t)
\subseteq \{\mathrm{school}, \mathrm{blue}, \mathrm{white}, \mathrm{home}\}.
$$

The transition is deterministic conditional on the choice. Schooling raises
$E_t$ by one year, blue-collar work raises $X^b_t$, white-collar work raises
$X^w_t$, and home leaves measured human capital unchanged:

$$
s_{t+1}=g(s_t,d_t).
$$

The deterministic part of the choice-specific value is

$$
v_t(d,s) =
\underbrace{u_d(s,t)}_{\text{current school payoff, wage payoff, or home payoff}} +
\underbrace{\beta \mathbb{E}_{t+1}(g(s,d))}_{\text{discounted continuation value}}.
$$

The underbraces are the key economic split. Current payoffs rank school, work,
and home today. The continuation value prices the human capital state that the
choice carries into tomorrow.

With Type-I extreme value taste shocks of scale $\sigma_\epsilon$, the Emax
function is

$$
\mathbb{E}_t(s) =
\underbrace{\sigma_\epsilon
\log \sum_{d \in D(s,t)} \exp(v_t(d,s)/\sigma_\epsilon)}_{\text{expected max over feasible discrete choices}} +
\underbrace{\sigma_\epsilon \gamma}_{\text{mean of the extreme-value shock}}.
$$

The exact recursion evaluates this expression at every reachable state. The
Keane-Wolpin approximation evaluates it only on a sampled set
$S_t^{sample}=\{s_{t,1},\dots,s_{t,M_t}\}$, then fits the regression

$$
Y_{t,i} = \mathbb{E}_t(s_{t,i}), \qquad
Y_{t,i} = \phi(s_{t,i},t)' b_t + \eta_{t,i}.
$$

In this tutorial the basis vector is

$$
\phi(s,t)=
(1, E, X^b, X^w, X^b+X^w, E^2, (X^b)^2, (X^w)^2,
E X^b, E X^w, X^b X^w, t).
$$

The fitted continuation surface is

$$
\widehat{\mathbb{E}}_t(s)=\phi(s,t)'\widehat b_t.
$$

This is the computational shortcut: unsampled states inherit continuation
values from the fitted Emax surface rather than from fresh exact integrations.

## Model Setup

| Symbol | Calibration | Meaning |
|---|---:|---|
| $t$ | ages 16-29 | Finite-horizon decision age |
| $s_t=(E_t,X^b_t,X^w_t)$ | starts at $(10,0,0)$ | Schooling, blue-collar experience, white-collar experience |
| $D(s,t)$ | $\{\mathrm{school},\mathrm{blue},\mathrm{white},\mathrm{home}\}$ subject to feasibility | Discrete choice set |
| $g(s,d)$ | deterministic | Human-capital transition after choice $d$ |
| $\beta$ | 0.94 | Discount factor in the Emax recursion |
| $\sigma_\epsilon$ | 0.22 | Type-I extreme value scale for choice shocks |
| $\sigma_w$ | 0.18 | Log wage shock used in simulated wage paths |
| $\phi(s,t)$ | 12 polynomial terms | Basis for the sampled Emax regression |
| $M_t$ | up to 260 sampled states | Exact Emax evaluations used to fit $\widehat b_t$ |
| $N_s$ | 2,310 pre-terminal states | Reachable state count in the exact benchmark |
| Synthetic panel | 6,000 workers | Simulated from the approximate conditional choice probabilities |

## Solution Method

The exact benchmark is the finite-horizon recursion

$$
Q_t(d,s)=u_d(s,t)+\beta \mathbb{E}_{t+1}(g(s,d)),
\qquad
\mathbb{E}_t(s)=\sigma_\epsilon \log \sum_{d\in D(s,t)}
\exp(Q_t(d,s)/\sigma_\epsilon)+\sigma_\epsilon\gamma.
$$

The approximate solver keeps the same recursion but estimates
$\mathbb{E}_t(s)$ from sampled states. At each age, stack the sampled targets
in $Y_t$ and the basis functions in $\Phi_t$. The regression step is

$$
\widehat b_t=(\Phi_t'\Phi_t+\lambda I)^{-1}\Phi_t'Y_t,
\qquad
\widehat{\mathbb{E}}_t(s)=\phi(s,t)'\widehat b_t.
$$

The small ridge term $\lambda$ only stabilizes the least-squares fit when an
early age has few reachable states.

```text
Algorithm: sampled Emax approximation
Given reachable sets S_t and terminal values E_T(s)
for t = T-1, T-2, ..., 0:
    draw S_t^sample = {s_{t,1}, ..., s_{t,M_t}} from S_t
    for each sampled state s_{t,i}:
        Q_{t,i}(d) = u_d(s_{t,i},t) + beta * Ehat_{t+1}(g(s_{t,i},d))
        Y_{t,i} = sigma_e * log sum_{d in D(s_{t,i},t)}
                  exp(Q_{t,i}(d) / sigma_e) + sigma_e * gamma
    form Phi_t with row i equal to phi(s_{t,i},t)'
    b_hat_t = (Phi_t' Phi_t + lambda I)^{-1} Phi_t' Y_t
    for each reachable state s in S_t:
        Ehat_t(s) = phi(s,t)' b_hat_t
        Qhat_t(d,s) = u_d(s,t) + beta * Ehat_{t+1}(g(s,d))
        P_t(d | s) = exp(Qhat_t(d,s)/sigma_e)
                     / sum_j exp(Qhat_t(j,s)/sigma_e)
simulate histories from P_t(d | s)
```

The approximation is deliberately auditable here. The exact solve is still run,
so the page can report whether the fitted Emax surface changes continuation
values or policies in this calibration.

## Results

Schooling is concentrated at early ages, when its option value is high. As the horizon shortens, workers move into blue- and white-collar jobs. White-collar work rises after schooling has accumulated.

<img src="figures/choice-shares.png" alt="Simulated choice shares by age" width="80%">

The synthetic wage profiles separate the two human-capital margins. Blue-collar wages pay earlier experience. White-collar wages are more education intensive and rise after workers leave school.

<img src="figures/wage-profiles.png" alt="Mean wages in the simulated panel" width="80%">

The exact solve provides a benchmark. Approximation errors are largest at young ages because early states carry many future option values. Policy agreement is the share of states where exact and approximate deterministic argmax choices match. In this run, the largest age-specific RMSE is 12.3% of the exact Emax standard deviation, and the lowest policy agreement is 90.0%. That is acceptable for a teaching approximation, not a universal tolerance: in estimation, the relevant test is whether simulated moments and the likelihood are stable when the sample size or basis is enlarged.

<img src="figures/emax-accuracy.png" alt="Approximation errors against exact backward induction" width="80%">

The table reports exact-vs-approximate Emax errors on every reachable state, not only on the sampled states used in the regression. The normalized RMSE divides by the exact Emax standard deviation at that age, so it is a scale check rather than a new objective.

**Emax approximation diagnostics by age**

|   Age |   States |   Mean abs. Emax error |   Emax RMSE |   RMSE / exact Emax sd |   90th pct. abs. error |   Max abs. error |   Policy agreement |
|------:|---------:|-----------------------:|------------:|-----------------------:|-----------------------:|-----------------:|-------------------:|
|    16 |        1 |                 0.0175 |      0.0175 |               nan      |                 0.0175 |           0.0175 |             1      |
|    17 |        4 |                 0.0254 |      0.0277 |                 0.123  |                 0.0377 |           0.0405 |             1      |
|    18 |       10 |                 0.0225 |      0.0275 |                 0.0797 |                 0.0421 |           0.0461 |             0.9    |
|    19 |       20 |                 0.0235 |      0.0295 |                 0.0645 |                 0.0466 |           0.0652 |             0.95   |
|    20 |       35 |                 0.0318 |      0.0378 |                 0.0661 |                 0.0605 |           0.0848 |             0.9429 |
|    21 |       56 |                 0.0436 |      0.0492 |                 0.071  |                 0.0718 |           0.0985 |             0.9821 |
|    22 |       84 |                 0.0541 |      0.0606 |                 0.0732 |                 0.0876 |           0.1125 |             0.9881 |
|    23 |      120 |                 0.0631 |      0.0706 |                 0.0727 |                 0.1042 |           0.1329 |             1      |
|    24 |      165 |                 0.0699 |      0.079  |                 0.0707 |                 0.1185 |           0.1499 |             0.9939 |
|    25 |      219 |                 0.0696 |      0.0797 |                 0.0665 |                 0.1248 |           0.1654 |             0.9863 |
|    26 |      282 |                 0.0697 |      0.0806 |                 0.0662 |                 0.1285 |           0.1811 |             0.9929 |
|    27 |      354 |                 0.07   |      0.0818 |                 0.0687 |                 0.1288 |           0.1946 |             0.9859 |
|    28 |      435 |                 0.0703 |      0.0833 |                 0.0738 |                 0.127  |           0.2054 |             0.9839 |
|    29 |      525 |                 0.0715 |      0.0862 |                 0.0831 |                 0.1406 |           0.2196 |             1      |

The Emax regression uses at most the sample cap at each age. Later ages have fewer continuation states, so the approximation becomes nearly exact.

**Sampled regression fit by age**

|   Age |   States |   Sampled states |   Regression RMSE on sampled states |   Sampled target sd |
|------:|---------:|-----------------:|------------------------------------:|--------------------:|
|    16 |        1 |                1 |                              0      |              0      |
|    17 |        4 |                4 |                              0      |              0.2284 |
|    18 |       10 |               10 |                              0.0006 |              0.3425 |
|    19 |       20 |               20 |                              0.0029 |              0.4523 |
|    20 |       35 |               35 |                              0.0032 |              0.5632 |
|    21 |       56 |               56 |                              0.0033 |              0.685  |
|    22 |       84 |               84 |                              0.0037 |              0.8182 |
|    23 |      120 |              120 |                              0.0042 |              0.9605 |
|    24 |      165 |              165 |                              0.0055 |              1.1071 |
|    25 |      219 |              219 |                              0.0057 |              1.1891 |
|    26 |      282 |              260 |                              0.0059 |              1.1872 |
|    27 |      354 |              260 |                              0.0063 |              1.2173 |
|    28 |      435 |              260 |                              0.0069 |              1.1188 |
|    29 |      525 |              260 |                              0.087  |              1.0517 |

These are moments from the simulated panel, not estimates from real data. They make the dynamic policy visible in familiar labor outcomes.

**Synthetic life-cycle moments**

| Moment                                     |   Value |
|:-------------------------------------------|--------:|
| Mean final schooling                       | 13.5915 |
| Share with at least 12 years               |  0.9098 |
| Share with at least 16 years               |  0.0798 |
| Mean blue experience at last observed age  |  5.9145 |
| Mean white experience at last observed age |  3.1485 |
| Mean years spent in school during model    |  3.5915 |
| Approximation runtime seconds              |  0.0748 |
| Exact runtime seconds                      |  0.0863 |

## Takeaway

Finite-horizon structural labor models are conceptually simple but expensive because schooling and experience create many continuation states. The Keane-Wolpin approximation keeps the dynamic logic intact while replacing most exact Emax evaluations with a fitted continuation-value surface. The tradeoff is visible: the method saves computation, but approximation error is largest where early human-capital choices have the most future consequences.

## References

- [Keane, M. P. and Wolpin, K. I. (1997). The Career Decisions of Young Men. *Journal of Political Economy*, 105(3), 473-522.](https://doi.org/10.1086/262080)
- [Keane, M. P. and Wolpin, K. I. (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. *Review of Economics and Statistics*, 76(4), 648-672.](https://doi.org/10.2307/2109768)
