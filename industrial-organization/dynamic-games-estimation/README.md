# Dynamic Games Estimation from First-Stage CCPs

> Estimate a small two-firm investment game without solving a full MPE at every trial parameter.

## Overview

Dynamic games are hard to estimate because each parameter guess can require a new Markov-perfect equilibrium. The method-first alternative is to estimate first-stage conditional choice probabilities (CCPs), hold those future strategies fixed, and evaluate continuation values under that policy.

The economic model is intentionally small: two firms invest on a four-rung quality ladder. Investment is costly, raises the chance of moving up, and is more attractive when the rival is ahead. The tutorial solves the game once to generate known-truth data, then estimates the payoff parameters using a Hotz-Miller/NPL-style forward value step. A Monte Carlo forward-simulation check is included to connect the exact finite-state value evaluation to the BBL intuition.

## Equations

The firm-view state is $\omega=(q_i,q_j)$ with $q_i,q_j\in\{0,1,2,3\}$.
Firm $i$ chooses $a_i\in\{0,1\}$, where one means invest. Flow payoff is

$$
\pi_i(\omega,a_i;\theta)
= \theta_q q_i - \theta_c a_i + \theta_g \max\{q_j-q_i,0\} a_i .
$$

First-stage CCPs estimate $p(\omega)=\Pr(a_i=1\mid \omega)$. Holding these CCPs
fixed, define the policy transition $\hat P$ and expected flow payoff
$\bar\pi_\theta(\omega;\hat p)$. The ex ante value under the first-stage policy is

$$
W_\theta = \bar\pi_\theta(\hat p) + \beta \hat P W_\theta.
$$

Choice-specific values for the current action use the rival's first-stage CCP
and the continuation value $W_\theta$:

$$
v_\theta(a_i,\omega)
= \pi_i(\omega,a_i;\theta) + \beta E_{\hat p_j}\left[W_\theta(\omega')\mid \omega,a_i\right].
$$

The second-stage pseudo likelihood is then

$$
\ell(\theta)=\sum_{i,t}
d_{it}\log \Lambda[v_\theta(1,\omega_{it})-v_\theta(0,\omega_{it})]
+(1-d_{it})\log\{1-\Lambda[v_\theta(1,\omega_{it})-v_\theta(0,\omega_{it})]\}.
$$

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Firms | 2 | Symmetric competitors observed in firm-view states |
| Quality rungs | 0 to 3 | Finite dynamic state space |
| Discount factor | 0.90 | Weight on future market position |
| True $\theta_q$ | 0.70 | Value of own quality |
| True $\theta_g$ | 0.40 | Extra incentive to invest when behind |
| True $\theta_c$ | 1.00 | Investment cost |
| Markets | 1,000 | Independent simulated two-firm markets |
| Periods | 30 | Panel length used for first-stage CCPs |

## Solution Method

The tutorial separates solving from estimating. The full soft MPE is solved once only to create data with known truth. The estimator does not solve an equilibrium inside the likelihood.

```text
Algorithm: CCP forward-value estimator for a dynamic game
Input: panel of states and actions, transition law, discount factor beta
First stage:
  Estimate p_hat(omega)=Pr(a_i=1 | omega) by state frequencies with smoothing
Second stage for each candidate theta:
  Build the transition matrix induced by p_hat for both firms
  Solve W_theta = flow_theta(p_hat) + beta P_hat W_theta
  Compute current choice-specific values using W_theta and rival p_hat
  Evaluate the logit pseudo likelihood of observed investment actions
Forward-simulation check:
  Simulate future paths under p_hat and compare simulated values with exact W_theta
Output: payoff estimates, policy fit, first-stage diagnostics, value-simulation error
```

This is the computational architecture behind NPL, EPL, and BBL-style estimators: future strategic behavior is summarized by first-stage policies, so each parameter guess requires policy evaluation rather than a full equilibrium solve.

## Results

The first-stage CCPs already contain most of the strategic pattern: firms invest more when they are behind and less when they are at the top rung. The second-stage model-implied CCPs smooth those empirical frequencies through the structural payoff parameters and the forward-value equation.

Rows are own quality and columns are rival quality in the firm-view state.

<img src="figures/ccp-heatmaps.png" alt="True, first-stage, and model-implied investment CCPs" width="80%">

Policy fit is the main diagnostic because the second-stage likelihood is built from choice probabilities. The model-implied policy RMSE against known truth is **0.001**, while the first-stage empirical CCP RMSE is **0.017**.

Known truth makes it possible to separate first-stage sampling error from second-stage structural fit.

<img src="figures/policy-fit.png" alt="State-level policy fit against true CCPs" width="80%">

The exact finite-state policy evaluation and Monte Carlo forward simulation target the same continuation values. The forward-simulation RMSE is **0.045** with 1,000 paths per state and a 70-period horizon.

The simulation check connects the matrix policy evaluation to the forward-simulation logic used in larger BBL-style applications.

<img src="figures/forward-values.png" alt="Exact versus simulated forward values" width="80%">

The estimator recovers the payoff parameters without re-solving the MPE during optimization.

**Known-truth parameter recovery**

| Parameter               |   True |   Estimate |    Error |
|:------------------------|-------:|-----------:|---------:|
| Quality value theta_q   |    0.7 |    0.70387 |  0.00387 |
| Gap incentive theta_g   |    0.4 |    0.40502 |  0.00502 |
| Investment cost theta_c |    1   |    0.99885 | -0.00115 |

Observation counts reveal where first-stage CCPs are precise and where smoothing matters more.

**State-level CCP diagnostics**

|   Own quality |   Rival quality |   Observations |   True CCP |   First-stage CCP |   Model-implied CCP |   First-stage error |   Model error |
|--------------:|----------------:|---------------:|-----------:|------------------:|--------------------:|--------------------:|--------------:|
|             0 |               0 |            218 |    0.67348 |           0.64545 |             0.67474 |            -0.02802 |       0.00127 |
|             0 |               1 |            264 |    0.70562 |           0.70301 |             0.70399 |            -0.00261 |      -0.00162 |
|             0 |               2 |            343 |    0.76602 |           0.77391 |             0.76751 |             0.00789 |       0.00149 |
|             0 |               3 |            434 |    0.82567 |           0.83486 |             0.82768 |             0.00919 |       0.002   |
|             1 |               0 |            264 |    0.72745 |           0.76316 |             0.72972 |             0.03571 |       0.00227 |
|             1 |               1 |            548 |    0.6808  |           0.65091 |             0.68308 |            -0.02989 |       0.00228 |
|             1 |               2 |            798 |    0.72081 |           0.74    |             0.72091 |             0.01919 |       0.0001  |
|             1 |               3 |           1511 |    0.78171 |           0.78652 |             0.78329 |             0.00481 |       0.00158 |
|             2 |               0 |            343 |    0.61349 |           0.62609 |             0.61509 |             0.0126  |       0.0016  |
|             2 |               1 |            798 |    0.5961  |           0.6225  |             0.5969  |             0.0264  |       0.0008  |
|             2 |               2 |           2652 |    0.56357 |           0.56858 |             0.56395 |             0.00501 |       0.00039 |
|             2 |               3 |           8326 |    0.62974 |           0.62428 |             0.63123 |            -0.00546 |       0.0015  |
|             3 |               0 |            434 |    0.31284 |           0.32798 |             0.31323 |             0.01514 |       0.00039 |
|             3 |               1 |           1511 |    0.31125 |           0.31659 |             0.3116  |             0.00534 |       0.00035 |
|             3 |               2 |           8326 |    0.30704 |           0.3092  |             0.30737 |             0.00216 |       0.00033 |
|             3 |               3 |          33230 |    0.30303 |           0.30257 |             0.30329 |            -0.00046 |       0.00027 |

The equilibrium solve is reported as data-generation evidence; the estimator itself uses first-stage CCPs and forward values.

**Solver and estimator diagnostics**

| Diagnostic                    |           Value |
|:------------------------------|----------------:|
| Truth MPE converged           |      1          |
| Truth MPE iterations          |    591          |
| Second-stage success          |      1          |
| Second-stage iterations       |     72          |
| Second-stage log likelihood   | -37238.2        |
| First-stage policy RMSE       |      0.0170873  |
| Model policy RMSE             |      0.00135434 |
| Forward simulation value RMSE |      0.0450165  |
| Firm-period observations      |  60000          |

## Takeaway

Dynamic-game estimation can be organized around policies rather than repeated equilibrium solution. First-stage CCPs summarize how rivals are expected to behave. Given those policies, each candidate payoff vector only needs a forward-value evaluation and a pseudo likelihood. That is the computational gain: the hard fixed point is moved out of the inner loop, and diagnostics shift toward first-stage CCP quality, policy fit, and forward-simulation accuracy.

## References

- [Aguirregabiria, V. and Mira, P. (2007). Sequential Estimation of Dynamic Discrete Games. *Econometrica*, 75(1), 1-53.](https://doi.org/10.1111/j.1468-0262.2007.00731.x)
- [Bajari, P., Benkard, C. L., and Levin, J. (2007). Estimating Dynamic Models of Imperfect Competition. *Econometrica*, 75(5), 1331-1370.](https://doi.org/10.1111/j.1468-0262.2007.00796.x)
- [Pesendorfer, M. and Schmidt-Dengler, P. (2008). Asymptotic Least Squares Estimators for Dynamic Games. *Review of Economic Studies*, 75(3), 901-928.](https://doi.org/10.1111/j.1467-937X.2008.00497.x)
