# Heaton-Lucas Risk Sharing and Asset Prices

> Constrained trade in equity and a bond makes the wealth distribution a pricing state.

## Overview

Heaton and Lucas (1996) ask how much asset prices can move when households cannot fully share risk. Two CRRA agents receive stochastic endowments, trade a claim to aggregate dividends, and trade a one-period bond. Equity short sales are ruled out and bond positions have a lower bound, so the identity of the household holding financial wealth matters for the stochastic discount factor.

The endogenous state is agent 1's wealth share, $\omega_1$. In a complete-markets Lucas tree, a fixed Pareto weight would summarize risk sharing. Here, constraints make the wealth share move over time, and the price of aggregate risk changes with it. This makes the tutorial a distributional counterpart to the [Lucas-tree pricing](../../dynamic-programming/asset-pricing/) example and a global-solution companion to the [Huggett bond-market](../../heterogeneous-agents/huggett-incomplete-markets/) tutorial.

## Equations

Let $z_t\in\{1,\ldots,8\}$ be the Markov shock. In state $z$, aggregate growth
is $g_z$, the equity dividend is $d_z$, and agent 1 receives endowment share
$\eta_{1z}$ with $\eta_{2z}=1-\eta_{1z}$. Agent $i$ chooses consumption $c_i$,
next-period equity holdings $s_i'$, and next-period bond holdings $b_i'$. The
equity and bond prices are $p_s$ and $p_b$.

For $i=1,2$, the budget constraint is

$$c_i+p_s s_i' + p_b b_i'
=\omega_i(p_s+d_z)+\eta_{iz},\qquad \omega_2=1-\omega_1.$$

Asset markets clear through

$$s_1'+s_2'=1,\qquad b_1'+b_2'=0,$$

with constraints

$$s_i'\geq 0,\qquad b_i'\geq \bar K^b.$$

The Kuhn-Tucker conditions for equity and bond positions are

$$1=\beta E_z\left[
g_{z'}^{1-\gamma}\left(\frac{c_i'}{c_i}\right)^{-\gamma}
\frac{p_s(z',\omega_1')+d_{z'}}{p_s}\right]+\mu_i^s,$$

$$1=\beta E_z\left[
g_{z'}^{-\gamma}\left(\frac{c_i'}{c_i}\right)^{-\gamma}
\frac{1}{p_b}\right]+\mu_i^b,$$

$$\mu_i^s\geq0,\quad \mu_i^s s_i'=0,\qquad
\mu_i^b\geq0,\quad \mu_i^b(b_i'-\bar K^b)=0.$$

The future wealth share is not an exogenous transition. It must be consistent
with the portfolio chosen today and the asset prices tomorrow:

$$\omega_1'(z')=
\frac{s_1'[p_s(z',\omega_1'(z'))+d_{z'}]+b_1'/g_{z'}}
{p_s(z',\omega_1'(z'))+d_{z'}}.$$

## Model Setup

| Object | Value | Role in the tutorial |
|---|---:|---|
| $\beta$ | 0.95 | Discount factor |
| $\gamma$ | 1.5 | CRRA risk aversion |
| $\bar{K}^b$ | -0.05 | Lower bound on each agent's bond position |
| Shock states | 8 | Joint Markov chain for growth, dividends, and endowment shares |
| Wealth-share grid | 201 points on $[-0.05,1.05]$ | Collocation grid for $\omega_1$, with a small buffer around $[0,1]$ |
| Unknowns per collocation point | 19 | Consumption, portfolios, prices, multipliers, and eight next-state wealth shares |
| Simulation | 24 paths x 10,000 periods | Used to approximate the ergodic wealth-share distribution |

## Solution Method

The hard object is the law of motion for $\omega_1$. It is implicit because tomorrow's wealth share depends on tomorrow's price, and tomorrow's price is itself a function of tomorrow's wealth share. Simultaneous Transition and Policy Function Iteration (STPFI) handles that by solving today's policies and all shock-contingent next wealth shares in the same nonlinear system.

```text
Algorithm: STPFI for the Heaton-Lucas wealth-share economy
Input: grid Omega, shock transition P, primitives beta, gamma, Kb
Output: policies c_i(z,omega), s_i'(z,omega), b_i'(z,omega), prices p_s, p_b, transition omega'(z')
Initialize c_1^0=z-dependent endowment resources, c_2^0 similarly, and p_s^0=1
repeat:
    for each current shock z and wealth grid point omega:
        take current guesses for future c_i^n(z',omega') and p_s^n(z',omega')
        solve for y=(c_1,c_2,s_1',b_1',b_2',mu^s,mu^b,p_s,p_b,{omega'(z')})
        impose Euler equations, complementary slackness, market clearing, budgets, and consistency
    damp the policy and transition updates
until the sup-norm policy change is below epsilon or the iteration cap is reached
Simulate the Markov chain and the implied omega transition to read the ergodic distribution
```

This run stopped at the iteration cap after **80** STPFI iterations with final policy change 2.06e-02 and maximum pointwise equation residual 1.27e-03. The nonlinear systems use `scipy.optimize.root`; JAX supplies the 19-by-19 Jacobian at each collocation point.

## Results

The computed equity premium is state dependent: across the displayed wealth-share grid it ranges from 0.43% to 1.42%. Those movements are not just different shock labels. They reflect which agent is close to a portfolio constraint and therefore whose marginal utility receives more weight in pricing aggregate dividends.

The first panel reads asset prices against the distributional state. The second panel shows where the simulated economy spends its time: the mean wealth share is 0.487, with the 10th and 90th percentiles at -0.050 and 1.050. The important point is that pricing is still evaluated over the whole state space, not only near the modal wealth shares.

<img src="figures/equity-premium-and-distribution.png" alt="Equity premium and ergodic distribution of wealth share." width="80%">

The multiplier panels are a map of the constraint regions. For agent 1, the no-short-sale multiplier is positive at 0.3% of interior collocation points and the borrowing multiplier at 2.7%. The equity-premium panel repeats the pricing object on the same states, making the link between constraints and risk compensation visible.

<img src="figures/policy-functions.png" alt="Multipliers and equity premium where constraints bind." width="80%">

The table should be read as a numerical diagnostic, not as a new economic moment. This Python/JAX translation keeps the model transparent and close to the original GDSGE file; the original C++ benchmark gives the tighter reference scale for the equity Euler equation.

**Euler Residuals and Benchmark Scale**

| Metric                    |   Equity EE | Bond EE      | Interpretation                                    |
|:--------------------------|------------:|:-------------|:--------------------------------------------------|
| Mean simulated residual   |    0.00303  | 9.16e-04     | Average Euler-equation miss on simulated states   |
| Median simulated residual |    0.00269  | 5.80e-04     | Typical miss away from the worst simulated states |
| Max simulated residual    |    0.24     | 1.49e-01     | Worst simulated miss in this coarse Python run    |
| GDSGE benchmark mean      |    2.08e-05 | not reported | Original GDSGE C++ scale reported for the model   |
| GDSGE benchmark max       |    0.0034   | not reported | Original GDSGE C++ scale reported for the model   |

The numerical status matters for interpretation. The plotted policies recover the state-dependent pricing and constraint patterns, but this coarse pedagogical run is looser than the optimized GDSGE benchmark. For production accuracy, the natural next step would be to raise the iteration cap, tighten damping, or run a denser grid in the original compiled implementation rather than treating the displayed Euler errors as final benchmark accuracy.

## Takeaway

The economic lesson is that incomplete markets turn the wealth distribution into an asset-pricing state. With moderate risk aversion, risk premia move because constrained households cannot freely trade away bad marginal-utility states. STPFI is useful here because it keeps the implicit wealth-share transition and the occasionally binding portfolio constraints inside the same global fixed point.

## References

- Heaton, J. & Lucas, D. (1996). *JPE* 104(3), 443-487.
- Cao, D., Luo, W. & Nie, G. (2023). *RED* 51, 199-225.
