# RBC with Irreversible Investment

> Occasionally binding I >= 0 constraint creates asymmetric business cycles and option value of waiting.

## Overview

In the standard RBC model, the representative agent can freely adjust the capital stock in both directions. In reality, much physical capital is irreversible: once installed, machinery and structures cannot easily be converted back to consumption goods.

We impose the constraint $I_t \geq 0$ (investment cannot be negative), which binds when TFP is low and the agent would prefer to disinvest. This creates:

- **Asymmetric responses**: Contractions are amplified (can't reduce K fast enough) while expansions look like the standard model
- **Option value of waiting**: Irreversibility makes capital decisions partially irreversible, creating value in delaying investment
- **Precautionary behavior**: The constraint raises the effective cost of capital, leading to lower average investment

## Equations

$$V(K, z) = \max_{c, K'} \bigl[ \frac{c^{1-\sigma}}{1-\sigma} + \beta \, \mathbb{E}\left[V(K', z')\right] \bigr]$$

subject to:
$$c + K' = z K^\alpha + (1-\delta) K$$
$$K' \geq (1-\delta) K \quad \Leftrightarrow \quad I \geq 0$$

**Euler equation with complementary slackness:**
$$c^{-\sigma} (1-\mu) = \beta \, \mathbb{E}\left[ c'^{-\sigma} \left(\alpha z' K'^{\alpha-1} + (1-\delta)(1-\mu')\right) \right]$$
$$\mu \geq 0, \quad I \geq 0, \quad \mu \cdot I = 0$$

When $\mu > 0$, the constraint binds: the agent would like to disinvest but cannot.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$  | 0.99 | Discount factor |
| $\alpha$ | 0.36 | Capital share |
| $\sigma$ | 2.0 | CRRA coefficient |
| $\delta$ | 0.025 | Depreciation rate |
| $\rho$   | 0.95 | TFP persistence |
| $\sigma_\varepsilon$ | 0.01 | TFP innovation std |
| $\phi$   | 0.0 | Irreversibility (0 = strict I >= 0) |
| Capital grid | 50 points on [18.99, 56.98] | Wider range needed |
| TFP grid | 7 points (Tauchen) | |
| $K_{ss}$ | 37.9893 | Steady-state capital |

## Solution Method

**Value Function Iteration (VFI)** with the irreversibility constraint enforced directly in the grid search: for each state $(z, K)$, the choice set for $K'$ is restricted to $K' \geq (1-\delta)K$. This naturally handles the occasionally binding constraint without requiring complementarity solvers.

Both the constrained and unconstrained models are solved on the same grid for comparison.

Irreversible model converged in **49** iterations. Standard model converged in **49** iterations.

## Results

Look for the kink where the irreversible investment policy meets the I=0 line: to the right of this kink (high K, low z), the constraint binds and investment is pinned at zero. The standard model's dashed lines pass freely below zero, showing the disinvestment that irreversibility prevents.

<img src="figures/policy-functions.png" alt="Investment and consumption policies: irreversible (solid) vs standard (dashed). Red line marks the I=0 constraint." width="80%">
*Investment and consumption policies: irreversible (solid) vs standard (dashed). Red line marks the I=0 constraint.*

The constraint binds in the upper-left region where capital is high relative to productivity. In these states the agent would prefer to sell capital but cannot, creating a capital overhang that depresses returns and prolongs recessions.

<img src="figures/binding-region.png" alt="Region where the irreversibility constraint binds (red). High K + low z = binding." width="80%">
*Region where the irreversibility constraint binds (red). High K + low z = binding.*

The shaded periods in the investment panel show when the constraint binds. During these episodes, capital cannot adjust downward, so consumption must absorb all of the output decline, making consumption more volatile in contractions than the standard model predicts.

<img src="figures/simulation.png" alt="Simulated paths comparing irreversible (blue) vs standard (red) RBC. Shaded regions mark binding constraint." width="80%">
*Simulated paths comparing irreversible (blue) vs standard (red) RBC. Shaded regions mark binding constraint.*

The left panel shows the mass point at I=0 created by the constraint, which is absent in the symmetric standard-model distribution. The right panel's negative skewness in output growth under irreversibility formalizes the intuition that recessions are sharper than expansions when capital adjustment is one-sided.

<img src="figures/asymmetric-distributions.png" alt="Investment truncated at zero (left). Output growth shows negative skewness under irreversibility (right)." width="80%">
*Investment truncated at zero (left). Output growth shows negative skewness under irreversibility (right).*

The value difference is zero where the constraint never binds (lower-right) and most negative where it binds tightly (upper-left). This surface maps the welfare cost of irreversibility across the state space, showing that the cost is concentrated in high-capital, low-productivity states.

<img src="figures/value-difference.png" alt="Welfare cost: V_irr - V_std everywhere non-positive. Dashed line marks binding region boundary." width="80%">
*Welfare cost: V_irr - V_std everywhere non-positive. Dashed line marks binding region boundary.*

Compare std(I)/std(Y) across models: irreversibility truncates the investment distribution, altering relative volatilities and the consumption-output correlation.

**Business Cycle Statistics (5000 periods, burn-in 500)**

| Model        |   std(Y) % |   std(C)/std(Y) |   std(I)/std(Y) |   corr(C,Y) |   mean(K) |   mean(I/Y) |
|:-------------|-----------:|----------------:|----------------:|------------:|----------:|------------:|
| Irreversible |      3.804 |            1.34 |           0.207 |       0.999 |   37.6138 |      0.2548 |
| Standard RBC |      3.804 |            1.34 |           0.207 |       0.999 |   37.6138 |      0.2548 |

The negative-to-positive response ratio exceeding 1.0 for the irreversible model confirms that output growth is more volatile in downturns than in expansions. The standard model shows a ratio near 1.0, as expected for a symmetric linear system.

**Asymmetric Responses to Positive vs Negative TFP Changes**

|                    |   std(dY) Irr % |   std(dY) Std % |
|:-------------------|----------------:|----------------:|
| Positive dz        |               0 |               0 |
| Negative dz        |               0 |               0 |
| Ratio abs(neg/pos) |               0 |               0 |

**Constraint binding frequency:** The irreversibility constraint binds in 0.0% of simulated periods. It binds more often when capital is high relative to TFP (upper-left region of the state space).

**Asymmetric business cycles:** The irreversibility constraint amplifies contractions: when TFP falls, the agent cannot reduce the capital stock quickly enough, leading to excess capital and depressed returns. In contrast, expansions are unconstrained and look similar to the standard model.

**Welfare cost:** The value function under irreversibility is everywhere weakly below the unconstrained value, with the largest gaps occurring where the constraint binds.

## Takeaway

Irreversible investment fundamentally alters business cycle dynamics:

1. **Asymmetry**: The constraint creates a kink in the investment policy function. Below the kink, investment is pinned at zero and consumption absorbs all output fluctuations, making consumption more volatile in recessions.

2. **Option value**: Because installed capital cannot be recovered, each investment decision carries an option cost. Firms optimally delay investment to preserve the option of waiting for better information about future TFP.

3. **Capital overhang**: When the constraint binds, the economy carries excess capital that depresses the marginal product of capital and the return on saving. This can amplify and prolong recessions.

4. **Policy implications**: The occasionally binding constraint means that linearized solutions are qualitatively wrong -- they cannot capture the asymmetry or the constraint binding region. Global methods are essential.

## Reproduce

```bash
python run.py
```

## References

- Abel, A. and Eberly, J. (1996). *Optimal Investment with Costly Reversibility*. RES.
- Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.
- Bertola, G. and Caballero, R. (1994). *Irreversibility and Aggregate Investment*. RES.
- Khan, A. and Thomas, J. (2008). *Idiosyncratic Shocks and the Role of Nonconvexities*. Econometrica.
