# Solow Growth and Conditional Convergence

> Exogenous saving, a one-line transition map, and a closed-form steady state for the level-versus-trend split.

## Overview

Solow strips out the household optimization problem. With saving fixed at a constant fraction of output, capital accumulation reduces to a one-line scalar map and the entire model can be solved with a calculator. That is not a limitation; it is what makes the model useful. The level effects of saving and depreciation become legible without taking a stand on preferences or the intertemporal elasticity of substitution.

The state is $k_t = K_t/(A_t L_t)$, capital per unit of effective labor. Whether $k_t$ rises or falls depends only on whether gross investment $s k_t^\alpha$ exceeds the dilution required to keep $k_t$ constant against depreciation, population growth, and labor-augmenting technical change. Below the cutoff capital deepens; above it dilution wins. Concavity of $f(k)=k^\alpha$ pins down a unique nonzero fixed point.

Solow sits between two nearby tutorials. [Cake eating](../cake-eating/) has a Bellman equation but no production. [Optimal growth](../optimal-growth/) has both production and an Euler equation. Solow keeps the production side and drops the Euler equation by fiat, which is what makes the level-versus-trend split so easy to read off.

## Equations

Let $K_t$ denote aggregate capital, $A_t$ labor-augmenting technology, and
$L_t$ raw labor. Output is Cobb-Douglas:

$$Y_t = K_t^\alpha (A_t L_t)^{1-\alpha}, \qquad \alpha\in(0,1).$$

Capital, technology, and labor evolve as

$$K_{t+1}=(1-\delta)K_t + sY_t, \qquad
A_{t+1}=(1+g)A_t, \qquad L_{t+1}=(1+n)L_t,$$

where $s$ is the saving rate, $\delta$ is depreciation, $g$ is
labor-augmenting productivity growth, and $n$ is labor-force growth. Switching
to effective-labor units,

$$k_t = \frac{K_t}{A_t L_t}, \qquad
y_t = \frac{Y_t}{A_t L_t} = k_t^\alpha,$$

the discrete-time law of motion collapses to a single scalar equation,

$$k_{t+1} = \phi(k_t) \;:=\; \frac{(1-\delta)\,k_t + s\,k_t^\alpha}{(1+g)(1+n)}.$$

The steady state $k^{\ast}$ solves $\phi(k^{\ast})=k^{\ast}$, equivalently
$s(k^{\ast})^\alpha = \Delta k^{\ast}$, where

$$\Delta \;:=\; (1+g)(1+n) - 1 + \delta$$

is the per-unit break-even investment required to keep $k$ constant. Hence

$$k^{\ast}=\left(\frac{s}{\Delta}\right)^{1/(1-\alpha)}, \qquad
y^{\ast}=(k^{\ast})^\alpha, \qquad c^{\ast}=(1-s)\,y^{\ast}.$$

Competitive factor prices follow from marginal products:

$$MPK_t = \alpha\, k_t^{\alpha-1}, \qquad
\frac{w_t}{A_t} = (1-\alpha)\, k_t^\alpha.$$

The plotted wage is $w_t/A_t$, the wage per unit of effective labor. The wage
per raw worker is $w_t = (1-\alpha)\,A_t\,k_t^\alpha$ and grows with $A_t$
along the balanced-growth path.

## Model Setup

| Symbol | Value | Role |
|--------|------:|------|
| $\alpha$ | 0.33 | Capital share in $K^\alpha(AL)^{1-\alpha}$ |
| $s$ | 0.24 | Exogenous saving rate |
| $\delta$ | 0.06 | Capital depreciation |
| $n$ | 0.01 | Labor-force growth |
| $g$ | 0.02 | Labor-augmenting productivity growth |
| $K_0,A_0,L_0$ | 1.0, 1.0, 1.0 | Initial stocks; implies $k_0=1.0$ |
| Horizon $T$ | 160 | Long enough to make the residual finite-horizon gap visible |
| $\Delta$ | 0.0902 | Break-even investment per unit of $k$ |
| $k^{\ast}$ | 4.3086 | Closed-form steady-state capital per effective worker |

## Solution Method

There is no Bellman equation here. Once $s$ is fixed, the model is the scalar map $\phi$ from the previous section, and $k^{\ast}$ has a closed form. The simulation iterates $\phi$ from $k_0$, and the closed-form steady state plays the role that a finely solved benchmark plays in less tractable problems: ground truth that the iteration is audited against.

Local convergence is read off the linearization of $\phi$ at the steady state:

$$k_{t+1} - k^{\ast} \;\approx\; \lambda\,(k_t - k^{\ast}), \qquad \lambda \;\equiv\; \phi'(k^{\ast}) \;=\; \frac{(1-\delta) + s\alpha\,(k^{\ast})^{\alpha-1}}{(1+g)(1+n)}.$$

When $\lambda \in (0,1)$, deviations from the balanced-growth path decay geometrically with half-life $\ln(0.5)/\ln(\lambda)$. With saving rates and depreciation rates calibrated to advanced economies, $\lambda$ is typically close to one and the half-life runs to decades. That slow rate is the empirical fact behind Mankiw, Romer, and Weil (1992): countries do converge to their own balanced-growth paths, but slowly enough that initial conditions still show up in cross-section growth regressions a generation later.

```text
Algorithm: Solow transition in effective-labor units
Input : primitives (alpha, s, delta, n, g), initial k0, horizon T
Output: paths {k_t, y_t, c_t, MPK_t, w_t/A_t};
        closed-form k_star, local rate lambda, half-life H

Delta   <- (1 + g)(1 + n) - 1 + delta              # break-even per unit k
k_star  <- (s / Delta)^(1 / (1 - alpha))           # closed-form fixed point
lambda  <- ((1 - delta) + s * alpha * k_star^(alpha - 1)) / ((1 + g)(1 + n))
H       <- ln(0.5) / ln(lambda)                    # local half-life

set k <- k0
for t = 0, 1, ..., T - 1:
    y_t       <- k^alpha
    c_t       <- (1 - s) * y_t
    invest_t  <- s * y_t
    MPK_t     <- alpha * k^(alpha - 1)
    w_t / A_t <- (1 - alpha) * k^alpha
    k         <- ((1 - delta) * k + s * y_t) / ((1 + g)(1 + n))

audit         : |k_T - k_star|, |y_T - y_star|, |c_T - c_star|
linearization : compare k_t to k_star + (k_0 - k_star) * lambda^t
```

For this calibration, $\lambda \approx 0.941$ and the local half-life is roughly 11.5 periods. Annual $g+n+\delta$ near nine percent makes the transition feel slow no matter what $s$ is set to.

## Results

At $k_0=1.00$ the curved schedule $s k^\alpha$ sits above the linear break-even line $\Delta k$, so $k_t$ deepens from the start. The two curves cross at $k^{\ast}=4.309$, the unique nonzero fixed point. Concavity of $f(k)=k^\alpha$ guarantees a single intersection and a stable one: above $k^{\ast}$ the linear schedule grows faster than the concave one and dilution wins. The crossing is not estimated from the simulation; it is the closed-form $(s/\Delta)^{1/(1-\alpha)}$ implied by the primitives.

<img src="figures/solow-diagram.png" alt="Solow diagram with investment and break-even investment in effective-labor units" width="80%">

All three series are normalized by their balanced-growth values. Output and consumption track each other identically because $c_t=(1-s)y_t$; the saving rate is the choice the model has refused to make. Capital lags both because it inherits its own past stock, and the gap closes at the geometric rate $\lambda$ derived above. The dotted black line is the linear-approximation prediction $k^{\ast}+(k_0-k^{\ast})\lambda^t$, plotted in the same normalized units. Linearization tracks the simulation closely as $k_t$ nears $k^{\ast}$ but is visibly off early on, where the curvature of $\phi$ still matters. By the terminal period, the simulated $k$ sits within 2.73e-04 of $k^{\ast}$ in absolute terms.

<img src="figures/transition-effective-units.png" alt="Capital, output, and consumption converging to steady state, with the linear approximation overlaid" width="80%">

Factor prices read the same convergence story from the firm side. Early on, capital is scarce, so $MPK_t$ is high and the effective wage is depressed. As $k_t$ deepens, both move monotonically toward the steady-state values implied by $k^{\ast}$: $MPK^{\ast}=0.124$ and $(w/A)^{\ast}=1.085$. In a multi-country reading this is the textbook prediction that the return on capital should be falling and wages rising as poorer economies catch up to richer ones with the same technology.

<img src="figures/factor-prices.png" alt="Factor prices along the Solow transition" width="80%">

Two ways to read the same model. The left panel runs three economies with identical primitives but different starting capital: $k_0$ at one fifth, one, and twice the steady state. All three converge to the same $k^{\ast}=1$ in normalized units. *Conditional* matters here, because the common steady state is the one pinned down by $(s,n,g,\delta,\alpha)$, not a common world level. The right panel makes the same point algebraic. Three saving rates, $s\in\{0.18, 0.24, 0.30\}$, slide the investment schedule up while leaving $\Delta k$ fixed; the new intersections give $k^{\ast}\in\{2.80,\,4.31,\,6.01\}$. Doubling $s$ raises $k^{\ast}$ by a factor of $2^{1/(1-\alpha)}\approx 2.81$, but once at the new steady state output per worker still grows at $g$. Permanently faster growth in this model has to come from $g$, not $s$.

<img src="figures/convergence-and-comparative-statics.png" alt="Conditional convergence from three starting points and the comparative statics of the saving rate" width="80%">

Both the transition map and the steady state are analytical, so any remaining gap in the table is finite-horizon truncation, not numerical error. The bound is the geometric residual $|k_0-k^{\ast}|\,\lambda^{T-1}$, which at $\lambda=0.941$ and $T=160$ is on the order of 2.21e-04.

**Closed-form steady state versus terminal simulation**

| Object                             |   Closed form |   Simulated t=159 |   Absolute gap |
|:-----------------------------------|--------------:|------------------:|---------------:|
| Capital per effective worker k     |      4.30859  |           4.30832 |       0.000273 |
| Output per effective worker y      |      1.61931  |           1.61928 |       3.38e-05 |
| Consumption per effective worker c |      1.23068  |           1.23065 |       2.57e-05 |
| Marginal product of capital MPK    |      0.124025 |           0.12403 |       5.26e-06 |
| Effective wage w/A                 |      1.08494  |           1.08492 |       2.27e-05 |

## Takeaway

Solow disciplines what saving can and cannot do. A higher $s$ raises the level of $k^{\ast}$ but leaves the long-run growth rate of output per worker equal to $g$. The same logic delivers conditional convergence: economies with the same primitives approach the same balanced-growth path, while economies that differ in $s$, $n$, or $\delta$ approach different ones. [Optimal growth](../optimal-growth/) lifts the constant-saving assumption and lets an Euler equation choose $s$ endogenously; what survives is the same level-versus-trend split that Solow puts on a single line of algebra.

## References

- Solow, R. M. (1956). "A Contribution to the Theory of Economic Growth." *Quarterly Journal of Economics*, 70(1), 65-94.
- Mankiw, N. G., Romer, D., and Weil, D. N. (1992). "A Contribution to the Empirics of Economic Growth." *Quarterly Journal of Economics*, 107(2), 407-437.
- Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 1.
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 1.
- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 2.
