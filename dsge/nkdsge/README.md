# New Keynesian Monetary Shocks and Determinacy

> Sticky prices, the IS curve, Phillips curve, and Taylor rule in a three-equation DSGE model.

## Overview

The question is why a central bank can move real activity when households and firms understand the policy rule. In this small New Keynesian model the answer is sticky prices. A surprise increase in the nominal interest rate raises the real rate before prices fully adjust, so demand falls. The Phillips curve then translates the weaker output gap into lower inflation.

The model is deliberately small: an output gap $y_t$, inflation $\pi_t$, a nominal policy rate $i_t$, and one persistent shock at a time. The `model.mod` spec sitting alongside `run.py` records the three-equation block in DSL syntax for documentation; the Python code solves the log-linear system directly, which makes the expectations algebra and the Taylor-rule determinacy condition easy to inspect. The same system is also solved by Klein (2000) generalized Schur (QZ) decomposition as a cross-check; the two methods agree to machine precision on this problem.

Compared with the [RBC tutorial](../rbc/), propagation here does not come from slow capital accumulation. It comes from the interaction between forward-looking demand, sticky-price inflation, and a policy rule that leans against inflation.

## Equations

All variables are deviations from the zero-inflation steady state. Let $y_t$ be
the output gap, $\pi_t$ inflation, $i_t$ the nominal policy rate, and $r^n_t$ the
natural real rate. The three equations are

$$
y_t =
\mathbb{E}_t y_{t+1} - \frac{1}{\sigma}
\left(i_t-\mathbb{E}_t\pi_{t+1}-r^n_t\right),
$$

$$
\pi_t = \beta \mathbb{E}_t \pi_{t+1}+\kappa y_t,
$$

$$
i_t = \phi_\pi \pi_t+\phi_y y_t+v_t.
$$

The monetary-policy disturbance follows

$$
v_t=\rho_v v_{t-1}+\varepsilon^v_t,
$$

and the demand experiment treats the natural-rate term as

$$
r^n_t=d_t,\qquad d_t=\rho_d d_{t-1}+\varepsilon^d_t.
$$

The accompanying `model.mod` spec writes the same core block as

```text
y = y(+1) - sigma^(-1)*(i - pi(+1) - rho)
pi = beta*pi(+1) + k*y
i = rho + phi_pi*pi + phi_y*y + e
```

The Python report uses $v_t$ for the Taylor-rule shock and $d_t$ for the
natural-rate shifter so the two impulse responses can be read separately.

## Model Setup

| Primitive | Value | Role |
|---|---:|---|
| $\sigma$ | 1 | Inverse EIS in the IS curve |
| $\beta$ | 0.99 | Quarterly discount factor |
| $\kappa$ | 0.3 | Slope of the New Keynesian Phillips curve |
| $\phi_\pi$ | 1.5 | Taylor-rule response to inflation |
| $\phi_y$ | 0.125 | Taylor-rule response to the output gap |
| $\rho_v$ | 0.5 | Persistence of the policy shock |
| $\rho_d$ | 0.8 | Persistence of the demand shock |
| Shock innovation | 0.010 | One-percentage-point innovation at date 0 |
| IRF horizon | 40 quarters | Periods shown in each impulse response |

The source `model.mod` uses $\phi_\pi=0.33$ and $\kappa=0.95$. This report uses a standard determinate calibration, $\phi_\pi=1.5$ and $\kappa=0.3$, because the economic point is monetary transmission under a stable Taylor rule. The contrast matters: when policy fails to lean hard enough against inflation, the forward-looking system no longer selects a unique stable path.

## Solution Method

For either shock, write the scalar state as $s_t=\rho_s s_{t-1}+\varepsilon_t$. Because the model is already log-linear, the rational-expectations solution is linear in that state:

$$y_t=\psi_y s_t,\qquad \pi_t=\psi_\pi s_t. $$

The Phillips curve gives

$$\psi_\pi=\frac{\kappa\psi_y}{1-\beta\rho_s}. $$

The IS curve and Taylor rule then pin down $\psi_y$. For a monetary-policy shock, the right-hand side is negative because $v_t$ raises the policy rate. For a demand shock, the right-hand side is positive because $d_t$ raises the natural rate:

$$\psi_y\left[(1-\rho_s)+\frac{\phi_y}{\sigma}+\frac{(\phi_\pi-\rho_s)\kappa}{\sigma(1-\beta\rho_s)}\right]= b_s,$$

where $b_s=-1/\sigma$ for $s_t=v_t$ and $b_s=1$ for $s_t=d_t$.

```text
Algorithm: New Keynesian impulse responses
Inputs: beta, sigma, kappa, phi_pi, phi_y, rho_s, shock eps_0, horizon T
Outputs: paths for y_t, pi_t, i_t, and the shock state s_t

1. Pick the shock experiment: monetary policy v_t or natural-rate demand d_t.
2. Guess y_t = psi_y s_t and pi_t = psi_pi s_t.
3. Use the Phillips curve to express psi_pi in terms of psi_y.
4. Substitute both coefficients into the IS curve and Taylor rule.
5. Match coefficients on s_t to solve for psi_y, then recover psi_pi.
6. Recover the policy-rate coefficient psi_i from the Taylor rule.
7. Set s_0 = eps_0 and iterate s_t = rho_s s_{t-1} for t = 1,...,T.
8. Plot y_t = psi_y s_t, pi_t = psi_pi s_t, and i_t = psi_i s_t.
```

There is no finer-grid benchmark to add here. Within this tutorial's log-linear model, coefficient matching is the exact solution. As an independent check the same system is also solved by Klein (2000) generalized Schur (QZ) decomposition; the two methods agree to 1.4e-15 on both shock experiments. The Klein algorithm is what general DSGE solvers use because it scales to many states; on this small model it is overkill, but the agreement establishes that the closed-form coefficients pick out the unique stable rational-expectations equilibrium. Approximation error would enter only if we replaced the three-equation block with a nonlinear price-setting model and then compared a local perturbation to a global or perfect-foresight solution.

## Results

The monetary shock is a wedge in the Taylor rule, not the total policy-rate response. On impact the wedge is one percentage point, but the systematic part of the rule partly offsets it because expected output and inflation fall. The real rate still rises, demand contracts, and inflation falls with the output gap. Persistence in $v_t$ controls how slowly the economy returns to steady state.

<img src="figures/irf-monetary-shock.png" alt="Impulse responses to a one-percentage-point contractionary monetary-policy shock" width="80%">

A positive natural-rate shock pushes current demand up at the same nominal rate. Output and inflation therefore rise together. The Taylor rule raises the policy rate in response, which dampens but does not eliminate the expansion because the shock is persistent and agents expect demand pressure to continue.

<img src="figures/irf-demand-shock.png" alt="Impulse responses to a one-percentage-point natural-rate demand shock" width="80%">

The impact table gives the signs and scale without asking the reader to read them off the figure. Output is in percent deviations; inflation and the policy rate are in quarterly percentage points. Monetary and demand shocks move output and inflation in opposite directions across experiments because the shocks enter different equations.

**Impact Responses to One-Percentage-Point Shocks**

| Variable     |   Monetary shock impact |   Demand shock impact |
|:-------------|------------------------:|----------------------:|
| Output gap   |                  -0.82  |                 0.749 |
| Inflation    |                  -0.487 |                 1.081 |
| Nominal rate |                   0.166 |                 1.715 |

## Takeaway

The three-equation New Keynesian model is compact, but it already separates two central ideas. First, sticky prices let a nominal policy surprise move the real rate and therefore current demand. Second, determinacy is not a numerical detail: with forward-looking inflation, the Taylor rule has to make expected inflation costly enough for the model to select one stable path.

The policy-shock and demand-shock experiments use the same solution method but differ in their economics. A policy wedge contracts demand and inflation. A natural-rate shock expands both, with the central bank leaning back through the Taylor rule. For a supply or cost-push shock, the same block would show the sharper output-inflation stabilization trade-off.

## References

- Gali, J. (2015). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press, 2nd edition.
- Woodford, M. (2003). *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.
- Clarida, R., Gali, J., and Gertler, M. (1999). The Science of Monetary Policy: A New Keynesian Perspective. *Journal of Economic Literature*, 37(4), 1661-1707.
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
