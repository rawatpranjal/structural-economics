# Fiscal-Shock Persistence and Income Dynamics

> How AR(1) propagation and a multiplier-accelerator block turn a spending innovation into an income path.

## Overview

Suppose a fiscal authority raises government spending during a downturn. The relevant policy question goes beyond the impact effect: how long does the spending impulse remain in the economy, and how does income react while households and firms adjust with lags? A one-quarter disturbance can fade quickly or behave like a slow-moving state. Those two cases lead to different business-cycle stories even before a full DSGE model is added.

This tutorial keeps the environment small so the timing is visible. We first compute the exact propagation of an AR(1) state, the workhorse law for technology, demand, and policy shocks. We then feed a persistent spending shock into Samuelson's multiplier-accelerator economy. Consumption follows lagged income, investment reacts to consumption growth, and output clears the accounting identity. The computation turns the primitive persistence parameter into impulse responses, autocorrelations, spectra, and simulated income paths.

## Equations

Let $x_t$ denote a generic shock state, such as technology, demand pressure, or
the discretionary component of fiscal policy. The scalar law of motion is

$$
x_t = \rho x_{t-1} + \varepsilon_t, \qquad
\varepsilon_t \sim N(0,\sigma^2), \qquad |\rho|<1.
$$

The coefficient $\rho$ tells us how much of today's state becomes tomorrow's
state. A high value makes the shock part of the short-run macro environment
rather than a one-period disturbance.

In the multiplier-accelerator economy, income $Y_t$ equals consumption $C_t$,
investment $I_t$, and government spending $G_t$:

$$
C_t = \beta Y_{t-1},
\qquad
G_t = \rho_g G_{t-1} + (1-\rho_g)\bar G + \varepsilon_t,
$$

$$
I_t = \alpha(C_t-C_{t-1}),
\qquad
Y_t = C_t + I_t + G_t.
$$

The steady state is $\bar Y=\bar G/(1-\beta)$, $\bar C=\beta \bar Y$, and
$\bar I=0$. Writing lowercase variables as deviations from that steady state
gives the income recursion used for the impulse responses:

$$
y_t = \beta(1+\alpha)y_{t-1}-\alpha\beta y_{t-2}+g_t,
\qquad
g_t=\rho_g g_{t-1}+\varepsilon_t.
$$

## Model Setup

**AR(1) shock process**

| Parameter | Value | Role |
|---|---:|---|
| $\rho$ | 0.90 | Share of the shock state carried into the next period |
| $\sigma$ | 0.01 | Standard deviation of new innovations |
| $T_{sim}$ | 220 | Simulated periods after burn-in |

**Multiplier-accelerator economy**

| Parameter | Value | Role |
|---|---:|---|
| $\alpha$ | 0.30 | Accelerator response of investment to consumption growth |
| $\beta$ | 0.80 | Marginal propensity to consume out of lagged income |
| $\rho_g$ | 0.90 | Carryover of government-spending deviations |
| $\bar G$ | 1.00 | Steady-state government spending |
| $\bar Y$ | 5.00 | Implied steady-state income |
| $\bar C$ | 4.00 | Implied steady-state consumption |

## Solution Method

A realized shock path pins down every variable because both blocks are backward-looking. There is no expectations fixed point in this example. That simplicity is useful: it lets us separate the persistence built into the exogenous state from the propagation created by lagged consumption and accelerator investment.

For the AR(1), the main population objects are available in closed form:

$$E[x_t]=0, \qquad \operatorname{Var}(x_t)=\frac{\sigma^2}{1-\rho^2}=0.000526, \qquad \operatorname{Corr}(x_t,x_{t-k})=\rho^k.$$

The half-life is $\log(0.5)/\log(\rho)=6.6$ periods. For the multiplier-accelerator block, the endogenous income roots are 0.346, 0.694; their largest modulus is 0.694, so the calibrated internal propagation is stable.

```text
Procedure: propagate a fiscal innovation through an AR(1) state
Inputs: rho, sigma, alpha, beta, rho_g, horizon T, shock sequence eps_t
Outputs: AR path x_t and multiplier-accelerator paths y_t, c_t, i_t, g_t

1. For an impulse response, set eps_0 = 1 and eps_t = 0 for t > 0.
   For a simulation, draw eps_t from N(0, sigma^2) and discard burn-in.
2. AR(1): update x_t = rho x_{t-1} + eps_t.
3. Government spending: update g_t = rho_g g_{t-1} + eps_t.
4. Consumption: set c_t = beta y_{t-1}.
5. Investment: set i_t = alpha(c_t - c_{t-1}).
6. Income: impose the identity y_t = c_t + i_t + g_t.
7. Summarize x_t with half-lives, autocorrelations, and spectra.
8. Read the y_t path as income propagation from the same fiscal shock.
```

## Results

A unit shock follows the exact path $\rho^h$. Raising $\rho$ from 0.5 to 0.9 changes the half-life from one period to roughly seven periods. At $\rho=0.99$, the shock behaves like a slow-moving state over the length of many macro samples.

<img src="figures/ar1-irfs.png" alt="Exact AR(1) impulse responses by persistence" width="80%">

Government spending decays smoothly after the innovation. Income inherits that persistence, then adds its own timing through lagged consumption and accelerator investment. The accelerator channel is largest near turning points, when consumption growth changes most.

<img src="figures/multiplier-accelerator-irfs.png" alt="Multiplier-accelerator impulse responses to a government spending shock" width="80%">

The simulated AR(1) path sits inside its analytic two-standard-deviation band for most dates, which gives the finite sample a population benchmark. In the multiplier-accelerator panel, income and consumption move together but not at the same date. Lagged consumption supplies the memory that carries government spending shocks forward.

<img src="figures/simulated-paths.png" alt="Simulated AR(1) and multiplier-accelerator paths" width="80%">

The left panel checks the simulation against the AR(1) population benchmark $\rho^k$, with remaining gaps from finite-sample noise. The right panel shows that the multiplier-accelerator economy creates serial correlation in income because government spending is filtered through lagged consumption.

<img src="figures/autocorrelation.png" alt="Autocorrelation functions for the AR(1) and multiplier-accelerator output" width="80%">

High persistence loads variance at low frequencies. In a short sample, the state can look like a slow cycle or trend rather than a sequence of isolated surprises. This is why the choice of $\rho$ in a DSGE shock process changes business-cycle timing as well as unconditional volatility.

<img src="figures/spectral-density.png" alt="Exact AR(1) spectral density by persistence" width="80%">

The analytic benchmarks show the calibration stakes. Holding $\sigma$ fixed, a higher $\rho$ raises the variance of the state and lengthens the time a shock remains economically relevant.

**AR(1) Analytical Benchmarks**

| Object                      | $\rho=0.5$   | $\rho=0.9$   | $\rho=0.99$   |
|:----------------------------|:-------------|:-------------|:--------------|
| Persistence ($\rho$)        | 0.50         | 0.90         | 0.99          |
| Unconditional variance      | 0.000133     | 0.000526     | 0.005025      |
| Half-life (periods)         | 1.0          | 6.6          | 69.0          |
| First-order autocorrelation | 0.50         | 0.90         | 0.99          |
| Spectral peak               | Frequency 0  | Frequency 0  | Frequency 0   |

## Takeaway

An AR(1) coefficient is an economic timing assumption. With $\rho=0.9$, a shock still has about half of its initial effect after 6.6 periods; with $\rho=0.99$, the same calculation gives 69.0 periods. A macro model then maps that persistent state into observables through its own accounting or equilibrium equations.

The multiplier-accelerator economy makes that mapping visible. Government spending supplies the persistent disturbance. Lagged consumption and accelerator investment decide how the disturbance becomes an income path.

## References

- Hamilton, J. (1994). *Time Series Analysis*. Princeton University Press.
- Samuelson, P. (1939). Interactions between the Multiplier Analysis and the Principle of Acceleration. *Review of Economics and Statistics*, 21(2), 75-78.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 2.
