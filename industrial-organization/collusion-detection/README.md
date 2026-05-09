# Repeated-Game Cartels and Price Screens

## Overview

Cartel investigations often start with a price path. Prices rise together and stay high. They fall after detection or a breakdown.

The object is a repeated agreement among Cournot firms. Each firm weighs future cartel profit against today's gain from extra output.

The computation compares Nash, collusive, and deviation profits. Those payoffs give the discount-factor threshold and benchmarks for a simulated price screen.

## Equations

Firms $i=1,\ldots,n$ choose quantities $q_i$. Total quantity is
$Q=\sum_i q_i$, inverse demand is $P(Q)=a-Q$, and all firms have constant
marginal cost $c<a$. Let $\delta\in(0,1)$ be the common discount factor.

| Regime | Per-firm quantity | Per-firm profit |
|---|---:|---:|
| Cournot-Nash | $q^N=\dfrac{a-c}{n+1}$ | $\pi^N=\left(\dfrac{a-c}{n+1}\right)^2$ |
| Joint monopoly split equally | $q^M=\dfrac{a-c}{2n}$ | $\pi^M=\dfrac{(a-c)^2}{4n}$ |
| One firm deviates while others collude | $q^D=\dfrac{(n+1)(a-c)}{4n}$ | $\pi^D=\dfrac{(n+1)^2(a-c)^2}{16n^2}$ |

A grim-trigger cartel colludes until a deviation is detected before reverting
to Cournot-Nash forever. The value of staying in the cartel is

$$
V^M=\frac{\pi^M}{1-\delta},
$$

while the value of deviating once is

$$
V^D=\pi^D+\frac{\delta\pi^N}{1-\delta}.
$$

The incentive constraint $V^M\geq V^D$ is equivalent to

$$
\delta\geq
\delta^{\ast}
=\frac{\pi^D-\pi^M}{\pi^D-\pi^N}
=\frac{(n+1)^2}{n^2+6n+1}.
$$

The price-screen simulation uses the same exact benchmarks. It observes

$$
P_t=P^{r_t}+\eta_t,\qquad
r_t\in\{N,M,N\},
$$

Here $P^N$ is the Cournot price. $P^M$ is the joint-monopoly price. The regime
$r_t$ moves from competition to cartel conduct and back after detection.

The reported margin is $m_t=(P_t-c)/P_t$.

## Model Setup

The calibration is small enough to audit by hand. Demand and cost pin down Nash and monopoly prices. The simulated data add only regime timing and noise.

| Object | Value | Role |
|---|---:|---|
| Demand intercept $a$ | 100 | Sets the competitive and monopoly price benchmarks |
| Marginal cost $c$ | 40 | Common cost used in profits and margins |
| Baseline firms | 2 | Duopoly used for the simulated price path |
| Firm-count grid | 2 to 50 | Exact cartel-stability thresholds by $n$ |
| Reference patience | $\delta=0.9$ | Used to mark which firm counts are sustainable |
| Regimes | 30+25+20 periods | Competition, cartel, post-detection |
| Price noise | $\sigma=1.5$ | Adds sampling noise around the exact regime price |

In the baseline duopoly, the Cournot price is 60. The monopoly price is 70. A deviating firm earns 506.2 when its rival holds the cartel quantity.

## Solution Method

First compute payoffs for each firm count. Then evaluate the repeated-game threshold. For the duopoly, simulate prices around Nash, monopoly, then Nash again. The margin series uses those prices and constant cost.

```text
Algorithm: repeated-Cournot cartel screen
Input: demand intercept a, marginal cost c, firm-count grid N, discount factor delta
Output: delta*(n), sustainability flags, price and margin benchmarks
1. For each n in N, compute the symmetric Cournot payoff pi^N(n).
2. Compute the equal-split joint-monopoly payoff pi^M(n).
3. Let one firm best respond to the other n-1 firms' collusive quantities;
   record the one-shot deviation payoff pi^D(n).
4. Evaluate delta*(n) = [pi^D(n)-pi^M(n)] / [pi^D(n)-pi^N(n)].
5. Mark collusion sustainable when delta >= delta*(n).
6. For the baseline duopoly, simulate prices around P^N, then P^M,
   then P^N again; compute margins m_t = (P_t-c)/P_t.
```

With $\delta=0.9$, the threshold allows at most 33 symmetric firms.

## Results

The two sides of the incentive constraint are visible directly. The distance from $\pi^M$ up to $\pi^D$ is the short-run gain from cheating. The distance from $\pi^N$ up to $\pi^M$ is the per-period rent lost after punishment. Adding members dilutes the monopoly rent faster than it shrinks the deviation opportunity.

<img src="figures/profits-by-regime.png" alt="Per-firm Nash, collusive, and deviation profits by firm count" width="80%">

The threshold curve is exact for the linear Cournot model. At $\delta=0.9$, 33 firms are still sustainable. At 34 firms, the deviation constraint fails.

<img src="figures/critical-discount-factor.png" alt="Exact critical discount factor as a function of the number of firms" width="80%">

The simulated price path has known regimes. Prices begin near the Nash benchmark, move toward monopoly, then return to Nash. This is a clean benchmark, not a detection test.

<img src="figures/price-series-structural-break.png" alt="Stylized price series with Nash and monopoly reference prices" width="80%">

The margin plot repeats the same break after normalizing by price. Because cost is constant, it adds no new identification in this toy run.

<img src="figures/price-cost-margin.png" alt="Stylized price-cost margin with Nash and monopoly reference margins" width="80%">

The table reports exact payoffs and thresholds. For $\delta=0.9$, the feasibility cutoff lies between 33 and 34 firms. The high-$n$ rows show how quickly the incentive constraint tightens as the cartel has to divide monopoly rents among more members.

**Exact Cartel Stability Conditions ($a=100$, $c=40$)**

|   n |   pi_N |   pi_M |   pi_D |   delta_star | delta_0.9_sustains   |
|----:|-------:|-------:|-------:|-------------:|:---------------------|
|   2 |  400   |  450   |  506.2 |       0.5294 | yes                  |
|   3 |  225   |  300   |  400   |       0.5714 | yes                  |
|   4 |  144   |  225   |  351.6 |       0.6098 | yes                  |
|   5 |  100   |  180   |  324   |       0.6429 | yes                  |
|   6 |   73.5 |  150   |  306.2 |       0.6712 | yes                  |
|   8 |   44.4 |  112.5 |  284.8 |       0.7168 | yes                  |
|  10 |   29.8 |   90   |  272.2 |       0.7516 | yes                  |
|  15 |   14.1 |   60   |  256   |       0.8101 | yes                  |
|  20 |    8.2 |   45   |  248.1 |       0.8464 | yes                  |
|  30 |    3.7 |   30   |  240.2 |       0.889  | yes                  |
|  33 |    3.1 |   27.3 |  238.8 |       0.8975 | yes                  |
|  34 |    2.9 |   26.5 |  238.4 |       0.9001 | no                   |
|  40 |    2.1 |   22.5 |  236.4 |       0.9131 | no                   |
|  50 |    1.4 |   18   |  234.1 |       0.9286 | no                   |

## Takeaway

A price break is a lead, not a cartel finding. The repeated-game check asks whether future cartel rents can deter one-period cheating. In the duopoly, $\delta^{\ast}=0.5294$. With ten symmetric firms, $\delta^{\ast}=0.7516$. At $\delta=0.9$, the exact firm-count cutoff is 33. Cost and demand evidence still matters.

## References

- Stigler, G. (1964). A Theory of Oligopoly. *Journal of Political Economy*, 72(1), 44--61.
- Porter, R. (1983). A Study of Cartel Stability: The Joint Executive Committee, 1880--1886. *Bell Journal of Economics*, 14(2), 301--314.
- Harrington, J. (2008). Detecting Cartels. In *Handbook of Antitrust Economics*. MIT Press.
- Igami, M. and Sugaya, T. (2021). Measuring the Incentive to Collude: The Vitamin Cartels, 1990--1999. *Review of Economic Studies*, 89(3), 1460--1494.
