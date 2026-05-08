# Mixed Logit Demand with Simulated Likelihood

> Estimate random-coefficient demand and compare substitution with plain logit.

## Overview

Consumers choose among differentiated products. The econometrician observes prices, qualities, and choices, but not each consumer's price sensitivity or quality taste.

Plain logit compresses everyone into one representative taste vector. Mixed logit lets those coefficients vary across consumers. The price of that extra flexibility is numerical integration: each candidate parameter vector requires simulated choice probabilities.

The key economic issue is substitution. A plain logit can match average shares and still say that all products are equally close substitutes after conditioning on their shares. Mixed logit keeps the logit formula for a simulated consumer with fixed tastes, then averages across consumers with different tastes.

## Equations

Consumer $i$ chooses product $j$ with utility

$$
u_{ij}
= \alpha_i p_{ij} + \beta_i q_{ij} + \varepsilon_{ij},
\qquad
\varepsilon_{ij}\sim\text{Type I EV}.
$$

Random coefficients are

$$
\alpha_i = \bar\alpha + \sigma_\alpha \nu_{i\alpha},
\qquad
\beta_i = \bar\beta + \sigma_\beta \nu_{i\beta},
\qquad
\nu_i\sim N(0,I).
$$

Conditional on a draw $\nu_r$, the logit probability is

$$
\begin{aligned}
P_{ij}(\theta,\nu_r)
&=
\frac{\exp(\alpha_r p_{ij}+\beta_r q_{ij})}
{\sum_{k\in\mathcal{J}} \exp(\alpha_r p_{ik}+\beta_r q_{ik})}.
\end{aligned}
$$

The mixed-logit probability integrates over random tastes. The code approximates
that integral with fixed simulation draws:

$$
\begin{aligned}
\widehat P_{ij}(\theta)
&=
\frac{1}{R}\sum_{r=1}^R P_{ij}(\theta,\nu_r).
\end{aligned}
$$

Simulated maximum likelihood chooses

$$
\begin{aligned}
\hat\theta
&=
\arg\max_\theta
\sum_{i=1}^N \log \widehat P_{i y_i}(\theta).
\end{aligned}
$$

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Products | 4 | Differentiated alternatives in each choice set |
| Choice occasions | 1,500 | Synthetic individual-level choices |
| Simulation draws | 120 | Fixed normal draws for simulated likelihood |
| True $\bar\alpha$ | -1.00 | Mean price taste |
| True $\bar\beta$ | 1.10 | Mean quality taste |
| True $\sigma_\alpha$ | 0.36 | Heterogeneity in price sensitivity |
| True $\sigma_\beta$ | 0.55 | Heterogeneity in quality taste |

**Numerical settings**

| Setting | Value | Role |
|---------|-------|------|
| Mixed-logit optimizer | L-BFGS-B | Handles simple bounds on mean tastes and log standard deviations |
| Mixed-logit start | (-0.75, 0.85, log 0.25, log 0.35) | Initial mean tastes and heterogeneity |
| Price-taste bound | [-3.00, -0.05] | Keeps price sensitivity negative |
| Quality-taste bound | [0.05, 2.50] | Keeps quality taste positive |
| SD bounds | [0.03, 1.30] | Applied to both random-coefficient standard deviations |
| Probability floor | 1e-14 | Prevents log zero during likelihood evaluation |
| Max iterations | 220 | L-BFGS-B iteration cap |
| Profile grid | 21 x 21 | Grid over $\sigma_\alpha$ and $\sigma_\beta$ for the likelihood surface |

## Solution Method

The estimator uses common random numbers. Draws are made once and then held fixed while the optimizer moves $\theta$. This turns the population integral into the same finite average at every trial parameter vector. Without common draws, fresh simulation noise would move the likelihood surface while the optimizer is trying to climb it.

The standard deviations are optimized in logs. The optimizer can move freely over log standard deviations, while the model sees positive values after exponentiation. The bounds are not an economic restriction in this example. They keep the teaching likelihood away from numerically irrelevant regions.

```text
Algorithm 1: evaluate the simulated log likelihood
Inputs   data {y_i,p_ij,q_ij}_{i=1..N,j=1..J};
         fixed draws nu_r=(nu_{r alpha},nu_{r beta}) for r=1,...,R;
         trial theta=(alpha_bar,beta_bar,ell_alpha,ell_beta)
Output   Q_R(theta), the negative simulated log likelihood

sigma_alpha <- exp(ell_alpha)
sigma_beta  <- exp(ell_beta)
initialise P_hat_ij(theta) <- 0 for every i,j

for r = 1,...,R:
    alpha_r <- alpha_bar + sigma_alpha * nu_{r alpha}
    beta_r  <- beta_bar  + sigma_beta  * nu_{r beta}
    V_ijr   <- alpha_r p_ij + beta_r q_ij
    P_ijr   <- exp(V_ijr) / sum_{k=1}^J exp(V_ikr)
    P_hat_ij(theta) <- P_hat_ij(theta) + P_ijr / R

ell_R(theta) <- sum_{i=1}^N log max(P_hat_{i y_i}(theta), eta)
Q_R(theta)   <- -ell_R(theta) / N
return Q_R(theta)
```

```text
Algorithm 2: estimate theta and compute price substitution
Inputs   starting value theta_0; bounds B; price step Delta p;
         common draws {nu_r}; observed data {y_i,p_ij,q_ij}
Outputs  theta_hat; fitted shares s_j; price-substitution matrix D_jk

repeat until L-BFGS-B stops:
    choose a candidate theta^m in B
    evaluate Q_R(theta^m) using Algorithm 1
    update the inverse-Hessian approximation and propose the next theta

theta_hat <- argmin_theta Q_R(theta)
s_j       <- N^{-1} sum_i P_hat_ij(theta_hat)

for each shocked product k:
    set p_{ik}^{+k} <- p_{ik} + Delta p for all i
    recompute shares s_j^{+k} with the same theta_hat and the same draws
    lost_k <- s_k - s_k^{+k}
    for each receiving product j != k:
        D_jk <- (s_j^{+k} - s_j) / lost_k
    set D_kk <- -1
```

The homogeneous logit is estimated on the same data. Its likelihood is easier because it does not integrate over tastes. The comparison is useful because the homogeneous model can fit mean shares while still forcing diversion to follow existing market shares.

## Results

The mixed-logit fit tracks the product shares closely. The homogeneous logit also fits average shares reasonably well, so share fit alone is not enough to show why heterogeneity matters.

<img src="figures/choice-fit.png" alt="Observed and fitted product shares" width="80%">

The profiled likelihood is lowest near positive taste dispersion. Setting the standard deviations close to zero collapses the model toward plain logit and loses the substitution patterns generated by heterogeneous consumers.

<img src="figures/heterogeneity-profile.png" alt="Profile likelihood over taste heterogeneity" width="80%">

When Premium is removed, the two models make different recapture predictions. Plain logit reallocates the lost demand according to average shares. Mixed logit moves more demand toward products that appeal to similar simulated consumers.

<img src="figures/substitution-patterns.png" alt="Diversion after removing one product" width="80%">

The price-substitution matrix asks where demand goes when one product becomes 0.10 price units more expensive. Each column is the product whose price is raised. Each off-diagonal entry is the share gain for the receiving product divided by the shocked product's lost share.

**Plain and mixed logit price substitution matrix**

| Model       | Receiving product   |   Price up: Budget |   Price up: Mainstream |   Price up: Premium |   Price up: Niche |
|:------------|:--------------------|-------------------:|-----------------------:|--------------------:|------------------:|
| Plain logit | Budget              |            -1      |                 0.3788 |              0.3716 |            0.3481 |
| Plain logit | Mainstream          |             0.368  |                -1      |              0.3508 |            0.337  |
| Plain logit | Premium             |             0.3461 |                 0.3364 |             -1      |            0.3149 |
| Plain logit | Niche               |             0.2859 |                 0.2848 |              0.2776 |           -1      |
| Mixed logit | Budget              |            -1      |                 0.4646 |              0.3964 |            0.391  |
| Mixed logit | Mainstream          |             0.426  |                -1      |              0.3425 |            0.3293 |
| Mixed logit | Premium             |             0.2985 |                 0.2821 |             -1      |            0.2798 |
| Mixed logit | Niche               |             0.2755 |                 0.2532 |              0.2611 |           -1      |

The parameter and fit tables separate two diagnostics. The parameter table checks known-truth recovery. The share table checks whether the fitted model matches observed product choices.

**Known-truth parameter recovery**

| Parameter          |   True | Plain logit   |   Mixed logit |   Mixed error |
|:-------------------|-------:|:--------------|--------------:|--------------:|
| Mean price taste   |  -1    | -1.0847       |       -1.0798 |       -0.0798 |
| Mean quality taste |   1.1  | 1.1963        |        1.2668 |        0.1668 |
| SD price taste     |   0.36 | not estimated |        0.5746 |        0.2146 |
| SD quality taste   |   0.55 | not estimated |        0.3119 |       -0.2381 |

**Observed and fitted product shares**

| Product    |   Observed share |   True probability |   Plain logit |   Mixed logit |   Mixed error |
|:-----------|-----------------:|-------------------:|--------------:|--------------:|--------------:|
| Budget     |           0.294  |             0.2983 |        0.2801 |        0.2928 |       -0.0012 |
| Mainstream |           0.2447 |             0.2522 |        0.2649 |        0.25   |        0.0053 |
| Premium    |           0.2593 |             0.2512 |        0.2482 |        0.2609 |        0.0016 |
| Niche      |           0.202  |             0.1983 |        0.2068 |        0.1963 |       -0.0057 |

## Takeaway

Mixed logit is a simulation estimator because choice probabilities require an integral over unobserved tastes. Fixed draws turn that integral into a smooth sample average. The payoff is economic: aggregate substitution is no longer forced to satisfy IIA, even though each simulated consumer still has a logit choice rule conditional on tastes.

**Computational tricks used here**

- Fixed draws make $\widehat P_{ij}(\theta)$ deterministic at each trial $\theta$.
- Log standard deviations keep $\sigma_\alpha$ and $\sigma_\beta$ positive.
- L-BFGS-B bounds keep the optimizer out of irrelevant parameter regions.
- The probability floor $\eta$ prevents one simulated zero from breaking the log likelihood.
- The finite price step $\Delta p$ turns substitution into a readable recapture matrix.

## References

- [Train, K. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.](https://eml.berkeley.edu/books/choice2.html)
- [McFadden, D., and Train, K. (2000). Mixed MNL Models for Discrete Response. *Journal of Applied Econometrics*, 15(5), 447-470.](https://doi.org/10.1002/1099-1255%28200009/10%2915:5%3C447::AID-JAE570%3E3.0.CO;2-1)
