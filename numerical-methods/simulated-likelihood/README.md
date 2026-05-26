# Simulated Maximum Likelihood, Common Random Numbers, and Halton Sequences

## Overview

Many structural likelihoods are integrals with no closed form. The choice probability in a mixed logit model averages a logit kernel over a distribution of random tastes. The market share in a random-coefficients demand model averages over a distribution of consumer types. The likelihood under a stochastic transformation model averages over a distribution of shocks. In each case the analyst observes data but does not observe the random object that the likelihood integrates out.

Lerman and Manski (1981) proposed replacing the integral with a finite average over simulation draws. McFadden (1989) gave the method its asymptotic foundation under the name *method of simulated moments* and asked how large the draw count must be before the estimator is consistent and efficient. The practical gap the paper was writing against was that existing estimators either required closed-form integrals (logit) or expensive numerical quadrature (probit with many random coefficients). Simulation offered a scalable alternative, but practitioners did not yet know which draw schemes were worth the effort.

The standard response is to hold the draws fixed across parameter values. Holding the draws fixed is the trick called common random numbers (CRN). It is essential, because resampling at every candidate parameter would add Monte Carlo noise to the objective and turn a smooth optimization problem into a noisy one.

Two questions remain. How big does the draw count need to be before the simulated estimator is close to the true integrated object? And do quasi-random draws (low-discrepancy sequences that fill space more uniformly than pseudo-random points) beat pseudo-random ones at the same draw count? This prelim runs a Monte Carlo experiment on a mixed binary logit panel and answers both. The same machinery is the integral-approximation engine in [`choice/mixed-logit-simulation/`](../../choice/mixed-logit-simulation/), in the simulated shares of [`industrial-organization/blp-random-coefficients/`](../../industrial-organization/blp-random-coefficients/), in the latent draws of [`structural-econometrics/rum-choice-networks/`](../../structural-econometrics/rum-choice-networks/), and in the smooth outer objective of [`structural-econometrics/adversarial-estimation/`](../../structural-econometrics/adversarial-estimation/).

## Read before

- [Mixed logit simulation](../../choice/mixed-logit-simulation/README.md)
- [GMM foundations](../../structural-econometrics/gmm-foundations/README.md)

## Equations

We need a notation that separates the parameter we estimate from the latent variable we integrate out. Let $`\theta`$ be the parameter of interest, let $`\xi`$ be a latent variable distributed according to a known density $`F`$, and let $`f(\theta; \xi)`$ be a tractable kernel (often a likelihood contribution or a market share). The *exact integrated object* is:

```math
P(\theta) = \int f(\theta; \xi)  dF(\xi).
```

In the running example, $`\theta = \sigma`$ is the dispersion of a random coefficient, $`\xi \sim N(0, 1)`$ is a standardised taste shock, and $`f(\theta; \xi)`$ is the logit-kernel product of one individual's $`T`$ choice probabilities given the latent draw. The integral has no closed form because the logit probability is a nonlinear function of $`\sigma \xi`$.

We need an estimator for $`P(\theta)`$ that uses a finite number of draws and avoids re-randomising as the optimiser moves. Draw $`R`$ values $`\xi_1, \ldots, \xi_R`$ once from $`F`$ and replace the integral with the sample average evaluated at the same draws across every candidate $`\theta`$:

```math
\widehat P(\theta) = \frac{1}{R} \sum_{r=1}^{R} f(\theta; \xi_r).
```

The draws $`\{\xi_r\}_{r=1}^{R}`$ are fixed across all candidate $`\theta`$. This is the simulated likelihood (when $`f`$ is a likelihood contribution) or the simulated probability (when $`f`$ is a choice probability). The "fixed across $`\theta`$" clause is what makes the estimator usable inside an optimiser: it is the common-random-numbers (CRN) property.

We need to be explicit about why CRN matters. If a fresh set of draws $`\xi^{(\theta)}_r`$ were drawn for every $`\theta`$, then $`\widehat P(\theta)`$ would inherit the simulation noise as an extra random term that varies with $`\theta`$. The optimiser's gradient information would be polluted by that noise, and the objective surface would become non-smooth. With $`\{\xi_r\}`$ held fixed, $`\widehat P(\theta)`$ is a deterministic function of $`\theta`$ given the draws. Its smoothness inherits the smoothness of $`f(\theta; \xi)`$. Formally, if $`f(\cdot; \xi)`$ is differentiable, so is $`\widehat P(\cdot)`$ with the same Lipschitz behaviour.

We need a way to generate draws that fill the integration region more uniformly than pseudo-random points. Quasi-random sequences (low-discrepancy sequences) are deterministic point sets engineered for that purpose. The Halton sequence in prime base $`b`$ is generated by writing each integer $`k`$ in base $`b`$ and reversing its digits around the radix point:

```math
\phi_b(k) = \sum_{i = 0}^{\infty} \frac{a_i(k)}{b^{i+1}},
```

```math
k = \sum_{i = 0}^{\infty} a_i(k)  b^{i}, \quad a_i(k) \in \{0, 1, \ldots, b-1\}.
```

The first few Halton-base-2 points are $`1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, \ldots`$. They are not random; they are arranged so that successive points fall in the largest empty gap. Pushing the $`\phi_b(k)`$ values through the inverse normal cumulative $`\Phi^{-1}`$ converts them into standard-normal draws that retain the space-filling property. Scrambled Sobol sequences play the same role with a more sophisticated digit permutation.

For a mixed-logit choice probability with $`T`$ panel observations per individual and a single random coefficient, the kernel $`f`$ has the closed form $`f(\sigma; \xi) = \prod_{t = 1}^{T} L(y_{it}; (\mu + \sigma \xi)  x_{it})`$, where $`L(y; v) = \exp(yv) / (1 + \exp(v))`$ is the binary-logit likelihood. Plugging this into $`\widehat P(\sigma)`$ and taking logs gives the simulated log-likelihood that the estimator maximises.

## Worked Numerical Example

Take a probit-style binary-choice kernel to keep arithmetic transparent: latent $`y_i^{\ast} = \theta x_i + \xi_i`$ with $`\xi_i \sim N(0,1)`$, observed $`y_i = \mathbf{1}\lbrace y_i^{\ast} > 0 \rbrace`$, kernel $`f(\theta; \xi_i) = \mathbf{1}\lbrace \theta x_i + \xi_i > 0 \rbrace`$ for $`y_i = 1`$ and its complement for $`y_i = 0`$. Two observations $`(x_1, y_1) = (0.5, 1)`$, $`(x_2, y_2) = (-0.3, 0)`$, parameter trial $`\theta = 1`$, and $`R = 3`$ fixed draws per observation.

The exact integrated object is $`P_i(\theta) = \Phi(\theta x_i)`$ when $`y_i = 1`$ and $`1 - \Phi(\theta x_i) = \Phi(-\theta x_i)`$ when $`y_i = 0`$, giving exact log-likelihood:

```math
\log L(1) = \log \Phi(0.5) + \log \Phi(0.3) \approx \log 0.6915 + \log 0.6179 \approx -0.369 + (-0.481) = -0.850.
```

Now form the *simulated estimator* $`\widehat P_i(\theta) = (1/R) \sum_r f(\theta; \xi_{i,r})`$ with fixed pseudo-random draws. For observation 1, draws $`\xi_{1,r} = (-0.2, 0.1, -0.8)`$ give latent values $`\theta x_1 + \xi_{1,r} = (0.3, 0.6, -0.3)`$, indicators $`(1, 1, 0)`$, so

```math
\widehat P_1(1) = \frac{1}{3}(1 + 1 + 0) = \frac{2}{3}.
```

For observation 2, draws $`\xi_{2,r} = (0.4, -0.1, 0.6)`$ give latent values $`\theta x_2 + \xi_{2,r} = (0.1, -0.4, 0.3)`$, indicators $`(1, 0, 1)`$ for $`y = 1`$, so $`y_2 = 0`$ matches with simulated probability

```math
\widehat P_2(1) = \frac{1}{3}(0 + 1 + 0) = \frac{1}{3}.
```

Sum the logs to get the simulated log-likelihood:

```math
\log \widehat L(1) = \log \tfrac{2}{3} + \log \tfrac{1}{3} \approx -0.405 + (-1.099) = \boxed{-1.504}.
```

The simulated value $`-1.504`$ differs from the exact $`-0.850`$ because $`R = 3`$ is small and the indicator kernel is discontinuous, so each $`\widehat P_i`$ is a coarse three-step staircase. The CRN property holds: the same draws $`(-0.2, 0.1, -0.8)`$ and $`(0.4, -0.1, 0.6)`$ are reused at every candidate $`\theta`$, so $`\log \widehat L(\theta)`$ is a deterministic function of $`\theta`$. Smoother kernels (logit, GHK) and larger $`R`$ shrink the gap between $`\log \widehat L`$ and $`\log L`$ at the rate displayed in the bias panel.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Parameter of interest $`\theta`$ | generic | Latent variable $`\xi`$ | integration variable |
| Latent density $`F`$ | distribution of $`\xi`$ | Kernel $`f(\theta; \xi)`$ | tractable function |
| True integrated object $`P(\theta)`$ | $`\int f  dF`$ | Simulated estimator $`\widehat P(\theta)`$ | $`(1/R) \sum_r f(\theta; \xi_r)`$, fixed draws |
| Number of draws $`R`$ | 50, 200, 1000 | Halton radical inverse $`\phi_b(k)`$ | base-$`b`$ digit-reversal |
| Individuals $`N`$ | 100 | Choices per individual $`T`$ | 5 |
| Population mean $`\mu`$ | 1.0 (known) | Population s.d. $`\sigma`$ | 0.8 (estimated) |
| Monte Carlo trials | 200 | Inverse normal cdf $`\Phi^{-1}`$ | maps $(0,1)$ to $`N(0,1)`$ |

## Solution Method

The *SML optimizer* iterates over candidate $`\sigma`$, evaluates the simulated log-likelihood at fixed draws, and returns the bounded scalar minimizer of the negative objective. Draws are built once per scheme and draw count and reused across all Monte Carlo trials and all candidate $`\sigma`$ values (CRN). The logsumexp trick is applied along the draw axis to prevent underflow when the product over $`T`$ choice occasions shrinks the kernel.

```
          N, T, R, scheme, MC trial count
                       |
                       v
    +------ draw construction (once per scheme, R) ------+
    |  pseudo  -->  [ pseudo draws ]                      |
    |  halton  -->  [ Halton base-2 ]  -->  z (fixed)    |
    |  sobol   -->  [ scrambled Sobol ]                   |
    +-----------------------------------------------------+
                       |
                       v
    +------ MC loop (one dataset per trial) -------------+
    |  trial  -->  [ simulate data ]  -->  y, x          |
    |                                                     |
    |  +---- SML optimizer (per scheme, R) -----------+  |
    |  |  sigma  -->  [ evaluate L_S ]  -->  L_S      |  |
    |  |  L_S    -->  [ Brent step ]    -->  sigma_new |  |
    |  +---- not converged: repeat --------------------+  |
    |                    |                                |
    |                converged                            |
    |                    v                                |
    |              sigma_hat (scheme, R, trial)           |
    +------ next trial: repeat ---------------------------+
                       |
                   all trials done
                       v
              bias, SD of sigma_hat by (scheme, R)
```

```python
def simulated_log_likelihood(sigma, y, x, mu, z):
    # beta_r = mu + sigma * z_r  for each draw r
    beta = mu + sigma * z                                # (R,)
    lin = beta[:, None, None] * x[None, :, :]           # (R, N, T)
    log_p1 = -np.logaddexp(0.0, -lin)                   # log logistic(lin)
    log_p0 = -np.logaddexp(0.0,  lin)                   # log(1 - logistic(lin))
    log_lik_rit = np.where(y[None, :, :] == 1, log_p1, log_p0)
    log_lik_ri  = log_lik_rit.sum(axis=2)               # sum over T: (R, N)
    # logsumexp over draws minus log R gives log mean_r
    log_mean_i = logsumexp(log_lik_ri, axis=0) - np.log(len(z))
    return float(np.mean(np.clip(log_mean_i, np.log(CLIP_LOW), 0.0)))
```

The Brent minimizer reaches the bounded optimum on $`\sigma \in (0.01, 5.0)`$ in a handful of function evaluations per trial.

## Results

The Monte Carlo experiment runs 200 trials at three draw counts ($`R = 50, 200, 1000`$) under three schemes (pseudo-random, Halton base-2, scrambled Sobol). The panel is $`N = 100`$ individuals with $`T = 5`$ repeated binary choices each. True $`\sigma = 0.8`$.

The first figure shows the *sampling distribution* of $`\widehat\sigma`$ at $`R = 200`$.

<img src="figures/sampling-distributions.png" alt="Boxplots of sigma_hat at R=200 for three draw schemes, with horizontal line at true sigma=0.8" width="80%">

All three schemes are roughly unbiased: the median sits within $`\pm 0.05`$ of the true value $`0.8`$. The interquartile spread is wide because the dominant noise source is the data (100 individuals with 5 choices each), not the simulation. The boxplot is the right place to read the gross magnitude of estimation uncertainty; the bias-variance figure isolates the simulation contribution.

The second figure plots the absolute bias and the standard deviation of $`\widehat\sigma`$ against $`R`$.

<img src="figures/bias-variance-vs-R.png" alt="Absolute bias and standard deviation of sigma_hat versus draw count R, for three draw schemes, on log axes" width="90%">

The left panel separates the schemes by bias. At $`R = 1000`$, the pseudo-random bias is $`0.021`$, the Halton bias is $`0.010`$, and the scrambled-Sobol bias is $`0.009`$. The quasi-random schemes cut bias roughly in half at the same draw count. Because the same Monte Carlo dataset is used across schemes within each trial, the across-scheme comparison is a paired difference rather than two independent samples, so the precision of the gap is much higher than the standard deviation of each scheme's bias would suggest. The right panel shows that standard deviation is essentially flat across $`R`$ and across schemes, around $`0.24`$. With $`N`$ and $`T`$ fixed, the data variance dominates the simulation variance; in this regime, larger $`R`$ helps bias but does not visibly help variance.

The third figure illustrates why quasi-random schemes have an advantage in the first place.

<img src="figures/halton-vs-pseudo-cloud.png" alt="Side-by-side scatter: pseudo-random and Halton base 2 and 3 draws in the unit square, R=200 points each" width="90%">

The pseudo-random panel has visible clumps and gaps. The Halton panel covers the unit square evenly. In higher dimensions, the disparity grows: pseudo-random draws fail to cover the integration region until $`R`$ is large, while Halton and Sobol stay close to uniform. That uniform coverage is what reduces the bias of the simulated estimator at small $`R`$.

Two practical caveats. First, antithetic variates (pair each draw $`\xi_r`$ with its sign-flipped twin $`-\xi_r`$) can match Halton and Sobol on bias for symmetric integrating densities at near-zero implementation cost. Second, scrambled Sobol below its preferred power-of-two sample size can perform worse than Halton; here at $`R = 50`$ Sobol's bias is $`0.047`$ against Halton's $`0.017`$, then both fall and converge at $`R = 200`$ and $`R = 1000`$.

### Estimation diagnostics

| Scheme | R | Bias | SD |
|:---|---:|---:|---:|
| Pseudo-random | 50 | ~0.052 | ~0.24 |
| Pseudo-random | 200 | ~0.032 | ~0.24 |
| Pseudo-random | 1000 | ~0.021 | ~0.24 |
| Halton (base 2) | 50 | ~0.017 | ~0.24 |
| Halton (base 2) | 200 | ~0.013 | ~0.24 |
| Halton (base 2) | 1000 | ~0.010 | ~0.24 |
| Scrambled Sobol | 50 | ~0.047 | ~0.24 |
| Scrambled Sobol | 200 | ~0.013 | ~0.24 |
| Scrambled Sobol | 1000 | ~0.009 | ~0.24 |

## Takeaway

*Simulated maximum likelihood* replaces an intractable integral with a finite average over draws that are held fixed across parameter values. The fixed draws are the common-random-numbers property; they make the simulated objective a deterministic, smooth function of the parameter that a standard optimiser can climb. Quasi-random draws (Halton, scrambled Sobol) reduce the bias of the simulated estimator at the same $`R`$ by filling the integration region more uniformly than pseudo-random points. McFadden's (1989) quantitative lesson was that the gains from quasi-random draws are real but moderate at practical draw counts, and that the dominant source of estimation uncertainty is the data rather than the simulation. That finding disciplined later applied work on which aspects of the simulation design were worth optimising. The SML framework became the engine for mixed logit (Train), random-coefficients demand (BLP), and every subsequent structural model whose likelihood is an integral over unobserved heterogeneity.

## See also

- [Mixed logit simulation](../../choice/mixed-logit-simulation/README.md)
- [BLP random-coefficients demand](../../industrial-organization/blp-random-coefficients/README.md)
- [GMM foundations](../../structural-econometrics/gmm-foundations/README.md)

## References

- Lerman, S. and Manski, C. (1981). "On the Use of Simulated Frequencies to Approximate Choice Probabilities." In C. Manski and D. McFadden (eds.), *Structural Analysis of Discrete Data with Econometric Applications*. MIT Press, 305-319. First proposal of simulation-based frequency estimators for discrete choice.
- McFadden, D. (1989). "A Method of Simulated Moments for Estimation of Discrete Response Models without Numerical Integration." *Econometrica*, 57(5), 995-1026. Foundational asymptotic theory for simulation-based estimators.
- Train, K. (2009). *Discrete Choice Methods with Simulation*, 2nd edition. Cambridge University Press, Chapter 9 ("Drawing from Densities"). Pedagogical anchor for SML, CRN, and Halton-vs-pseudo comparisons.
- Bhat, C. R. (2001). "Quasi-random Maximum Simulated Likelihood Estimation of the Mixed Multinomial Logit Model." *Transportation Research Part B*, 35(7), 677-693. Empirical evidence that Halton beats pseudo-random on mixed logit.
- Hess, S., Train, K., and Polak, J. (2006). "On the use of a Modified Latin Hypercube Sampling Method." *Transportation Research Part B*, 40(2), 147-163. Practical comparison of Halton, MLHS, and scrambled Sobol.
