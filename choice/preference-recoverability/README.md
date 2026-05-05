# Preference Bounds from Revealed Choices

> Afriat numbers, recovered upper contours, and the limits of finite choice data.

## Overview

Passing GARP is only the first revealed-preference question. Once the choices are rationalizable, the natural next question is what they say about the preference ordering itself. The answer is deliberately set-valued: finite budgets do not identify a unique demand system, but they do restrict the utility functions that could have generated the data.

This tutorial starts from the same object as [Afriat's revealed-preference test](../revealed-preference-afriat/): a finite sample of prices $p_t$ and chosen bundles $x_t$. Instead of stopping at pass/fail rationalizability, it constructs Afriat numbers and uses them to draw a concave utility index that exactly rationalizes the observed choices. Because the synthetic data come from a Cobb-Douglas consumer, the true indifference curve is available as a diagnostic; the Afriat construction does not use that functional form.

## Equations

There are two goods and $T$ observations. At observation $t$, prices are
$p_t=(p_{1t},p_{2t})\in\mathbb{R}_{++}^{2}$, the chosen bundle is
$x_t=(x_{1t},x_{2t})\in\mathbb{R}_{+}^{2}$, and expenditure is
$m_t=p_t\cdot x_t$.

Afriat's inequalities ask for numbers $u_t$ and $\lambda_t>0$ such that
$$
u_i-u_j \leq \lambda_j p_j\cdot(x_i-x_j)
\qquad \text{for all } i,j=1,\ldots,T .
$$
When these inequalities are feasible, one rationalizing utility index is
$$
\widehat U(y)=\min_{j=1,\ldots,T}
\left[u_j+\lambda_j p_j\cdot(y-x_j)\right].
$$
The minimum of affine supporting functions is concave, monotone when prices and
$\lambda_j$ are positive, and satisfies $\widehat U(x_t)=u_t$ at the observed
choices.

For a target observation $k$, the recovered upper-contour set is
$$
\widehat U(y)\geq u_k .
$$
Writing $y=(y_1,y_2)$, its lower boundary can be computed pointwise:
$$
y_2(y_1)=
\max_{j=1,\ldots,T}
\left[
x_{2j}
+
\frac{
u_k-u_j-\lambda_j p_{1j}(y_1-x_{1j})
}{
\lambda_j p_{2j}
}
\right].
$$
The data-generating benchmark, used only for comparison, is
$$
U^0(x)=x_1^{\alpha}x_2^{1-\alpha},\qquad \alpha=0.60 .
$$

## Model Setup

| Object | Value | Role in the exercise |
|---|---:|---|
| Observations $T$ | 18 | Budget-choice pairs used for recovery |
| Goods | 2 | Keeps the recovered contour visible |
| True $\alpha$ | 0.60 | Cobb-Douglas benchmark, held out from recovery |
| Income range | [5.07, 13.33] | Shifts observed budget lines |
| Price range | [0.57, 1.96] | Rotates observed budget lines |
| GARP violations | 0 | Required before recovery is meaningful |
| Max Afriat residual | 1.30e-15 | Numerical feasibility check |
| Target observation | 7 | Bundle whose contour is recovered |

## Solution Method

The computation has two pieces. First, the observed choices must pass the same revealed-preference consistency check used in the preceding tutorial. Second, the Afriat inequalities are solved as linear difference constraints. Utility levels are ordinal, so the report fixes their sample mean at one; the expenditure normalization $\lambda_t=1/m_t$ fixes the utility units for this generated sample.

```text
Algorithm: Afriat contour recovery
Input: budgets (p_t, x_t) for t=1,...,T and target observation k
Output: recovered utility index U_hat and contour through x_k

1. Build R[i,j] = 1 if p_i . x_i >= p_i . x_j.
2. Compute the transitive closure R_star and verify that GARP has no strict cycle.
3. Set lambda_t = 1 / (p_t . x_t) to choose utility units.
4. Solve for u_t subject to
       u_i - u_j <= lambda_j p_j . (x_i - x_j) for every pair (i,j),
       average_t u_t = 1, and u_t >= 0.
5. Define U_hat(y) = min_j [u_j + lambda_j p_j . (y - x_j)].
6. For a grid of y_1 values, compute the smallest y_2 that satisfies
       U_hat((y_1,y_2)) >= u_k.
```

A more general implementation could leave the $\lambda_t$ values as additional linear-programming variables. Here the fixed normalization keeps the tutorial focused on recoverability rather than on the many equivalent ordinal scalings of the same revealed-preference information.

## Results

The raw data are only budgets and chosen bundles. Price variation rotates the budget lines, while income shifts them out. The highlighted bundle is the one whose upper-contour boundary is recovered below. The computation treats the Cobb-Douglas origin of the choices as unknown.

<img src="figures/budget-lines.png" alt="Observed budget lines and chosen bundles with the target observation highlighted" width="80%">

The recovered boundary is not a parametric estimate of Cobb-Douglas demand. It is one concave utility contour that passes through the target choice and rationalizes every observed bundle. The dashed curve is the true contour from the simulation and is shown only because this example has a known benchmark. On the plotted overlap, the median recovered-to-true $x_2$ ratio is **0.86**, and the largest absolute contour gap is **9.89** units of good 2.

<img src="figures/indifference-curve.png" alt="Recovered Afriat contour and the held-out Cobb-Douglas contour" width="80%">

The Afriat numbers are not structural parameters. They are a finite-data certificate: $u_t$ ranks observed bundles in a way that respects all budget comparisons, and $\lambda_t$ gives the supporting hyperplane slope in utility units. Since the sample is simulated, we can check the ordinal relationship against the true utility index; the correlation is **0.973**.

<img src="figures/afriat-numbers.png" alt="Afriat utility levels and marginal utility normalizations" width="80%">

The last column checks that the recovered utility evaluates to $u_t$ at each observed bundle. The true normalized utility column is a diagnostic for this simulation, not an input to the recovery algorithm.

**Afriat numbers and fit diagnostics**

|   Observation |   Expenditure |    u_t |   lambda_t |   True U normalized |   Fit error |
|--------------:|--------------:|-------:|-----------:|--------------------:|------------:|
|             1 |          6.3  | 0.3738 |     0.1587 |              0.5457 |    0        |
|             2 |          9.76 | 0.7013 |     0.1025 |              0.7205 |    0        |
|             3 |          7.27 | 1.0046 |     0.1376 |              0.9025 |    0        |
|             4 |         11.7  | 0.9263 |     0.0855 |              0.8797 |    0        |
|             5 |          9.37 | 1.4329 |     0.1067 |              1.3645 |    0        |
|             6 |         13.33 | 1.3368 |     0.075  |              1.2453 |    0        |
|             7 |         12    | 1.0048 |     0.0833 |              0.9537 |    0        |
|             8 |          8.12 | 1.0568 |     0.1231 |              0.9896 |    0        |
|             9 |         13.32 | 1.6398 |     0.0751 |              1.7191 |    0        |
|            10 |         13.05 | 1.0747 |     0.0766 |              1.0049 |    0        |
|            11 |          8.87 | 0.8358 |     0.1127 |              0.8122 |    0        |
|            12 |          7.88 | 0.3015 |     0.1269 |              0.5145 |    0        |
|            13 |         11.82 | 1.2694 |     0.0846 |              1.1898 |    0        |
|            14 |          6.4  | 0.9517 |     0.1563 |              0.8971 |    0        |
|            15 |          7    | 0.9713 |     0.1429 |              0.8885 |   -1.33e-15 |
|            16 |          5.07 | 0.0113 |     0.1971 |              0.3625 |    0        |
|            17 |         12.87 | 1.6214 |     0.0777 |              1.5794 |    0        |
|            18 |         11.65 | 1.4857 |     0.0858 |              1.4304 |    0        |

## Takeaway

Preference recoverability is useful precisely because it does not pretend that finite data identify a unique utility function. Afriat numbers give a concrete rationalizing utility index and a way to draw local upper-contour restrictions from observed budgets. The Cobb-Douglas overlay shows the right interpretation: the data discipline the preference ordering where budgets create support, while regions with little price variation remain weakly pinned down.

If the data fail GARP, this exercise should stop and the next question is which observations break rationalizability; see the [money pump index](../money-pump-index/) and [Houtman-Maks rational subsets](../houtman-maks-rational-subsets/) tutorials. If the data pass, Afriat recovery gives the nonparametric object that a later parametric demand model has to respect.

## References

- Afriat, S. N. (1967). The Construction of Utility Functions from Expenditure Data. *International Economic Review*, 8(1), 67-77.
- Varian, H. R. (1982). The Nonparametric Approach to Demand Analysis. *Econometrica*, 50(4), 945-973.
- Varian, H. R. (2006). Revealed Preference. In M. Szenberg et al. (Eds.), *Samuelsonian Economics and the Twenty-First Century*. Oxford University Press.
