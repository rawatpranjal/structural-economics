# McCall Job Search and the Reservation Wage

## Overview

Before McCall (1970), economists modeled wage search as static: a worker drew all offers simultaneously and picked the best. McCall asked what happens when offers arrive sequentially and the worker cannot recall rejected ones. The answer is a *reservation wage* that fully summarizes the optimal policy: accept any offer at or above it, reject all others, and return to search next period.

The object is the reservation wage as a function of the offer distribution, the discount factor, and the unemployment benefit. The computation needs only one continuation value per iteration because rejection returns the same scalar regardless of the current offer.

## Read before

- [Optimal growth model](../optimal-growth/README.md)
- [Consumption-savings under income risk](../consumption-savings/README.md)
- [Root-finding methods](../../numerical-methods/root-finding/README.md)

## Equations

Let $`W`$ be a wage offer with distribution $`F`$.
The current offer is $`w`$.
The worker discounts at $`\beta\in(0,1)`$.

Accepting gives a permanent income stream:

```math
A(w)=\frac{w}{1-\beta}.
```

Rejecting pays $`b`$ today.
Tomorrow the worker draws again.
Because today's rejected offer is gone, rejection has one value:

```math
C=b+\beta\mathbb{E}_{F}[V(W')],
```

where $`W'\sim F`$ is the next-period wage draw.
The Bellman equation compares the two values:

```math
V(w)=\max\left(\frac{w}{1-\beta}, C\right).
```

Since $`A(w)`$ rises with $`w`$ and $`C`$ is constant, the policy is a cutoff.
At indifference, $`A(w^{\ast})=C`$:

```math
\frac{w^{\ast}}{1-\beta}=b+\beta\mathbb{E}_{F}[V(W')].
```

Substitution gives a scalar fixed point in $`w^{\ast}`$:

```math
w^{\ast}=(1-\beta) b+\beta\mathbb{E}_{F}[\max(W',w^{\ast})].
```

The equation shows the two main margins.
Higher $`b`$ raises the outside option.
Higher $`\beta`$ raises the value of waiting.
The right tail of $`F`$ matters through the expectation.

## Worked Numerical Example

Take $`W \in \{2, 5, 10\}`$ each with probability $`1/3`$, $`\beta = 0.9`$, and $`b = 1`$.

Conjecture $`w^{\ast} \in (5, 10)`$, so only $`W = 10`$ is accepted. Under that conjecture:

```math
\mathbb{E}_{F}[\max(W, w^{\ast})]
  = \tfrac{1}{3} w^{\ast} + \tfrac{1}{3} w^{\ast} + \tfrac{1}{3}(10)
  = \tfrac{2}{3} w^{\ast} + \tfrac{10}{3}.
```

Substitute into the *fixed-point* equation:

```math
w^{\ast} = (0.1)(1) + (0.9)\!\left[\tfrac{2}{3} w^{\ast} + \tfrac{10}{3}\right]
         = 0.1 + 0.6\, w^{\ast} + 3.0.
```

Collect terms:

```math
0.4\, w^{\ast} = 3.1 \implies \boxed{w^{\ast} = 7.75}.
```

Since $`5 < 7.75 < 10`$, the conjecture holds.

The mean offer is $`\mathbb{E}[W] = 17/3 \approx 5.67`$; the reservation wage 7.75 sits above it. Acceptance probability is $`1/3`$, so expected unemployment duration is 3 periods. The right tail (the offer of 10) makes waiting worthwhile even though two of the three offers are rejected.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Discount factor $`\beta`$ | 0.95 | Location $`\mu`$ | 0.0 |
| Flow benefit $`b`$ | 1.0 | Scale $`\sigma`$ | 1.0 |
| Wage law | $`\log W\sim N(\mu,\sigma^2)`$ | Median offer $`e^{\mu}`$ | 1.0000 |
| Wage grid (equiprobable bins) | 50 | Mean offer $`\mathbb{E}[W]`$ | 1.6487 |
| VFI tolerance (sup-norm) | 1e-08 | Continuous benchmark | exact lognormal moments |

## Solution Method

The *Bellman operator* $`T`$ maps a candidate value function $`V`$ to a new one. Repeated application converges to the fixed point because the update is almost scalar. Each sweep computes one expectation against the offer distribution. The lognormal offer law is replaced by $`n_w=50`$ equiprobable bins, each represented by its conditional mean inside the bin.

```
          wages, probs, beta, b
                    |
                    v
    +---------- VFI loop ----------+
    |  V_k --> [ Bellman op ] --> V_{k+1}  |
    +-------- err >= tol: repeat ----------+
                    |
                 err < tol
                    v
              V*(w), reservation wage w*
```

```python
# V(w) = max(w / (1 - beta), C); C = b + beta * E[V(W')]
def solve_mccall(beta, b, wages, probs, tol=1e-8):
    accept_values = wages / (1.0 - beta)   # A(w) = w / (1 - beta)
    value = accept_values.copy()
    for _ in range(1_000):
        # C = b + beta * sum_i p_i V_i; one scalar per sweep
        continuation_value = b + beta * np.dot(probs, value)
        new_value = np.maximum(accept_values, continuation_value)
        if np.max(np.abs(new_value - value)) < tol:
            break
        value = new_value
    # invert C = w* / (1 - beta) to recover w*
    reservation_wage = (1.0 - beta) * (b + beta * np.dot(probs, value))
    return value, reservation_wage
```

At baseline, VFI converges in 178 iterations to a sup-norm error of 9.84e-09. The grid cutoff is 4.7054. The continuous lognormal benchmark (solved by Brent's method on the scalar residual) gives 4.7055. Absolute grid error is 9.1e-05.

## Results

The threshold panel shows the accept and reject values crossing at the cutoff. The shaded region marks acceptable offers. The grid and continuous cutoffs nearly coincide. The density panel shows why the cutoff exceeds the mean: the lognormal right tail makes waiting valuable even though most offers are rejected. The convergence panel shows geometric decay of the sup-norm error. Acceptance probability falls steeply as patience rises, because a more patient worker holds out for rarer high offers.

![VFI result: threshold logic, offer density, convergence, and acceptance by patience](figures/vfi-result.png)

Patience raises the reservation wage. As $`\beta`$ approaches one, the worker values future draws more and the cutoff moves into the right tail. The grid solution stays close to the continuous benchmark. The gap widens only at high $`\beta`$. Benefits raise the cutoff less than one-for-one. Higher $`b`$ makes rejection less costly, but search still has upside from the right tail.

![Comparative statics: reservation wage by patience and by benefit](figures/comparative-statics.png)

The table separates the benefit and patience margins. Both higher $`b`$ and higher $`\beta`$ raise the cutoff, lower acceptance rates, and lengthen expected duration. Grid error stays small at moderate $`\beta`$ and grows at high $`\beta`$ because the cutoff sits deeper in the tail.

### Reservation wage diagnostics

|   Discount ($`\beta`$) |   b |   w* grid |   w* cont. |   grid gap |   Accept % (cont.) |   E[duration] |   VFI iter. |
|-------:|----:|----------:|-----------:|-----------:|-------------------:|--------------:|------------:|
|   0.9  | 0.5 |    3.3118 |     3.3126 |    -0.0007 |               11.6 |           8.7 |          86 |
|   0.9  | 1   |    3.5654 |     3.5656 |    -0.0002 |               10.2 |           9.8 |          95 |
|   0.9  | 2   |    4.1194 |     4.1196 |    -0.0002 |                7.8 |          12.8 |         106 |
|   0.95 | 0.5 |    4.4718 |     4.4794 |    -0.0076 |                6.7 |          15   |         176 |
|   0.95 | 1   |    4.7054 |     4.7055 |    -0.0001 |                6.1 |          16.5 |         178 |
|   0.95 | 2   |    5.1727 |     5.1946 |    -0.0219 |                5   |          20.1 |         181 |
|   0.99 | 0.5 |    8.1646 |     8.1789 |    -0.0143 |                1.8 |          56.2 |         694 |
|   0.99 | 1   |    8.3324 |     8.3631 |    -0.0308 |                1.7 |          59.4 |         696 |
|   0.99 | 2   |    8.6679 |     8.7514 |    -0.0834 |                1.5 |          66.5 |         699 |

## Takeaway

*Sequential search* turns unemployment duration into a reservation wage. Workers accept only offers that beat the price of waiting, so most spells end when a right-tail offer arrives. McCall's surprise was that the reservation wage exceeds the mean offer: a patient worker rationally rejects the majority of draws. The paper established that unemployment duration reflects optimal choice under uncertainty, not passivity. That insight anchored the subsequent equilibrium search literature, from Mortensen and Pissarides on matching to models of posted wages and directed search.

## See also

- [Huggett incomplete-markets model](../../heterogeneous-agents/huggett-incomplete-markets/README.md)
- [Mortensen-Pissarides matching model](../../search-matching/mortensen-pissarides/README.md)
- [Consumption-savings under income risk](../consumption-savings/README.md)

## References

- McCall, J.J. (1970). Economics of Information and Job Search. *Quarterly Journal of Economics*, 84(1), 113-126.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 6.
- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.
- Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.
