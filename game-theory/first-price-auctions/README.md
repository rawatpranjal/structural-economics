# First-Price Auctions, Bid Shading, and Deviation Checks

> Private-value bidding, the symmetric equilibrium rule, and a direct grid audit.

## Overview

A first-price sealed-bid auction forces each bidder to price its private information. Suppose a bidder values the object at 0.80. A bid close to 0.80 wins often but leaves little surplus. A lower bid gives a larger margin if it wins, but it loses more often to rival bids. Equilibrium bidding balances those two forces for every possible value, not only for this one bidder.

The uniform independent-private-values model lets us write the symmetric Bayesian Nash bid rule in closed form. The computation then has a clear job: take that candidate rule and check it type by type. If rivals use the analytic rule, a grid best response over possible bids should return the same bid, up to grid error. This is the Bayesian-game version of the no-deviation checks in [normal-form games](../normal-form-games/), with private values replacing payoff-table cells.

## Equations

An auction has $n$ risk-neutral bidders. Bidder $i$ observes a private value
$v_i \sim U[0,1]$, independently across bidders, and submits one sealed bid.
The winner pays its own bid. A pure symmetric strategy is an increasing bid
function $b(v)$.

For a general distribution $F$, the symmetric first-price bid rule satisfies

$$
b(v)
= v-\frac{\int_0^v F(t)^{n-1}\,dt}{F(v)^{n-1}},
\qquad v>0,
$$

with $b(0)=0$. With $F(v)=v$ on $[0,1]$, this becomes

$$
b^{\ast}(v)=\frac{n-1}{n}v.
$$

The bid is below value because the winner pays its own bid. To check the
equilibrium restriction, fix a type $v$ and let that bidder deviate to dollar
bid $\hat b$ while opponents use $b^{\ast}$. The rival value threshold beaten by
$\hat b$ is

$$
x(\hat b)=\min\left(\frac{n}{n-1}\hat b,\ 1\right).
$$

The probability of winning is

$$
\Pr(\text{win}\mid \hat b)=x(\hat b)^{n-1},
$$

and expected payoff is

$$
\pi(v,\hat b)=(v-\hat b)x(\hat b)^{n-1}.
$$

Expected revenue under the equilibrium is

$$
R_n = E[b^{\ast}(V_{n:n})]
    = \frac{n-1}{n}E[V_{n:n}]
    = \frac{n-1}{n+1},
$$

which is also $E[V_{n-1:n}]$, the expected second-highest value in the uniform
auction.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Value distribution | $U[0,1]$ | Independent private values |
| Risk preferences | Risk neutral | Payoff is value minus payment when winning |
| Reserve price | None | Every nonnegative bid is admissible |
| Bidder counts | 2, 3, 5, 10 | Comparative statics in competition |
| Deviation grid | 2,001 bids per value | Best-response check for each $v$ |
| Check values | 19 values in [0.05, 0.95] | Types used for residuals |
| Focal deviation plot | $n=3$, $v=0.8$ | Payoff shape for one bidder type |

## Solution Method

The analytic bid function gives the equilibrium benchmark. The numerical work asks whether that benchmark satisfies the type-by-type optimality condition. For each bidder count, the code evaluates the closed-form rule, searches a fine grid of deviations for each checked value, and records the largest distance between the grid best response and the analytic bid.

```text
Algorithm: bid shading and unilateral-deviation audit
Inputs: bidder count n, type grid V, bid grid B(v) on [0,v]
Outputs: exact bid rule b*(v) and max best-response residual Delta_n

1. For each type v in V, set the exact bid b*(v) = ((n-1)/n) v.
2. For each candidate bid bhat in B(v), compute x(bhat)=min{n bhat/(n-1), 1}.
3. Evaluate pi(v,bhat) = (v-bhat) x(bhat)^(n-1).
4. Let BR(v) be the bid on the grid with the highest pi(v,bhat).
5. Report Delta_n = max_v |BR(v)-b*(v)|.
```

This audit approximates the best response against the candidate strategy. It does not estimate a new equilibrium. A small residual means the finite bid grid selects the analytic bid that the Bayesian Nash equilibrium prescribes.

## Results

The bid functions show how competition disciplines shading. With two bidders, the equilibrium bid is one half of value. As the number of rivals rises, a small bid reduction gives up more win probability, so the bidder shades less. The dashed 45-degree line is truthful bidding. First-price bidders stay below it because the payment equals their own bid.

<img src="figures/bid-functions.png" alt="Equilibrium first-price bid functions by bidder count" width="80%">

For a bidder with value 0.8 facing 2 rivals, the payoff curve is single-peaked at the exact bid. Bidding lower raises surplus only when the bidder still wins, while bidding higher buys extra win probability at a higher payment. The red point is the grid best response, and its overlap with the analytic vertical line is the equilibrium check for this type.

<img src="figures/best-response-check.png" alt="Grid best response compared with the exact equilibrium bid" width="80%">

The revenue curve is not a simulation artifact. In the uniform model, the expected first-price winning bid equals the expected second-highest value. The points and dashed benchmark coincide, which is the revenue-equivalence result in this simple risk-neutral environment.

<img src="figures/revenue-by-bidders.png" alt="Expected first-price revenue and second-highest-value benchmark" width="80%">

**Auction Summary**

|   Bidders | Equilibrium bid rule   |   Shading at v=1 |   Expected revenue |
|----------:|:-----------------------|-----------------:|-------------------:|
|         2 | b*(v)=1/2 v            |            0.5   |              0.333 |
|         3 | b*(v)=2/3 v            |            0.333 |              0.5   |
|         5 | b*(v)=4/5 v            |            0.2   |              0.667 |
|        10 | b*(v)=9/10 v           |            0.1   |              0.818 |

The table keeps the model implication separate from the grid diagnostic. Residuals are largest when the exact bid falls between adjacent grid bids.

**Best-Response Check**

|   Bidders |   Max absolute BR error |
|----------:|------------------------:|
|         2 |               2.776e-17 |
|         3 |               0.0001583 |
|         5 |               1.11e-16  |
|        10 |               1.11e-16  |

## Takeaway

The first-price auction turns private information into bid shading. In the uniform symmetric benchmark, the equilibrium bid is a constant fraction of value, and that fraction rises with competition. The reusable computational move is to express a candidate equilibrium as a type-indexed payoff comparison, then check that no type has a profitable unilateral deviation given the strategy used by rival types.

## References

- [Vickrey, W. (1961). Counterspeculation, Auctions, and Competitive Sealed Tenders. *Journal of Finance*, 16(1), 8-37.](https://doi.org/10.1111/j.1540-6261.1961.tb02789.x)
- [Riley, J. G. and Samuelson, W. F. (1981). Optimal Auctions. *American Economic Review*, 71(3), 381-392.](https://www.jstor.org/stable/1802786)
- [Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.](https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1)
