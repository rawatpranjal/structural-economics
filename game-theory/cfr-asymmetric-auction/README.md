# Asymmetric First-Price Auctions by Counterfactual Regret Minimization

## Overview

Two bidders compete in a sealed-bid first-price auction with private values. Their value distributions are not the same. One bidder draws from a uniform distribution on a small support; the other draws from a uniform distribution on a wider support. The symmetric closed-form bid rule from the existing first-price auction tutorial does not apply. Best-response iteration cycles instead of converging.

Counterfactual regret minimization is a learning algorithm that handles this case. Each bidder type is treated as an information set. The bidder accumulates regret for each candidate bid against the opponent's current strategy. The next strategy puts probability on bids in proportion to their positive cumulative regret. The time-averaged strategy converges to a Bayesian Nash equilibrium.

The tutorial implements vanilla CFR and CFR+ on the asymmetric game. It checks both algorithms against the symmetric closed form by setting the two value distributions equal. It tracks exploitability of the average strategy on the asymmetric game as the no-deviation diagnostic, the same idea as the bid-grid deviation check in the existing first-price auction tutorial.

## Equations

The auction has two bidders indexed by $i \in \lbrace 1, 2 \rbrace$. Bidder $i$ draws a
private value $v_i$ from a known distribution $F_i$, independently across bidders.
Each bidder submits one sealed bid $b_i$ on a finite bid grid $B$, the highest bid
wins, the winner pays its own bid, and ties are broken uniformly at random.

A behavioral strategy $\sigma_i(b \mid v)$ is a probability distribution over bids
for each type $v$. The information set $I_v$ for bidder $i$ is the set of game
histories at which the bidder has observed type $v$. There is one information set
per type.

The expected payoff to bidder $i$ from bidding $b$ at type $v$ against the
opponent strategy $\sigma_{-i}$ is

$$
u_i(v, b; \sigma_{-i}) = (v - b) \cdot \Pr(\text{win} \mid b, \sigma_{-i}).
$$

The win probability uses the marginal opponent bid distribution
$q_{-i}(b') = \sum_{v'} P(v') \sigma_{-i}(b' \mid v')$ and the uniform tie-break
$\Pr(\text{win} \mid b) = \sum_{b' < b} q_{-i}(b') + \tfrac{1}{2} q_{-i}(b)$.

The counterfactual value at information set $I_v$ for action $b$ multiplies the
expected payoff by the chance reach probability $P(v)$:

$$
v_i^{\sigma}(I_v, b) = P(v) \cdot u_i(v, b; \sigma_{-i}).
$$

The instantaneous regret at iteration $t$ is the gap between the value of
deviating to $b$ and the value of the current mixed strategy:

$$
r_i^{t}(I_v, b) = v_i^{\sigma^{t}}(I_v, b) - \sum_{b'} \sigma_i^{t}(b' \mid v) \cdot v_i^{\sigma^{t}}(I_v, b').
$$

Cumulative regret accumulates these one-shot gaps:

$$
R_i^{T}(I_v, b) = \sum_{t = 1}^{T} r_i^{t}(I_v, b).
$$

The next strategy is regret matching, which puts mass on each bid in proportion
to its positive cumulative regret:

$$
\sigma_i^{T+1}(b \mid v) = \frac{\max(R_i^{T}(I_v, b), 0)}{\sum_{b'} \max(R_i^{T}(I_v, b'), 0)},
$$

with a uniform fallback when every cumulative regret is non-positive.

CFR+ replaces the cumulative regret with a non-negative running sum. Negative
contributions cannot accumulate:

$$
R_i^{+,T}(I_v, b) = \max(R_i^{+,T-1}(I_v, b) + r_i^{t}(I_v, b),\ 0).
$$

CFR+ also alternates updates so that one player at a time refreshes its regret,
and weighs each iteration's strategy by $t$ in the average:

$$
\bar{\sigma}_i^{T}(b \mid v) = \frac{\sum_{t = 1}^{T} t \cdot \sigma_i^{t}(b \mid v)}{\sum_{t = 1}^{T} t \cdot \sum_{b'} \sigma_i^{t}(b' \mid v)}.
$$

Vanilla CFR uses uniform averaging in place of the linear weight $t$.

The exploitability of a strategy profile $\sigma$ is the sum across players of
the most a single bidder could gain by switching to the best response:

$$
\varepsilon(\sigma) = \sum_{i = 1}^{2} \left(\max_{\sigma'_i} U_i(\sigma'_i, \sigma_{-i}) - U_i(\sigma_i, \sigma_{-i})\right),
$$

where $U_i(\sigma) = \sum_{v} P(v) \sum_{b} \sigma_i(b \mid v) \cdot u_i(v, b; \sigma_{-i})$
is the ex-ante expected payoff. Exploitability equals zero exactly at a Bayesian
Nash equilibrium of the discretized game. The best response in the maximization
is computed by picking, at each type, the bid on $B$ with the highest expected
payoff.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Weak bidder values | $v_1 \sim U[0, 1]$ | Smaller-support distribution |
| Strong bidder values | $v_2 \sim U[0, 2]$ | Larger-support distribution |
| Type grid | 21 nodes per bidder | Each type is one information set |
| Bid grid | 41 nodes on $[0, 1]$ | Shared discrete action set |
| Iterations | 5,000 | Same budget for vanilla CFR and CFR+ |
| Tie-break | Uniform | Splits ties evenly across bidders |
| Symmetric check | $v_1, v_2 \sim U[0, 1]$ | Compares to $b^{\ast}(v) = v / 2$ |

## Solution Method

Each bidder type is its own information set. The bidder runs regret matching locally at every type, accumulating regret for each candidate bid against the opponent's current strategy. The Hannan-consistent regret bound at each information set, together with the chance reach weighting, gives a global bound on the average regret of the time-averaged strategy. In two-player zero-sum games, that bound translates directly into exploitability, so the average strategy is an approximate Bayesian Nash equilibrium.

Why regret matching works can be seen in a one-information-set toy. Suppose action $a$ always pays 2 and action $b$ always pays 1 against a fixed opponent. Starting from uniform play the average payoff is 1.5. Action $a$ accumulates regret 0.5 per iteration and action $b$ accumulates regret minus 0.5. After a few iterations the strategy puts all mass on $a$. The time-averaged strategy converges to the dominant action.

```text
Algorithm: vanilla CFR for the asymmetric first-price auction
Inputs: type grids V_1, V_2 with PMFs P_1, P_2; bid grid B; iterations T
Outputs: time-averaged strategies sigma_bar_1, sigma_bar_2

1. Initialize R_i(v, b) = 0 and S_i(v, b) = 0 for i in {1, 2}.
2. For t = 1, 2, ..., T:
   a. For each i, compute sigma_i^t(b | v) by regret matching on R_i(v, .).
   b. For each i, form the marginal opponent bid PMF
      q_{-i}(b) = sum_{v'} P_{-i}(v') sigma_{-i}^t(b | v').
   c. Compute the win probability w_{-i}(b) under uniform tie-break.
   d. For each i, v in V_i, b in B, compute the counterfactual value
      cf_i(v, b) = P_i(v) (v - b) w_{-i}(b)
      and the iteration-average value cf_i_avg(v) = sum_b sigma_i^t(b | v) cf_i(v, b).
   e. R_i(v, b) <- R_i(v, b) + cf_i(v, b) - cf_i_avg(v).
   f. S_i(v, b) <- S_i(v, b) + sigma_i^t(b | v).
3. Return sigma_bar_i(b | v) = S_i(v, b) / sum_{b'} S_i(v, b').
```

CFR+ changes three lines and converges much faster on this game.

```text
Algorithm: CFR+ (changes versus vanilla CFR)
- Step 2e: R_i(v, b) <- max(R_i(v, b) + cf_i(v, b) - cf_i_avg(v), 0).
- Step 2 alternates updates: refresh player 1, then refresh player 2 against
  player 1's already updated regret. Each iteration still touches both players.
- Step 2f: S_i(v, b) <- S_i(v, b) + t * sigma_i^t(b | v) (linear averaging).
```

Exploitability of the average strategy is the deviation diagnostic. At each logged iteration the code computes the best-response payoff at every type and subtracts the average-strategy payoff. The expected gap, summed across bidders, is the exploitability.

## Results

The average strategies of CFR+ on the asymmetric game show the textbook asymmetry. The weak bidder bids more aggressively per unit of value than they would in the symmetric game with the same support. The weak bidder faces a rival who often holds a higher value, so shading too much loses too many auctions. The strong bidder shades more deeply for any given value because the weak rival rarely bids above the weak support upper bound. Strong-bidder bids stay below the weak-support upper bound at one. The asymmetric game has no closed-form solution but the bid functions are smooth and monotone.

<img src="figures/bid-functions-asymmetric.png" alt="Asymmetric bid functions from CFR+" width="80%">

Exploitability of the average strategy falls steadily for both algorithms. Vanilla CFR drops at roughly the textbook rate of order one over the square root of iterations. CFR+ is about an order of magnitude smaller for a fixed iteration budget thanks to the regret floor, alternating updates, and linear weighting of the strategy average. Exploitability never reaches exactly zero because the bid grid is finite and the symmetric closed form is the true target only on a continuum, but the residual gap is small relative to expected revenue. Exploitability is the asymmetric analogue of the bid-grid deviation check used by the existing first-price auction tutorial.

<img src="figures/exploitability.png" alt="Exploitability decay for vanilla CFR and CFR+" width="80%">

Setting both value distributions to uniform on the unit interval recovers a case where the analytic Bayesian Nash bid is half the value. Both CFR variants track the closed form to within bid-grid spacing. CFR+ tracks the closed form more tightly because it averages later iterations more heavily, when the strategy is closer to the equilibrium bid. The sanity check confirms that the implementation finds the right equilibrium on a case where the right answer is known.

<img src="figures/bid-functions-symmetric.png" alt="Symmetric uniform sanity check" width="80%">

The symmetric residual is the maximum gap between the CFR average bid and the closed-form rule when both bidders draw from the same uniform distribution. Asymmetric exploitability is the sum of best-response payoff gains at the average strategy on the asymmetric game.

**Methods summary**

| Method      |   Symmetric residual (max bid error) |   Asymmetric exploitability (final) | Iterations   |
|:------------|-------------------------------------:|------------------------------------:|:-------------|
| Vanilla CFR |                             0.009741 |                           0.0003624 | 5,000        |
| CFR+        |                             0.008108 |                           5.573e-05 | 5,000        |

Logarithmically spaced iteration checkpoints. Each row reports the exploitability of the time-averaged strategy under vanilla CFR and CFR+.

**Exploitability decay on the asymmetric game**

|   Iteration |   Vanilla CFR exploitability |   CFR+ exploitability |
|------------:|-----------------------------:|----------------------:|
|           1 |                    0.3465    |             0.3465    |
|           2 |                    0.2286    |             0.1875    |
|           3 |                    0.1809    |             0.1236    |
|           4 |                    0.1505    |             0.09046   |
|           5 |                    0.1291    |             0.07089   |
|           6 |                    0.1133    |             0.05831   |
|           7 |                    0.1011    |             0.04962   |
|           9 |                    0.08366   |             0.03852   |
|          11 |                    0.0718    |             0.03174   |
|          14 |                    0.05955   |             0.02533   |
|          17 |                    0.05098   |             0.02133   |
|          21 |                    0.04301   |             0.01793   |
|          26 |                    0.03614   |             0.01506   |
|          33 |                    0.03      |             0.01243   |
|          41 |                    0.02508   |             0.01007   |
|          51 |                    0.02093   |             0.00817   |
|          63 |                    0.01745   |             0.006664  |
|          79 |                    0.01446   |             0.005375  |
|          98 |                    0.01216   |             0.004377  |
|         122 |                    0.01016   |             0.003589  |
|         152 |                    0.008572  |             0.002911  |
|         189 |                    0.007109  |             0.002485  |
|         235 |                    0.005916  |             0.002065  |
|         292 |                    0.00493   |             0.001692  |
|         364 |                    0.004067  |             0.001323  |
|         453 |                    0.003386  |             0.001066  |
|         563 |                    0.002807  |             0.0008233 |
|         700 |                    0.002282  |             0.0006412 |
|         871 |                    0.001847  |             0.0005137 |
|        1084 |                    0.001505  |             0.0003971 |
|        1349 |                    0.001266  |             0.0003055 |
|        1678 |                    0.001019  |             0.0002454 |
|        2087 |                    0.0008371 |             0.0001853 |
|        2597 |                    0.0006796 |             0.0001392 |
|        3231 |                    0.0005538 |             9.982e-05 |
|        4019 |                    0.0004489 |             7.473e-05 |
|        5000 |                    0.0003624 |             5.573e-05 |

## Takeaway

Counterfactual regret minimization replaces the analytic Bayesian Nash calculation with a regret-matching loop on the discretized game. The algorithm applies whenever each player has its own information set, including auctions where the closed form is unavailable.

Exploitability is the no-deviation diagnostic that takes the place of the bid-grid deviation check used in the symmetric tutorial. CFR+ converges roughly an order of magnitude faster than vanilla CFR by clipping negative cumulative regret, alternating updates, and weighing later iterations more heavily in the strategy average.

The same algorithm, scaled up to large extensive-form games, is the engine behind modern poker AI.

## References

- [Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2007). Regret Minimization in Games with Incomplete Information. *Advances in Neural Information Processing Systems*, 20.](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)
- [Tammelin, O., Burch, N., Johanson, M., and Bowling, M. (2015). Solving Heads-Up Limit Texas Hold'em. *IJCAI*, 645-652.](https://www.ijcai.org/Proceedings/15/Papers/097.pdf)
- [Maskin, E. and Riley, J. (2000). Asymmetric Auctions. *Review of Economic Studies*, 67(3), 413-438.](https://doi.org/10.1111/1467-937X.00137)
- [Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.](https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1)
