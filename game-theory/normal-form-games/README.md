# Normal-Form Games and Nash Equilibrium Checks

> Pure profiles, mixed supports, and unilateral-deviation residuals.

## Overview

A normal-form game is the payoff table behind many richer models. Before adding states, prices, or private information, Nash equilibrium is just a set of no-profitable-deviation restrictions on that table. This tutorial keeps the games small enough that those restrictions can be inspected directly.

The point is not to showcase a solver. Pure equilibria come from checking every action profile. Interior mixed equilibria in 2x2 games come from the two indifference equations that make randomization optimal. The same logic is the static baseline for [Cournot best-response dynamics](../static-games/) and the exact benchmark that [quantal response equilibrium](../quantal-response-equilibrium/) softens into noisy best responses.

## Equations

There are two players. The row player has actions $i \in I$, the column player
has actions $j \in J$, and the payoff matrices are $A$ for the row player and
$B$ for the column player. At pure profile $(i,j)$, payoffs are $(A_{ij},B_{ij})$.

The row player's one-step deviation gain at $(i,j)$ is

$$
d_1(i,j)=\max_{i' \in I} A_{i'j}-A_{ij},
$$

and the column player's gain is

$$
d_2(i,j)=\max_{j' \in J} B_{ij'}-B_{ij}.
$$

A pure Nash equilibrium is a profile $(i^{\ast}, j^{\ast})$ with

$$
d_1(i^{\ast},j^{\ast})=d_2(i^{\ast},j^{\ast})=0,
$$

equivalently

$$
A_{i^{\ast}j^{\ast}} \geq A_{ij^{\ast}} \quad \forall i \in I,
\qquad
B_{i^{\ast}j^{\ast}} \geq B_{i^{\ast}j} \quad \forall j \in J.
$$

For a 2x2 game, let the row player use mixed strategy $x=(p,1-p)$ and the
column player use $y=(q,1-q)$. An interior mixed equilibrium requires both
players to be indifferent across the actions used with positive probability:

$$
A_{11}q + A_{12}(1-q) = A_{21}q + A_{22}(1-q),
\qquad
B_{11}p + B_{21}(1-p) = B_{12}p + B_{22}(1-p).
$$

The candidate is an equilibrium only if $p,q \in [0,1]$. The reported mixed
residual is the maximum absolute gap in these two indifference equations.

## Model Setup

The tutorial uses four canonical 2x2 games. Each payoff table is small enough that the economic tension is visible in the cells, but the equilibrium patterns are different enough to separate dominance, zero-sum mixing, and coordination.

| Game | Actions | What the payoffs isolate |
|---|---|---|
| Prisoner's Dilemma | Cooperate/Defect | Individual incentives overturn the efficient profile. |
| Matching Pennies | Heads/Tails | No pure action can be predictable in equilibrium. |
| Battle of the Sexes | Opera/Football | Coordination is valuable, but players prefer different conventions. |
| Stag Hunt | Stag/Hare | Safe and payoff-dominant coordination profiles coexist. |

## Solution Method

The computation is exact for these finite games. Enumeration handles pure profiles. The 2x2 mixed calculation solves the closed-form indifference system and then checks the candidate rather than trusting the formula mechanically.

```text
Algorithm: Nash checks for a two-player finite game
Inputs: payoff matrices A, B and action labels I, J
Outputs: pure Nash set E and, for 2x2 games, an interior mixed candidate

1. For each pure profile (i,j), compute d1(i,j) and d2(i,j).
2. Add (i,j) to E when max{d1(i,j), d2(i,j)} = 0.
3. If the game is 2x2, solve the two linear indifference equations for p and q.
4. Keep the mixed candidate only when p and q lie in [0,1].
5. Recompute both expected-payoff gaps and report the largest absolute residual.
```

This residual is the useful diagnostic. A profile or mixed strategy is not interesting because an algorithm named it; it is interesting because no player can improve by changing only its own action.

## Results

The first figure colors each pure action profile by the largest profitable unilateral deviation. A zero-deviation cell is a pure Nash equilibrium, so the black outlines are not decorative markers; they are exactly the cells where the equilibrium inequalities bind. This also separates efficiency from equilibrium. In Prisoner's Dilemma, mutual cooperation has higher joint payoff than mutual defection, but it is not stable against a one-player deviation.

<img src="figures/pure-deviation-gains.png" alt="Pure payoff tables colored by unilateral-deviation gain" width="80%">

The mixed-strategy figure uses the closed-form indifference equations as the ground truth. Each curve is the expected payoff from the first action minus the expected payoff from the second action. A root is where the opponent's mix makes that player willing to randomize. Matching Pennies has the symmetric half-half root; Battle of the Sexes has asymmetric mixing because the players prefer different coordinated outcomes; Stag Hunt's mixed equilibrium is the knife-edge between the safe and payoff-dominant basins.

<img src="figures/mixed-indifference.png" alt="Exact mixed-equilibrium indifference roots" width="80%">

The summary table reports the exact pure-equilibrium set and the interior mixed candidate when one exists. Residuals are numerical checks of the closed-form indifference equations.

**Equilibrium Summary by Game**

| Game                | Pure Nash equilibria                 | Interior mixed equilibrium                  | Indifference residual   | Equilibrium pattern                                        |
|:--------------------|:-------------------------------------|:--------------------------------------------|:------------------------|:-----------------------------------------------------------|
| Prisoner's Dilemma  | (Defect, Defect)                     | None                                        | None                    | Dominance: defection is the unique stable profile.         |
| Matching Pennies    | None                                 | Pr(row Heads)=0.500; Pr(column Heads)=0.500 | 0.0e+00                 | No pure equilibrium; mixing removes predictable play.      |
| Battle of the Sexes | (Opera, Opera), (Football, Football) | Pr(row Opera)=0.600; Pr(column Opera)=0.400 | 2.2e-16                 | Two coordination equilibria plus one mixed conflict point. |
| Stag Hunt           | (Stag, Stag), (Hare, Hare)           | Pr(row Stag)=0.667; Pr(column Stag)=0.667   | 4.4e-16                 | Two coordination equilibria, one payoff dominant.          |

Expected payoffs at the mixed equilibria are included because the probabilities alone can hide the economic tradeoff. In zero-sum Matching Pennies both players get zero; in the coordination games the mixed point is worse than successful coordination.

**Expected Payoffs at Interior Mixed Equilibria**

| Game                |   Row first-action probability p |   Column first-action probability q |   Row payoff |   Column payoff |
|:--------------------|---------------------------------:|------------------------------------:|-------------:|----------------:|
| Matching Pennies    |                            0.5   |                               0.5   |        0     |           0     |
| Battle of the Sexes |                            0.6   |                               0.4   |        1.2   |           1.2   |
| Stag Hunt           |                            0.667 |                               0.667 |        2.667 |           2.667 |

## Takeaway

For finite static games, Nash equilibrium is best read as a residual condition. Pure equilibria are zero-deviation cells in the payoff table. Interior mixed equilibria choose probabilities that make opponents indifferent across the actions they use. This direct calculation is the right benchmark before moving to fixed-point iteration, noisy response, or dynamic games where the same no-deviation logic is harder to see.

## References

- [Nash, J. (1950). Equilibrium Points in N-Person Games. *Proceedings of the National Academy of Sciences*, 36(1), 48-49.](https://doi.org/10.1073/pnas.36.1.48)
- [Osborne, M. and Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.](https://mitpress.mit.edu/9780262650403/a-course-in-game-theory)
- [Lemke, C. E. and Howson, J. T. (1964). Equilibrium Points of Bimatrix Games. *SIAM Journal on Applied Mathematics*, 12(2), 413-423.](https://doi.org/10.1137/0112033)
