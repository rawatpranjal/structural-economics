# Finite Strategic Games and Nash Equilibrium Checks

> Payoff tables, deviation gains, and mixed-strategy indifference.

## Overview

Many economic situations ask the same question: after each participant chooses an action, would anyone want to switch? A pair of firms may decide whether to compete hard or cooperate tacitly, two groups may need a convention, and a matching game may reward players who avoid being predictable. A normal-form game records that strategic situation in one payoff table.

Because the table is finite, the computation can stay close to the definition. For each action profile, calculate the best payoff each player could get by changing only their own action. A Nash equilibrium is a cell with zero profitable deviation. In 2x2 games, mixed equilibria add one more calculation: choose probabilities that make each player indifferent across the actions they randomize over. These checks give the static benchmark for [Cournot best-response dynamics](../static-games/) and for [quantal response equilibrium](../quantal-response-equilibrium/), where best responses become payoff-sensitive choice probabilities.

## Equations

A finite two-player game has a row player with actions $i \in I$ and a column
player with actions $j \in J$. The matrices $A$ and $B$ record row and column
payoffs. At pure profile $(i,j)$, the players receive $(A_{ij},B_{ij})$.

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
residual is the maximum absolute gap left in these two indifference equations.

## Model Setup

Four canonical 2x2 games keep the economic forces visible in the cells. Prisoner's Dilemma shows private incentives defeating joint surplus. Matching Pennies shows why predictable pure actions cannot survive in a strictly opposed game. Battle of the Sexes and Stag Hunt show two kinds of coordination: conflict over convention and risk around cooperation.

| Game | Actions | What the payoffs isolate |
|---|---|---|
| Prisoner's Dilemma | Cooperate/Defect | Individual incentives overturn the efficient profile. |
| Matching Pennies | Heads/Tails | No pure action can be predictable in equilibrium. |
| Battle of the Sexes | Opera/Football | Coordination is valuable, but players prefer different conventions. |
| Stag Hunt | Stag/Hare | Safe and payoff-dominant coordination profiles coexist. |

## Solution Method

The algorithm treats equilibrium as a checkable set of inequalities. It computes deviation gains at every pure profile. When a 2x2 game has a possible interior mixture, it solves two linear equations for the probabilities that make each player indifferent.

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

The residual turns the equilibrium claim into a number. A pure profile passes when both deviation gains equal zero. A mixed profile passes when the payoff gaps for the actions in the support are numerically zero.

## Results

The heat maps read each payoff table through incentives to deviate. Warmer cells have larger one-player gains from switching action. A black outline marks a zero-deviation cell, so it marks a pure Nash equilibrium. Prisoner's Dilemma shows the main economic lesson: mutual cooperation creates more total surplus, yet mutual defection is the stable prediction because each player wants to defect when the other cooperates.

<img src="figures/pure-deviation-gains.png" alt="Payoff tables colored by profitable deviation gains" width="80%">

The mixed-strategy panels plot the payoff differences behind randomization. Each curve shows the first-action payoff minus the second-action payoff as the opponent's probability changes. A root gives the probability that makes that player willing to mix. Matching Pennies lands at half-half. In Battle of the Sexes, the probabilities are asymmetric because the players value different conventions. In Stag Hunt, the mixed point separates attraction to safe and payoff-dominant coordination.

<img src="figures/mixed-indifference.png" alt="Mixed-strategy indifference roots in 2x2 games" width="80%">

The summary table translates the same checks into equilibrium objects. Pure-equilibrium entries list zero-deviation cells. The mixed entries list the interior probability pair and the largest indifference residual.

**Equilibrium Summary by Game**

| Game                | Pure Nash equilibria                 | Interior mixed equilibrium                  | Indifference residual   | Economic pattern                                                            |
|:--------------------|:-------------------------------------|:--------------------------------------------|:------------------------|:----------------------------------------------------------------------------|
| Prisoner's Dilemma  | (Defect, Defect)                     | None                                        | None                    | Defection is stable even though cooperation has higher joint payoff.        |
| Matching Pennies    | None                                 | Pr(row Heads)=0.500; Pr(column Heads)=0.500 | 0.0e+00                 | Any predictable pure action invites a profitable response.                  |
| Battle of the Sexes | (Opera, Opera), (Football, Football) | Pr(row Opera)=0.600; Pr(column Opera)=0.400 | 2.2e-16                 | Two conventions are stable; mixing balances conflicting preferred outcomes. |
| Stag Hunt           | (Stag, Stag), (Hare, Hare)           | Pr(row Stag)=0.667; Pr(column Stag)=0.667   | 4.4e-16                 | Safe and payoff-dominant conventions both satisfy no-deviation.             |

Expected payoffs put the mixed probabilities in economic terms. In zero-sum Matching Pennies both players get zero. In the coordination games the mixed point gives lower payoffs than successful coordination.

**Expected Payoffs at Interior Mixed Equilibria**

| Game                |   Row first-action probability p |   Column first-action probability q |   Row payoff |   Column payoff |
|:--------------------|---------------------------------:|------------------------------------:|-------------:|----------------:|
| Matching Pennies    |                            0.5   |                               0.5   |        0     |           0     |
| Battle of the Sexes |                            0.6   |                               0.4   |        1.2   |           1.2   |
| Stag Hunt           |                            0.667 |                               0.667 |        2.667 |           2.667 |

## Takeaway

A finite normal-form game makes Nash equilibrium observable in the payoff table. Enumeration finds pure equilibria by asking whether anyone can profitably switch actions. The 2x2 mixed check chooses probabilities that erase payoff gaps within each player's support. This direct calculation is the benchmark for larger fixed-point, noisy-response, and dynamic-game computations.

## References

- [Nash, J. (1950). Equilibrium Points in N-Person Games. *Proceedings of the National Academy of Sciences*, 36(1), 48-49.](https://doi.org/10.1073/pnas.36.1.48)
- [Osborne, M. and Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.](https://mitpress.mit.edu/9780262650403/a-course-in-game-theory)
- [Lemke, C. E. and Howson, J. T. (1964). Equilibrium Points of Bimatrix Games. *SIAM Journal on Applied Mathematics*, 12(2), 413-423.](https://doi.org/10.1137/0112033)
