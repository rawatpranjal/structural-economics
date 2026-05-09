# Schelling Segregation on a Checkerboard

## Overview

Schelling's segregation model starts from a simple city. Two groups occupy a checkerboard with some vacant locations. Each person looks only at nearby neighbors and moves if too few are from the same group.

The economic point is not that agents choose segregation as a social outcome. They choose local neighborhoods. The aggregate pattern is produced by the feedback from many moves.

This tutorial keeps the model in its classic form. We simulate a 50 x 50 city, sweep the minimum same-group neighbor share $\tau$, and track the segregation index $S(t)$ until the city stops moving.

## Equations

Let $G$ be the city grid. The state variable is the whole checkerboard, not an
aggregate stock. Cell $i$ at iteration $t$ is
$X_t(i)$. Empty cells have $X_t(i)=0$. Occupied cells have group
$g_i(t)=X_t(i)$, where $g_i(t)\in\lbrace A,B\rbrace$.

For each occupied cell, $N_i$ is the local Moore neighborhood: the adjacent
horizontal, vertical, and diagonal cells, up to eight in total. Empty cells are
possible destinations, but they are not neighbors whose type enters the local
composition. The occupied-neighbor count is

$$
O_i(t)=\sum_{j\in N_i} \mathbf{1}[X_t(j)\neq 0].
$$

The same-group neighbor count keeps only occupied neighbors with the same group
as the resident in cell $i$:

$$
m_i(t)=\sum_{j\in N_i} \mathbf{1}[X_t(j)=g_i(t)].
$$

When $O_i(t)>0$, the local same-group share is the object agents care about:

$$
s_i(t)=\frac{m_i(t)}{O_i(t)}.
$$

When an occupied cell has no occupied neighbors, set $s_i(t)=1$. This convention
treats isolation as not violating the local same-group requirement. The
threshold $\tau$ is the minimum acceptable same-group share. An agent is content
when

$$
s_i(t)\geq \tau.
$$

If $s_i(t)<\tau$, the agent is dissatisfied. Let $E_t$ be the set of vacant
cells. A move is allowed only if the destination would satisfy the same local
threshold for that agent's group. Thus agents do not choose a global segregation
target; they only search for an acceptable local neighborhood.

The aggregate segregation index averages local exposure among occupied cells:

$$
S(t)=\frac{1}{M}\sum_{i:X_t(i)\neq 0} s_i(t),
$$

where $M$ is the number of occupied cells, which stays fixed because moves only
swap an occupied cell with a vacancy. A random initial city with equal group
sizes has $S(t)$ near one half. Large values of $S(t)$ mean that the typical
person mostly sees same-group neighbors, even though each decision used only
the local rule above.

## Model Setup

The calibration follows the checkerboard spirit of Schelling's spatial proximity model. The numbers are artificial by design; they let us see the dynamic mechanism.

| Symbol | Value | Role |
|---|---|---|
| $G$ | 50 x 50 cells | City grid |
| $E_t$ | 10% of cells initially vacant | Empty cells that permit movement |
| $g_i$ | $A$ or $B$ | Occupant group in cell $i$ |
| $N_i$ | Moore neighborhood, up to 8 cells | Local reference group |
| $O_i(t)$ | occupied neighbors in $N_i$ | Denominator for local exposure |
| $s_i(t)$ | between 0 and 1 | Same-group neighbor share for cell $i$ |
| $\tau$ | 0.20 to 0.50 | Local tolerance threshold |
| $S(t)$ | average of $s_i(t)$ | Aggregate segregation index |
| $T$ | 100 iterations | Stop rule cap |
| Replications | 5 per threshold | Simulation noise check |

## Solution Method

The model is an agent-based simulation. There is no representative agent and no global optimization problem. The state is the whole checkerboard.

```text
Algorithm: Schelling checkerboard dynamics
Input: initial grid X_0, vacancy set E_0, threshold tau, horizon T
Output: city states X_t, segregation path S(t), move counts

For t = 0, 1, ..., T - 1:
  1. For every occupied cell i, compute O_i(t), m_i(t), and s_i(t).
  2. Let D_t be occupied cells with s_i(t) < tau.
  3. If D_t is empty, stop with a locally stable city.
  4. Visit cells in D_t in random order.
  5. For each still-dissatisfied i, form candidate vacant cells C_i(t):
       vacant cells e in E_t where group g_i would have exposure at least tau.
  6. If C_i(t) is nonempty, draw e from C_i(t), move g_i from i to e,
       and update X_t and E_t before visiting the next agent.
  7. Record X_{t+1}, S(t+1), and the number of moves.
  8. Stop if no one moves or if t + 1 = T.
```

The random order matters because one move changes the neighborhoods of nearby agents. That dependence is the point of the model. Small local moves change the local incentives faced by others, and the city can tip toward a much more sorted pattern.

## Results

The animation shows the focal run at $\tau=0.35$, just above one third. Blue cells are one group, orange cells are the other group, and the light cells are empty locations. Each frame is a saved city state after a wave of relocation decisions. The city begins close to a random mix. Dissatisfied agents move into locations where their local threshold is met, and same-group clusters become self-reinforcing.

<img src="figures/schelling-tau-035.gif" alt="Animated Schelling checkerboard at tau 0.35" width="80%">

The path plot tracks the segregation index $S(t)$ for four thresholds. The critical hyperparameter is $\tau$, the local tolerance threshold. At low thresholds, the city settles after little sorting. Near the one-third region, the same local rule produces a visibly higher same-group exposure. Small integer neighborhoods make this region important: with only a few occupied neighbors, one additional same-group neighbor can move an agent across the threshold. The plateaus come from the integer number of neighbors on a finite checkerboard.

<img src="figures/segregation-paths.png" alt="Segregation-index paths for selected thresholds" width="80%">

The threshold sweep makes the nonlinearity clearer. Schelling emphasized that a demand around one third generated much less segregation than a demand near one half in his checkerboard examples. This run shows the same qualitative lesson: final segregation rises quickly as the local demand moves out of the low-tolerance range.

<img src="figures/phase-transition.png" alt="Final segregation index by same-group threshold" width="80%">

Movement is concentrated early. Once enough agents have relocated, many neighborhoods become locally stable even though the aggregate city is far more sorted than the initial draw.

<img src="figures/move-counts.png" alt="Moved agents by iteration" width="80%">

The final city at $\tau=0.35$ has same-group clusters even though every agent used only local neighbor composition.

<img src="figures/final-city-tau-035.png" alt="Final checkerboard city for tau 0.35" width="80%">

Simulation detail behind the phase-transition figure. Each row averages over 5 random initial cities.

**Threshold sweep summary**

|   Threshold tau |   Mean final segregation S |   SD final segregation S |   Mean iterations |   Mean moves |   Converged runs |   Replications |
|----------------:|---------------------------:|-------------------------:|------------------:|-------------:|-----------------:|---------------:|
|           0.2   |                      0.574 |                    0.011 |               3.6 |          154 |                5 |              5 |
|           0.225 |                      0.583 |                    0.012 |               3.6 |          178 |                5 |              5 |
|           0.25  |                      0.583 |                    0.012 |               3.6 |          178 |                5 |              5 |
|           0.275 |                      0.682 |                    0.019 |               7   |          389 |                5 |              5 |
|           0.3   |                      0.752 |                    0.014 |               7.8 |          531 |                5 |              5 |
|           0.333 |                      0.752 |                    0.014 |               7.8 |          531 |                5 |              5 |
|           0.35  |                      0.767 |                    0.011 |               6.8 |          581 |                5 |              5 |
|           0.375 |                      0.767 |                    0.011 |               6.8 |          581 |                5 |              5 |
|           0.4   |                      0.82  |                    0.011 |               7.2 |          743 |                5 |              5 |
|           0.425 |                      0.836 |                    0.006 |               8.6 |          807 |                5 |              5 |
|           0.45  |                      0.867 |                    0.008 |               9.2 |          968 |                5 |              5 |
|           0.475 |                      0.867 |                    0.008 |               9.2 |          968 |                5 |              5 |
|           0.5   |                      0.867 |                    0.008 |               9.2 |          968 |                5 |              5 |

## Takeaway

The Schelling model is a warning about aggregation. Modest local tolerance rules need not preserve a mixed city. When movement changes the local environment faced by others, individual relocation decisions can create segregated aggregate patterns that are much stronger than the rule each agent follows.

## References

- [Schelling, T. C. (1971). Dynamic Models of Segregation. *The Journal of Mathematical Sociology*, 1(2), 143-186.](https://doi.org/10.1080/0022250X.1971.9989794)
- [Schelling, T. C. (1978). *Micromotives and Macrobehavior*. W. W. Norton.]
