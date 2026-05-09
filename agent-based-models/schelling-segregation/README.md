# Schelling Segregation on a Checkerboard

> How local tolerance rules can generate aggregate residential sorting.

## Overview

Schelling's segregation model starts from a simple city. Two groups occupy a checkerboard with some vacant locations. Each person looks only at nearby neighbors and moves if too few are from the same group.

The economic point is not that agents choose segregation as a social outcome. They choose local neighborhoods. The aggregate pattern is produced by the feedback from many moves.

This tutorial keeps the model in its classic form. We simulate a 50 x 50 city, sweep the minimum same-group neighbor share $\tau$, and track the segregation index $S(t)$ until the city stops moving.

## Equations

Let the city be a finite grid. Each occupied location $i$ has type
$g_i\in\{A,B\}$. Empty locations have no type. The local neighborhood
$N(i)$ is the set of at most eight adjacent cells around $i$.

The occupied neighbors of $i$ are

$$O(i)=\{j\in N(i): j \text{ is occupied}\}.$$

The same-group neighbor share is

$$s_i =
\begin{cases}
| \{j\in O(i): g_j=g_i\} | / |O(i)|, & |O(i)|>0, \\
1, & |O(i)|=0.
\end{cases}$$

An occupied location is content when

$$s_i \geq \tau.$$

If $s_i < \tau$, the agent is dissatisfied and may move to a vacant location.
The aggregate segregation index is the average same-group exposure among
occupied agents:

$$S(t)=\frac{1}{M}\sum_{i:g_i(t)\neq \emptyset} s_i(t),$$

where $M$ is the number of occupied cells. A random initial city with equal
group sizes has $S(t)$ near one half. Large values of $S(t)$ mean that the
typical person mostly sees same-group neighbors.

## Model Setup

The calibration follows the checkerboard spirit of Schelling's spatial proximity model. The numbers are artificial by design; they let us see the dynamic mechanism.

| Object | Value | Role |
|---|---:|---|
| Grid size | 50 x 50 | City locations |
| Vacancy share | 10% | Empty cells that permit movement |
| Group shares among occupied cells | 50% / 50% | Symmetric two-group benchmark |
| Neighborhood | Moore, up to 8 cells | Local reference group |
| Threshold sweep | 0.20 to 0.50 | Minimum same-group neighbor share |
| Replications per threshold | 5 | Simulation noise check |
| Maximum iterations | 100 | Stop rule cap |

## Solution Method

The model is an agent-based simulation. There is no representative agent and no global optimization problem. The state is the whole checkerboard.

```text
Algorithm: classic Schelling checkerboard dynamics
Input: grid size n, vacancy share v, threshold tau, maximum iterations T
Output: city path, segregation index S(t), moved agents by iteration

1. Randomly place the two groups and vacant cells on the grid.
2. For every occupied cell, compute the share of occupied neighbors from the same group.
3. Mark an agent dissatisfied if the share is below tau.
4. Visit dissatisfied agents in random order.
5. Move each still-dissatisfied agent to a random vacant cell that satisfies the threshold.
6. Record S(t) and repeat until no agent is dissatisfied or T is reached.
```

The random order matters because one move changes the neighborhoods of nearby agents. That dependence is the point of the model. Small local moves change the local incentives faced by others, and the city can tip toward a much more sorted pattern.

## Results

The animation shows the focal run at $\tau=0.35$, just above one third. The city begins close to a random mix. Dissatisfied agents move into locations where their local threshold is met, and same-group clusters become self-reinforcing.

<img src="figures/schelling-tau-035.gif" alt="Animated Schelling checkerboard at tau 0.35" width="80%">

The path plot tracks the segregation index $S(t)$ for four thresholds. At low thresholds, the city settles after little sorting. Near the one-third region, the same local rule produces a visibly higher same-group exposure. The plateaus come from the integer number of neighbors on a finite checkerboard.

<img src="figures/segregation-paths.png" alt="Segregation-index paths for selected thresholds" width="80%">

The threshold sweep makes the nonlinearity clearer. Schelling emphasized that a demand around one third generated much less segregation than a demand near one half in his checkerboard examples. This run shows the same qualitative lesson: final segregation rises quickly as the local demand moves out of the low-tolerance range.

<img src="figures/phase-transition.png" alt="Final segregation index by same-group threshold" width="80%">

Movement is concentrated early. Once enough agents have relocated, many neighborhoods become locally stable even though the aggregate city is far more sorted than the initial draw.

<img src="figures/move-counts.png" alt="Moved agents by iteration" width="80%">

The final city at $\tau=0.35$ has same-group clusters even though every agent used only local neighbor composition.

<img src="figures/final-city-tau-035.png" alt="Final checkerboard city for tau 0.35" width="80%">

Simulation detail behind the phase-transition figure. Each row averages over 5 random initial cities.

**Threshold sweep summary**

|   tau |   mean_final_S |   sd_final_S |   mean_iterations |   mean_moves |   converged_runs |   runs |
|------:|---------------:|-------------:|------------------:|-------------:|-----------------:|-------:|
| 0.2   |          0.574 |        0.011 |               3.6 |          154 |                5 |      5 |
| 0.225 |          0.583 |        0.012 |               3.6 |          178 |                5 |      5 |
| 0.25  |          0.583 |        0.012 |               3.6 |          178 |                5 |      5 |
| 0.275 |          0.682 |        0.019 |               7   |          389 |                5 |      5 |
| 0.3   |          0.752 |        0.014 |               7.8 |          531 |                5 |      5 |
| 0.333 |          0.752 |        0.014 |               7.8 |          531 |                5 |      5 |
| 0.35  |          0.767 |        0.011 |               6.8 |          581 |                5 |      5 |
| 0.375 |          0.767 |        0.011 |               6.8 |          581 |                5 |      5 |
| 0.4   |          0.82  |        0.011 |               7.2 |          743 |                5 |      5 |
| 0.425 |          0.836 |        0.006 |               8.6 |          807 |                5 |      5 |
| 0.45  |          0.867 |        0.008 |               9.2 |          968 |                5 |      5 |
| 0.475 |          0.867 |        0.008 |               9.2 |          968 |                5 |      5 |
| 0.5   |          0.867 |        0.008 |               9.2 |          968 |                5 |      5 |

## Takeaway

The Schelling model is a warning about aggregation. Modest local tolerance rules need not preserve a mixed city. When movement changes the local environment faced by others, individual relocation decisions can create segregated aggregate patterns that are much stronger than the rule each agent follows.

## References

- [Schelling, T. C. (1971). Dynamic Models of Segregation. *The Journal of Mathematical Sociology*, 1(2), 143-186.](https://doi.org/10.1080/0022250X.1971.9989794)
- [Schelling, T. C. (1978). *Micromotives and Macrobehavior*. W. W. Norton.]
