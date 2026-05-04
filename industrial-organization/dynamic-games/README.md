# Dynamic Games and Markov-Perfect Investment

> Ericson-Pakes style strategic dynamics with quality states and investment choices.

## Overview

Dynamic IO games let firm actions change the future state of competition. A firm may invest today not only because it raises future quality, but also because it changes its rival's future incentives. The Markov-perfect restriction keeps strategies payoff-relevant: actions depend on the current state, not on the entire history.

This tutorial solves a two-firm quality ladder. Each firm chooses whether to invest. Investment is costly but can move the firm up one quality state. Current profits come from a differentiated-products share formula.

## Equations

State is the quality pair:
$$\omega_t = (q_{1t}, q_{2t})$$

Markov-perfect values satisfy:
$$V_i(\omega) = \max_{a_i\in\{0,1\}} \pi_i(\omega) - \kappa a_i + \beta E[V_i(\omega')|\omega,a_i,a_{-i}^*(\omega)]$$

Investment changes transition probabilities:
$$Pr(q_i'=q_i+1|a_i=1)=0.62$$

The equilibrium policy maps each state into investment actions:
$$a_i^*(\omega)\in\{0,1\}$$

## Model Setup

| Object | Value |
|--------|-------|
| Firms | 2 |
| Quality states | 0 through 4 |
| Actions | Invest or do not invest |
| Discount factor | 0.90 |
| Investment cost | 2.20 |
| Equilibrium concept | Pure-strategy Markov-perfect equilibrium at each state |

## Solution Method

The solver iterates on firm value functions. At each state it constructs the two-by-two investment payoff matrix using the previous value function, finds a pure Nash equilibrium of that state game, and updates continuation values. The state space is deliberately small so the dynamic-game logic is visible.

## Results

Investment incentives are strongest when a firm is behind or close to its rival. At high own quality, the marginal benefit of another quality step is smaller.

<img src="figures/investment-policy.png" alt="Firm 1 investment policy over the quality state space" width="80%">
*Firm 1 investment policy over the quality state space*

Dynamic state variables are payoff relevant. A one-step quality lead changes both current market share and future investment incentives.

<img src="figures/value-advantage.png" alt="Value advantage across states" width="80%">
*Value advantage across states*

The vertical lines mark periods with at least one investment action. Quality leadership is persistent but not permanent because investment and depreciation keep the state moving.

<img src="figures/simulated-quality-path.png" alt="Simulated quality paths under Markov-perfect policies" width="80%">
*Simulated quality paths under Markov-perfect policies*

**Selected state policies and values**

| State   | Firm 1 policy   | Firm 2 policy   |   Firm 1 value |   Firm 2 value |
|:--------|:----------------|:----------------|---------------:|---------------:|
| (0,0)   | Invest          | Invest          |          60.03 |          60.03 |
| (1,2)   | Invest          | Invest          |          58.87 |          78.59 |
| (2,1)   | Invest          | Invest          |          78.59 |          58.87 |
| (4,4)   | Wait            | Wait            |          78.52 |          78.52 |

## Takeaway

Dynamic games turn IO counterfactuals into state-transition problems. The hard part is not just computing a price or entry outcome today; it is tracking how current actions change tomorrow's competitive state and therefore tomorrow's incentives.

## Reproduce

```bash
python run.py
```

## References

- Ericson, R., and Pakes, A. (1995). Markov-Perfect Industry Dynamics. *Review of Economic Studies*, 62(1), 53-82.
- Pakes, A., and McGuire, P. (1994). Computing Markov-Perfect Nash Equilibria. *RAND Journal of Economics*, 25(4), 555-589.
- Lecture 17 Slides 2023: Dynamic games and the Ericson-Pakes framework.
