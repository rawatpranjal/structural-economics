# Proofread: industrial-organization/dynamic-games/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T08:45:00Z._

## Paper / Source Verification

### Ericson, R., and Pakes, A. (1995). Markov-Perfect Industry Dynamics. *Review of Economic Studies*, 62(1), 53-82.

- **Located:** https://academic.oup.com/restud/article-abstract/62/1/53/1568000
- **Tutorial claims:** Foundational reference for the two-firm quality-ladder model and Markov-perfect equilibrium framework.
- **Source says:** Full title is "Markov-Perfect Industry Dynamics: A Framework for Empirical Work"; volume 62, issue 1, pages 53-82, year 1995, DOI 10.2307/2297841. All cited fields match.
- **Verdict:** OK
- **Note:** Subtitle ": A Framework for Empirical Work" is omitted; omitting subtitles is standard citation practice.

### Pakes, A., and McGuire, P. (1994). Computing Markov-Perfect Nash Equilibria. *RAND Journal of Economics*, 25(4), 555-589.

- **Located:** https://ideas.repec.org/a/rje/randje/v25y1994iwinterp555-589.html
- **Tutorial claims:** Reference for the computational algorithm that iterates on values and strategies to find a fixed point.
- **Source says:** Full title is "Computing Markov-Perfect Nash Equilibria: Numerical Implications of a Dynamic Differentiated Product Model"; volume 25, issue 4 (Winter 1994), pages 555-589. All cited fields match.
- **Verdict:** OK
- **Note:** Subtitle ": Numerical Implications of a Dynamic Differentiated Product Model" is omitted; standard practice.

### Lecture 17 Slides 2023: Dynamic games and the Ericson-Pakes framework.

- **Located:** NOT FOUND
- **Tutorial claims:** Supplementary reference for the dynamic-games topic.
- **Source says:** No external source located; this appears to be internal course material.
- **Verdict:** NOT FOUND
- **Note:** Internal course slide decks cannot be verified; no URL or author is supplied.

## Main Message Audit

> "The quality-ladder game makes investment rivalry a state-transition problem. The computed policy maps quality states into actions and continuation values. In this calibration, firms invest until the cap binds. Off-diagonal states show why a quality lead is valuable."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Investment rivalry is a state-transition problem | Equations section defines $\omega_t=(q_{1t},q_{2t})$ and the fixed-point system over states | OK |
| Computed policy maps states into actions and continuation values | Results table lists policy and values at four states; figures show the full state space | OK |
| Firms invest until the cap binds | Results table shows Invest at (0,0), (1,2), (2,1) and Wait at (4,4); figure description confirms "invests at every interior quality state, waits at the top rung" | OK |
| Off-diagonal states show a quality lead is valuable | Value-advantage figure and table rows (1,2)/(2,1) show asymmetric values of 58.87 and 78.59 | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $\omega_t$ | Equations, sentence 1 | Yes - "the quality pair $\omega_t=(q_{1t},q_{2t})$" | OK |
| $q_{it}$ | Equations, sentence 1 | Yes - defined alongside $\omega_t$ | OK |
| $Q$ | Equations, sentence 1 ($q_{it}\in\{0,\ldots,Q\}$) | Partial - numerically fixed to 4 only in Model Setup table | Defined after first use; within 50 lines and in a table, so Acceptable |
| $a_{it}$ | Equations, sentence 2 | Yes - "Each firm chooses $a_{it}\in\{0,1\}$" | OK |
| $\pi_i$ | Equations, profit display | Yes - "Flow profit uses a logit-share reduced form" | OK |
| $M$ | Equations, profit display | Partial - numerically fixed in Model Setup ($M=14$) | Acceptable; within 50 lines |
| $\eta$ | Equations, profit display | Partial - numerically fixed in Model Setup ($\eta=0.75$) | Acceptable; within 50 lines |
| $\lambda$ | Equations, profit display ($+\lambda q_i$) | Partial - numerically fixed in Model Setup ($\lambda=0.35$, "Direct quality payoff") | **Overloaded**: also used in pseudocode (Solution Method) as a damping weight; see flagged issues |
| $q_i'$ | Equations, transition display | Implicit - prime denotes next-period quality | Standard notation; OK |
| $V_i$ | Equations, prose before $G_i$ | Yes - "candidate continuation values $V_i$" | OK |
| $G_i$ | Equations, display for state-game payoff | Yes - defined inline at that display equation | OK |
| $\kappa$ | Equations, $G_i$ display | Partial - numerically fixed in Model Setup ($\kappa=2.20$) | Acceptable; within 50 lines |
| $\beta$ | Equations, $G_i$ display | Partial - numerically fixed in Model Setup ($\beta=0.90$) | Acceptable; within 50 lines |
| $P(\omega'\mid\omega,a_i,a_j)$ | Equations, $G_i$ display | Implicit - transition kernel introduced by transition probability equations | OK |
| $a^{\ast}(\omega)$ | Equations, MPE definition | Yes - "A pure-strategy Markov-perfect equilibrium is a policy $a^{\ast}(\omega)$" | OK |
| `lambda` (pseudocode) | Solution Method pseudocode, line `V_i^{n+1} = lambda T_i V^n + (1-lambda) V_i^n` | No separate definition; numerically equals 0.35 in the code | **Overloaded** with $\lambda$ from Equations; see flagged issues |

Flagged issues:
- **`lambda` overload**: In the Equations section and Model Setup table, $\lambda=0.35$ is the direct quality payoff (coefficient on $q_i$ in $\pi_i$). In the Solution Method pseudocode, `lambda` is used as the damping weight in the value-update step `V_i^{n+1} = lambda T_i V^n + (1-lambda) V_i^n`. The code (`run.py` line 80) uses the hardcoded value `0.35` for this damping weight, coincidentally matching $\lambda$, but the two are conceptually distinct objects. The pseudocode should use a different identifier (such as `alpha` or `mu`) to avoid collision with the previously defined $\lambda$.

## Summary

Both verifiable citations (Ericson-Pakes 1995 and Pakes-McGuire 1994) check out with correct volume, issue, page range, and year; 0 MAJOR issues, 0 MINOR citation issues, 1 NOT FOUND (internal lecture slides). The main message is fully supported by the tutorial's equations and results; 0 OVERREACH clauses. The single notation problem is a symbol overload: `lambda` in the Solution Method pseudocode refers to a damping weight, while $\lambda$ in the Equations and Model Setup refers to the direct quality payoff. The most important fix is renaming the pseudocode damping identifier to avoid collision with the model parameter $\lambda$.
