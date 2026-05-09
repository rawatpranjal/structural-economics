# Proofread: computational-methods/perturbation-linearization/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:10:00Z._

## Paper / Source Verification

### Blanchard, O. J. and Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations. *Econometrica*, 48(5), 1305-1311.

- **Located:** https://doi.org/10.2307/1912186
- **Tutorial claims:** Listed as a background reference for linearization of difference models under rational expectations. Bibliographic details as cited: Econometrica 48(5), 1305-1311, 1980.
- **Source says:** Title, authors, journal, volume, issue, page range, and year all confirmed correct via JSTOR/Econometrica record.
- **Verdict:** OK
- **Note:** All bibliographic fields match.

### Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.

- **Located:** https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/
- **Tutorial claims:** Listed as a background reference for perturbation and numerical methods. Bibliographic details as cited: MIT Press, 1998, ISBN 9780262100717.
- **Source says:** Title, author, publisher, year, and ISBN confirmed correct via the MIT Press page.
- **Verdict:** OK
- **Note:** All bibliographic fields match.

### Schmitt-Grohe, S. and Uribe, M. (2004). Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function. *Journal of Economic Dynamics and Control*, 28(4), 755-775.

- **Located:** https://doi.org/10.1016/S0165-1889(03)00043-5
- **Tutorial claims:** Listed as a background reference for second-order perturbation approximation in dynamic general equilibrium models. Bibliographic details as cited: JEDC 28(4), 755-775, 2004.
- **Source says:** Title, authors, journal, volume, issue, page range, and year all confirmed correct via ScienceDirect record.
- **Verdict:** OK
- **Note:** All bibliographic fields match.

## Main Message Audit

> Linearization is useful for small deviations. It also imposes symmetric responses. Higher-order perturbation adds curvature without solving the full nonlinear model. Always trace the path and check that it stays near the expansion point.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Linearization is useful for small deviations | Results table: order 1 max map error 0.0245 in local domain vs 0.38 in wide domain | OK |
| It also imposes symmetric responses | Results: "First order is symmetric by construction"; asymmetry figure shows order 1 sum near zero; equal Positive and Negative IRF RMSE for order 1 (0.0116, 0.0117) | OK |
| Higher-order perturbation adds curvature without solving the full nonlinear model | Equations section defines Taylor maps $F_1, F_2, F_3$; Solution Method shows algorithm is differentiation then iteration, not full model solution | OK |
| Always trace the path and check that it stays near the expansion point | Solution Method: "Check whether simulated paths stay near $x=0$ before trusting the approximation" | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $x_t$ | Overview, line 5 | Yes - informal in Overview, formal in Equations | Consistent throughout |
| $x$ | Equations, line 14 | Yes - "steady state normalized to $x=0$" | Used as generic argument to $F$; distinguished from $x_t$ by iteration rule |
| $F(x)$ | Equations, line 16 | Yes - defined by the polynomial formula inline | OK |
| $\rho$ | Equations, $F(x)$ formula | Partial - named in Model Setup table ~30 lines later | Within 50-line threshold; acceptable |
| $\gamma$ | Equations, $F(x)$ formula | Partial - named in Model Setup table ~30 lines later | Within 50-line threshold; acceptable |
| $\eta$ | Equations, $F(x)$ formula | Partial - named in Model Setup table ~30 lines later | Within 50-line threshold; acceptable |
| $\kappa$ | Equations, $F(x)$ formula | Partial - named in Model Setup table ~30 lines later | Within 50-line threshold; acceptable |
| $n$ | Equations, line 21 | Yes - "perturbation of order $n$" | OK |
| $F_n(x)$ | Equations, line 22 | Yes - defined by Taylor sum formula | OK |
| $F^{(j)}(0)$ | Equations, Taylor sum | Standard derivative notation, clear from context | Audience-known convention; not a flag |
| $j$ | Equations, Taylor sum | Standard summation index | OK |
| $F_1, F_2, F_3$ | Equations, aligned block | Yes - defined explicitly by their polynomial expressions | OK |
| $\epsilon$ | Equations, line 38 | Yes - "one-time shock $\epsilon$", used in $x_0 = \epsilon$ | OK |
| $x_0$ | Equations, line 41 | Yes - "initial condition" in iteration rule | OK |
| $x_{t+1}$ | Equations, line 41 | Yes - defined by iteration $x_{t+1} = F_n(x_t)$ | OK |

Flagged issues:
- None. All symbols are defined before first use or within the 50-line acceptable range. No overloaded or drifting symbols found.

## Summary

All three cited references are verified correct with no bibliographic errors. All four clauses of the takeaway are directly supported by the tutorial's equations, solution method, and results. All notation is either defined inline before first use or named in the Model Setup table within 30 lines of its first appearance in the equations, which falls within the acceptable 50-line threshold. The report finds 0 MAJOR, 0 MINOR, 0 NOT FOUND reference issues, 0 OVERREACH, 0 UNSUPPORTED takeaway clauses, and 0 flagged notation problems. The tutorial is clean and no corrections are required.
