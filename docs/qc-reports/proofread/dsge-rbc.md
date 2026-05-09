# Proofread: dsge/rbc/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T05:35:00Z._

## Paper / Source Verification

### Kydland, F. and Prescott, E. (1982). Time to Build and Aggregate Fluctuations. *Econometrica*, 50(6), 1345-1370.

- **Located:** https://ideas.repec.org/a/ecm/emetrp/v50y1982i6p1345-70.html
- **Tutorial claims:** Foundational RBC reference; the tutorial's first reference anchors the RBC research program.
- **Source says:** Finn E. Kydland and Edward C. Prescott, "Time to Build and Aggregate Fluctuations," *Econometrica* 50(6), 1345-1370, 1982. A capital-gestation model that launched the RBC program.
- **Verdict:** OK
- **Note:** Authors, year, journal, volume, issue, and page range all confirmed correct.

### King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.

- **Located:** https://ideas.repec.org/a/eee/moneco/v21y1988i2-3p195-232.html
- **Tutorial claims:** Reference for the fixed-labor neoclassical model structure used in the tutorial.
- **Source says:** Robert G. King, Charles I. Plosser, and Sergio T. Rebelo, same title, *Journal of Monetary Economics* 21(2-3), 195-232, 1988. Develops a neoclassical growth model as a framework for RBC analysis.
- **Verdict:** OK
- **Note:** Authors, year, journal, volume, issue (2-3), and page range all confirmed correct.

### Uhlig, H. (1999). A Toolkit for Analysing Nonlinear Dynamic Stochastic Models Easily. In *Computational Methods for the Study of Dynamic Economies*.

- **Located:** https://academic.oup.com/book/25747/chapter/193297574
- **Tutorial claims:** Reference for the log-linearization and undetermined-coefficients approach used in the tutorial.
- **Source says:** Harald Uhlig, "A Toolkit for Analysing Nonlinear Dynamic Stochastic Models Easily," chapter in *Computational Methods for the Study of Dynamic Economies*, ed. Marimon and Scott, Oxford University Press, 1999. Provides log-linearization methods and undetermined-coefficients solution for DSGE models.
- **Verdict:** OK
- **Note:** British spelling "Analysing" is confirmed as the canonical published-chapter spelling; year 1999 and book title are correct. An earlier 1995 working paper with "Analyzing" also circulates but the tutorial cites the published version.

### Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.

- **Located:** https://ideas.repec.org/a/eee/dyncon/v24y2000i10p1405-1423.html
- **Tutorial claims:** Reference for the Klein QZ cross-check used to validate the undetermined-coefficients solution.
- **Source says:** Paul Klein, same title, *Journal of Economic Dynamics and Control* 24(10), 1405-1423, 2000. Presents a numerically stable QZ-decomposition method for solving linear rational expectations models.
- **Verdict:** OK
- **Note:** Author, year, journal, volume, issue, and page range all confirmed correct.

## Main Message Audit

> "In this RBC model, a productivity shock raises output on impact. Investment responds strongly because the marginal product of capital is temporarily high. Consumption moves more smoothly, and capital accumulates only gradually. First-order perturbation isolates that propagation mechanism near steady state."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Productivity shock raises output on impact | Results table: Output impact = 1.0%, peak quarter = 0 | OK |
| Investment responds strongly (MPK temporarily high) | Results: Investment impact = 3.204%, largest of all variables; Equations: MPK $= \alpha A_{t+1} K_t^{\alpha-1}$ rises with $A$ | OK |
| Consumption moves more smoothly | Results table: Consumption impact = 0.323%, peak at quarter 16 vs investment peak at quarter 0 | OK |
| Capital accumulates only gradually | Results table: Capital impact = 0.08%, peak at quarter 21 | OK |
| First-order perturbation isolates propagation near steady state | Solution Method: log-linearization + undetermined coefficients; Results: nonlinear gap $\leq 0.033$ pp confirms local accuracy | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $A_t$ | Equations - "Let $A_t$ denote total factor productivity" | Yes, at first use | |
| $K_{t-1}$ | Equations - "Let... $K_{t-1}$ predetermined capital" | Yes, at first use | |
| $C_t$ | Equations - "Let... $C_t$ consumption" | Yes, at first use | |
| $I_t$ | Equations - "Let... $I_t$ investment" | Yes, at first use | |
| $Y_t$ | Equations - "Let... $Y_t$ output" | Yes, at first use | |
| $\alpha$ | Equations - production function $Y_t = A_t K_{t-1}^\alpha$ | Model Setup table, ~43 lines after first use | Acceptable: within 50-line window |
| $\delta$ | Equations - capital law $K_t = I_t + (1-\delta)K_{t-1}$ | Model Setup table, ~38 lines after first use | Acceptable: within 50-line window |
| $\beta$ | Equations - Euler equation | Model Setup table, ~28 lines after first use | Acceptable: within 50-line window |
| $\mathbb{E}_t$ | Equations - Euler equation | Standard conditional expectations operator | Not flagged; audience-standard notation |
| $\sigma$ | Equations - Euler equation $C_t^{-\sigma}$ | Model Setup table, ~28 lines after first use | Acceptable: within 50-line window; distinct from $\sigma_\varepsilon$ |
| $\rho$ | Equations - TFP process $\log A_t = \rho \log A_{t-1} + \varepsilon_t$ | Model Setup table, ~21 lines after first use | Acceptable: within 50-line window |
| $\varepsilon_t$ | Equations - TFP process | Defined inline: "$\varepsilon_t \sim N(0,\sigma_\varepsilon^2)$" | Yes, at first use |
| $\sigma_\varepsilon$ | Equations - TFP process variance $\sigma_\varepsilon^2$ | Defined inline and in Model Setup table | Yes |
| $K, Y, C, I$ (steady state) | Equations - steady-state conditions block | Introduced by "At the deterministic steady state with $A=1$"; no-subscript convention is clear | Acceptable |
| $\hat{k}_t$ | Solution Method | Defined: "$\hat k_t = \log(K_t/K)$" | Yes, at first use |
| $\hat{a}_t$ | Solution Method | Defined: "$\hat a_t = \log A_t$" | Yes, at first use; consistent with TFP process since $A=1$ at steady state |

Flagged issues:
- None.

## Summary

The `dsge/rbc` tutorial is clean. All four citations are bibliographically correct - authors, year, journal or volume, issue, and page range confirmed against authoritative sources. Every main-message clause in the Takeaway is directly supported by the Equations, Results table, and Solution Method: the impulse-response numbers, propagation narrative, and local-accuracy claim are all grounded in what the README shows. Notation is internally consistent throughout: variables introduced in the "Let..." sentence, parameters defined in the Model Setup table within 43 lines of first use (inside the 50-line acceptability window), and hat-notation symbols defined at their point of introduction in Solution Method. No undefined, late-defined, overloaded, or drifting symbols were found. Verdict: 0 MAJOR, 0 MINOR, 0 NOT FOUND, 0 OVERREACH, 0 UNSUPPORTED.
