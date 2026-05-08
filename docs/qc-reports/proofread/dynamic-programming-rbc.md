# Proofread: dynamic-programming/rbc/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T20:00:00Z._

## Paper / Source Verification

### Kydland, F. and Prescott, E. (1982). "Time to Build and Aggregate Fluctuations." *Econometrica*, 50(6), 1345-1370.

- **Located:** https://ideas.repec.org/a/ecm/emetrp/v50y1982i6p1345-70.html
- **Tutorial claims:** (no explicit claim about paper content beyond the citation itself; the tutorial uses this as the founding RBC reference)
- **Source says:** Econometrica, volume 50, issue 6, pages 1345-1370, November 1982.
- **Verdict:** OK
- **Note:** Volume, issue, and page range are exact.

### King, R., Plosser, C., and Rebelo, S. (1988). "Production, Growth and Business Cycles: I. The Basic Neoclassical Model." *Journal of Monetary Economics*, 21(2-3), 195-232.

- **Located:** https://ideas.repec.org/a/eee/moneco/v21y1988i2-3p195-232.html
- **Tutorial claims:** (no explicit claim about paper content; cited as the neoclassical model reference)
- **Source says:** Journal of Monetary Economics, volume 21, combined issue 2-3, pages 195-232, 1988.
- **Verdict:** OK
- **Note:** Volume, combined issue, and page range are exact.

### Cooley, T. and Prescott, E. (1995). "Economic Growth and Business Cycles." In Cooley (ed.), *Frontiers of Business Cycle Research*, Princeton University Press.

- **Located:** https://press.princeton.edu/books/hardcover/9780691043234/frontiers-of-business-cycle-research
- **Tutorial claims:** Chapter titled "Economic Growth and Business Cycles" by Cooley and Prescott, in an edited volume by Cooley, Princeton University Press, 1995.
- **Source says:** Princeton University Press confirms the book was edited by Thomas F. Cooley and published in 1995; the chapter "Economic Growth and Business Cycles" by Cooley and Prescott (pages 1-38) is confirmed.
- **Verdict:** OK
- **Note:** Citation is accurate; chapter covers pages 1-38 but the tutorial does not cite a page range.

### Hansen, G. (1985). "Indivisible Labor and the Business Cycle." *Journal of Monetary Economics*, 16(3), 309-327.

- **Located:** https://ideas.repec.org/a/eee/moneco/v16y1985i3p309-327.html
- **Tutorial claims:** (no explicit claim about paper content; cited as motivation for the labor margin)
- **Source says:** Journal of Monetary Economics, volume 16, issue 3, pages 309-327, November 1985.
- **Verdict:** OK
- **Note:** Volume, issue, and page range are exact.

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 12.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** 4th edition published in 2018 by MIT Press, Chapter 12 covers RBC-related material.
- **Source says:** MIT Press confirms the 4th edition was published September 11, 2018. Chapter 12 is titled "Recursive Competitive Equilibrium: II" and covers the stochastic growth model, which is the direct antecedent to the RBC framework used in this tutorial.
- **Verdict:** OK
- **Note:** Edition year and publisher are correct; Chapter 12 addresses the stochastic growth model, consistent with citation intent.

## Main Message Audit

> "The global-grid RBC model turns a two-state productivity shock into familiar business-cycle comovements. Investment is volatile, consumption is smooth, and hours are procyclical. Capital is the persistent state that carries shocks forward. The fine-grid audit shows the moments are not driven by coarse discretization."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Investment is volatile | Results table: Std Dev 18.75%, Relative to Y 4.12 | OK |
| Consumption is smooth | Results table: Std Dev 1.54%, Relative to Y 0.34 | OK |
| Hours are procyclical | Results table: Corr with Y 0.94 | OK |
| Capital is the persistent state that carries shocks forward | Results table: Autocorr(1) = 0.95 for capital; capital is the model's state variable | OK |
| The fine-grid audit shows the moments are not driven by coarse discretization | Fine-grid audit reports max relative value error 2.1e-04 and max policy gaps (0.0461 capital, 0.0150 hours); no fine-grid business-cycle moments are computed or compared | OVERREACH |

Issues:
- The claim "the moments are not driven by coarse discretization" is not demonstrated by the fine-grid audit. The audit compares value functions and policy functions across grid resolutions but does not compute or compare business-cycle moments on the fine grid. Close policy functions are evidence of robustness, but the moments themselves are not re-tabulated for the fine grid.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k_t$ | Equations: "Capital $k_t$, labor $l_t\in(0,1)$, and TFP $z_t$" | Yes | Defined at first use |
| $l_t$ | Equations: same phrase | Yes | Defined at first use |
| $z_t$ | Equations: same phrase | Yes | Defined at first use |
| $y_t$ | Equations: LHS of production function | Yes | Defined by equation |
| $\alpha$ | Equations: production function, "$\alpha\in(0,1)$" | Partial | Identified as parameter at first use; named "Capital share in Cobb-Douglas" in Model Setup table (~47 lines later) — within 50-line window |
| $c_t$ | Equations: resource constraint | Yes | Defined by constraint context |
| $k_{t+1}$ | Equations: resource constraint | Yes | Defined by constraint context |
| $\delta$ | Equations: resource constraint "$(1-\delta)\,k_t$" | Partial | Unnamed at first use; named "Depreciation rate" in Model Setup table (~43 lines later) — within 50-line window |
| $u(c,l)$ | Equations: "Period utility uses log consumption and log leisure" | Yes | Defined at first use |
| $\phi$ | Equations: utility function, "$\phi>0$" | Partial | Identified as positive at first use; named "Leisure weight in utility" in Model Setup table (~39 lines later) — within 50-line window |
| $\beta$ | Equations: household objective | Partial | Unnamed at first use; named "Discount factor (quarterly)" in Model Setup table (~33 lines later) — within 50-line window |
| $\mathbb{E}_0$ | Equations: household objective | No explicit definition | Standard notation for expectation at $t=0$; acceptable for target audience |
| $z_L,z_H$ | Equations: TFP process, "$z_t\in\{z_L,z_H\}=\{0.95,1.05\}$" | Yes | Defined with values at first use |
| $P_{ij}$ | Equations: TFP process | Yes | Defined as conditional probability at first use |
| $P$ | Equations: transition matrix | Yes | Shown explicitly |
| $V(k,z_i)$ | Equations: Bellman equation | Yes | Defined as value function |
| $k'$ | Equations: Bellman equation (choice variable) | Yes | Introduced as next-period capital in Bellman |
| $g_k(k,z)$ | Equations: after Bellman constraint | Yes | Defined as capital policy function |
| $g_l(k,z)$ | Equations: after Bellman constraint | Yes | Defined as labor policy function |
| $k_{ss}$ | Equations: deterministic benchmark | Yes | Defined as steady-state capital |
| $l_{ss}$ | Equations: deterministic benchmark | Yes | Defined as steady-state hours |
| $c_{ss}$ | Equations: appears inside $l_{ss}$ formula | Partial | Used in the $l_{ss}$ expression before being named; defined as "steady-state consumption" in Model Setup table (~17 lines later) — within 50-line window |
| $w_{ss}$ | Equations: deterministic benchmark | Yes | Defined in the same display equation block |
| $i_{ss}$ | Model Setup table | Partial | Defined in table as "Deterministic steady-state investment"; however, investment $i$ is never given a defining equation in the Equations section |
| $TV$ | Solution Method | Yes | Defined as Bellman operator |

Flagged issues:
- **$i$ (investment) has no defining equation.** The symbol $i_{ss}$ first appears in the Model Setup table and the variable "Investment (I)" appears in the business-cycle results table, but the README's Equations section never defines $i_t$. The resource constraint is written in terms of $c_t$ and $k_{t+1}$; the identity $i_t = k_{t+1} - (1-\delta)k_t$ is implicit but unstated. The definition "$i_t = k_{t+1} - (1-\delta)k_t$" is absent from the Equations section.

## Summary

All five cited references are verified correct (volume, issue, page range, and edition year each match authoritative sources). The main message contains one OVERREACH: the claim that the fine-grid audit demonstrates moments are robust to coarse discretization is not supported because no fine-grid business-cycle moments are computed or compared — the audit validates only value functions and policy functions. The notation is largely complete, with one flagged gap: investment $i$ is reported in the results and its steady-state value $i_{ss}$ appears in the Model Setup table, but no defining equation for $i_t$ appears in the Equations section. The most important fix is adding "$i_t = k_{t+1} - (1-\delta)k_t$" to the Equations section so that the investment variable is formally introduced before it is used in the results.

Overall: 0 MAJOR, 0 MINOR (references), 1 OVERREACH, 1 notation gap.
