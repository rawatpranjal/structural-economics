# Proofread: industrial-organization/vertical-contracts/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:55:00Z._

## Paper / Source Verification

### Conlon, C., and Mortimer, J. (2021). JPE article on vertical contracts in vending-machine markets.

- **Located:** https://www.journals.uchicago.edu/doi/abs/10.1086/716563
- **Tutorial claims:** A 2021 JPE article on vertical contracts in vending-machine markets.
- **Source says:** "Efficiency and Foreclosure Effects of Vertical Rebates: Empirical Evidence," Journal of Political Economy, Vol. 129, No. 12 (2021), pp. 3357-3404. Uses vending-machine industry data to study rebate contracts that create competing efficiency and foreclosure effects.
- **Verdict:** OK
- **Note:** The title is not given; the description "vertical contracts in vending-machine markets" is accurate.

### Hristakeva, S. (2022). JPE article on vertical contracts and product selection in retail markets.

- **Located:** https://www.journals.uchicago.edu/doi/10.1086/720631
- **Tutorial claims:** A 2022 JPE article on vertical contracts and product selection in retail markets.
- **Source says:** "Vertical Contracts with Endogenous Product Selection: An Empirical Analysis of Vendor Allowance Contracts," Journal of Political Economy, Vol. 130, No. 12 (2022), pp. 3202-3252. Studies vendor allowance contracts and how they distort which products retailers choose to stock.
- **Verdict:** OK
- **Note:** Year, journal, and topic all match.

### Lecture 9 Slides 2023: Vertical contracts, vending assortments, and slotting fees.

- **Located:** NOT FOUND
- **Tutorial claims:** Course lecture slides covering vertical contracts, vending assortments, and slotting fees.
- **Source says:** N/A - internal course material; no public URL available.
- **Verdict:** NOT FOUND
- **Note:** Internal course material cannot be verified; no public source exists.

## Main Message Audit

> "Vertical contracts can change availability without large retail price movement. In this example, rebates and slotting fees move scarce slots toward Mars. Empirical work on vertical contracts needs product availability and transfers."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Vertical contracts can change availability without large retail price movement | Results table: Mars slots shift from 4 to 5 under both non-standard contracts; average prices move from 1.91 to 1.82-1.88, a range of 0.09 | OK |
| Rebates and slotting fees move scarce slots toward Mars | Results table: both the all-unit discount and slotting-fee contract raise Mars slots from 4 to 5 | OK |
| Empirical work on vertical contracts needs product availability and transfers | Nothing in Equations, Solution Method, or Results; the tutorial is a stylized computational model, not an empirical study | OVERREACH |

Issues:
- "Empirical work on vertical contracts needs product availability and transfers" is an OVERREACH: the tutorial demonstrates a theoretical/computational result about a stylized vending model but does not conduct or evaluate any empirical methodology, so the claim about what empirical work "needs" is not demonstrated.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $\mathcal{J}$ | Equations, sentence 1 | Yes, at first use | "Let $\mathcal J$ be the product catalog" |
| $K$ | Equations, sentence 1 | Yes, at first use | "let $K$ be the number of vending slots" |
| $j$ | Equations, sentence 2 | Yes, implicitly at first use | "Product $j$ has..." |
| $a_j$ | Equations, sentence 2 | Yes, at first use | "demand intercept $a_j$" |
| $c_j$ | Equations, sentence 2 | Yes, at first use | "marginal cost $c_j$" |
| $m(j)$ | Equations, sentence 2 | Yes, at first use | "manufacturer label $m(j)$" |
| $b$ | Equations, demand equation | Late - defined in Model Setup table (~54 lines after first use) | Model Setup: "common slope $b=4$" |
| $p_j$ | Equations, demand equation | Yes, implicitly (retail price argument) | Used as argument to demand function |
| $q_j(p_j)$ | Equations, demand equation | Yes, defined by equation | Demand function |
| $w_j$ | Equations, retail pricing paragraph | Yes, at first use | "Given wholesale price $w_j$" |
| $p_j^{\ast}(w_j)$ | Equations, retail pricing equation | Yes, defined by argmax | Optimal retail price |
| $C$ | Equations, contract paragraph | Yes, at first use | "Contract $C$ maps assortment $A$..." |
| $A$ | Equations, contract paragraph | Yes, constrained in argmax same paragraph | Assortment subset |
| $F_j^C(A)$ | Equations, contract paragraph | Yes, at first use | "upstream side pays $F_j^C(A)$ to the retailer" |
| $A_C^{\ast}$ | Equations, assortment argmax | Yes, defined by argmax | Optimal assortment under contract $C$ |
| $\Pi_C^U(A)$ | Equations, upstream payoff equation | Yes, defined by equation | Upstream payoff |
| $\tau$ | Equations, all-unit discount paragraph | Yes, at first use | "at least $\tau$ Mars products" |
| $\mu$ | Equations, discount equation | Late - defined in Model Setup table (~12-17 lines after first use) | Model Setup: "Per-unit margin $\mu=0.42$"; within 50-line threshold, Acceptable |
| $d$ | Equations, discount equation | Late - defined in Model Setup table (~12-17 lines after first use) | Model Setup: "Mars margin falls by $d=0.18$"; within 50-line threshold, Acceptable |
| $M(A)$ | Equations, discount equation | Yes, in same equation block | $M(A)=\sum_{j\in A}\mathbf{1}\{m(j)=\text{Mars}\}$ |
| $\Pi^D_C(A)$ | Solution Method, pseudocode | Never defined | Appears in pseudocode ("Pi^D_C(A)") but has no counterpart in the Equations section |

Flagged issues:
- $b$ (demand slope): first used in the demand equation in the Equations section, defined in the Model Setup table approximately 54 lines later - slightly beyond the 50-line Acceptable threshold for late-defined symbols.
- $\Pi^D_C(A)$: appears in the Solution Method pseudocode as `Pi^D_C(A)` (the retailer's objective), but the Equations section never assigns a named symbol to the retailer's aggregate payoff. The upstream analogue $\Pi_C^U(A)$ is defined, but no $\Pi^D_C(A)$ equation appears anywhere in the README.

## Summary

The tutorial is well-structured and the code matches the prose closely. Paper verification found no substantive errors: both Conlon-Mortimer (2021) and Hristakeva (2022) have the correct year and journal, and the descriptions are accurate. The lecture-slide reference cannot be verified. The main message has one OVERREACH - the claim that "empirical work on vertical contracts needs product availability and transfers" is not demonstrated by the tutorial's stylized computational model. Notation is largely clean, with two flagged items: $b$ is used in the Equations section before being defined in the Model Setup table (54 lines later, just past the 50-line Acceptable threshold), and $\Pi^D_C(A)$ appears in the Solution Method pseudocode without ever being formally defined in the Equations section. Overall: 0 MAJOR, 0 MINOR paper issues, 1 NOT FOUND, 1 OVERREACH, 2 notation flags; the most important fix is adding a definition of $\Pi^D_C(A)$ (or the retailer objective symbol) in the Equations section to match its use in the pseudocode.
