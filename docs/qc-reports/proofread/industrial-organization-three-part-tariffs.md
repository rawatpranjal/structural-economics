# Proofread: industrial-organization/three-part-tariffs/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:55:00Z._

## Paper / Source Verification

### Nevo, A., Turner, J., and Williams, J. (2016). Usage-Based Pricing and Demand for Residential Broadband. *Econometrica*, 84(2), 411-443.

- **Located:** https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA11927
- **Tutorial claims:** The tutorial cites this as the foundational empirical reference for three-part tariff structure and forward-looking broadband demand.
- **Source says:** The paper estimates demand for residential broadband from subscribers facing a three-part tariff (fixed charge plus tiered usage pricing), finds consumers respond dynamically to usage-based pricing, and simulates demand under alternative pricing structures.
- **Verdict:** OK
- **Note:** All bibliographic details - authors, year, journal, volume, issue, and page range - match the published record.

### Lecture 18 Slides 2023: Three-part tariffs and forward-looking broadband demand.

- **Located:** NOT FOUND
- **Tutorial claims:** Referenced as a source for the three-part tariff and forward-looking broadband demand framework used in this tutorial.
- **Source says:** Not applicable - no publicly accessible version located.
- **Verdict:** NOT FOUND
- **Note:** This is internal course material and cannot be independently verified.

## Main Message Audit

> "A three-part tariff changes demand before the cap is reached. Unused allowance is valuable because it can be spent later in the billing cycle. The household responds to expected overage risk. The contract menu sorts consumers by usage intensity. Low types avoid the fixed fee, middle types buy the allowance, and high types choose unlimited access."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| A three-part tariff changes demand before the cap is reached | Results (policy heat map: early-month usage cuts near cap) | OK |
| Unused allowance is valuable because it can be spent later in the billing cycle | Equations (overage cost $q_k \Delta O_k$ inside the Bellman recursion; terminal value zero) | OK |
| The household responds to expected overage risk | Equations (value function internalizes future overage cost via backward induction) | OK |
| The contract menu sorts consumers by usage intensity | Results (plan-choice summary: Metered 10%, Three-part 69%, Unlimited 21% by ascending taste) | OK |
| Low types avoid the fixed fee, middle types buy the allowance, and high types choose unlimited access | Results (plan-comparison figure and plan-choice summary table sorted by taste $h_i$) | OK |

Issues:
- None. All clauses are directly supported by the equations, solution method, or results shown in the tutorial.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $t$ | Equations, line 1 | Yes, at first use | Day index |
| $T$ | Equations, line 1 | Yes, at first use (upper bound of $t$) | Total days in billing cycle |
| $C_{t-1}$ | Equations, line 1 | Yes, at first use | Cumulative usage before day $t$ |
| $k$ | Equations, line 2 | Yes, at first use | Plan index |
| $F_k$ | Equations, line 2 | Yes, at first use | Fixed fee under plan $k$ |
| $A_k$ | Equations, line 2 | Yes, at first use | Allowance under plan $k$ |
| $q_k$ | Equations, line 2 | Yes, at first use | Overage price under plan $k$ |
| $c_t$ | Equations, line 4 | Yes, at first use | Daily usage |
| $h$ | Equations, line 4 | Yes, at first use | Consumer taste type |
| $u(c_t;h)$ | Equations, utility equation | Yes, defined by equation | Gross daily utility |
| $\psi$ | Equations, utility equation | Yes, immediately after equation | Satiation parameter |
| $C_t$ | Equations, accumulation equation | Yes, defined by equation | Cumulative usage after day $t$ |
| $\Delta O_k(C_{t-1},c_t)$ | Equations, overage equation | Yes, defined by equation | Incremental overage on day $t$ |
| $V_{k,t}(C_{t-1};h)$ | Equations, Bellman equation | Yes, defined by equation | Within-cycle value function |
| $\bar{c}$ | Equations, Bellman equation ($c_t \in [0, \bar{c}]$) | Partial - value given in Model Setup | Defined as 6 GB in Model Setup, within ~16 lines of first use |
| $g_{k,t}(C_{t-1};h)$ | Equations, after Bellman | Yes, at first use | Daily usage policy |
| $B(s_k)$ | Equations, plan-choice equation | Yes, at first use | Speed value function |
| $s_k$ | Equations, inside $B(s_k)$ | Yes, at first use implicitly; explicit in Model Setup | Speed under plan $k$ |
| $W_i(k)$ | Equations, plan-choice equation | Yes, defined by equation | Net consumer value |
| $i$ | Equations, plan-choice equation ($W_i$, $h_i$, $d_i$) | No explicit definition | Consumer/type index; appears without prior introduction |
| $h_i$ | Equations, plan-choice equation | Partial - generic $h$ defined earlier; subscript $i$ undefined | Indexed taste type; shifts from unsubscripted $h$ in utility to $h_i$ in plan choice |
| $d_i$ | Equations, plan-choice equation | Yes, defined by argmax equation | Optimal plan choice for consumer $i$ |
| $\omega_i$ | Solution Method pseudocode only | No - appears only in pseudocode as "omega_i" | Type weights; shown numerically in Model Setup table but never given a symbol in math notation |

Flagged issues:
- **$i$ (consumer index) is undefined.** The subscript $i$ first appears in the plan-choice equation ($W_i(k)$, $h_i$, $d_i$) without being introduced as a consumer or type index. The Model Setup table heading "Taste type $h_i$" uses the subscripted form but does not formally introduce $i$ either.
- **$\omega_i$ (type weights) is undefined in math notation.** The weights appear numerically in the Model Setup table under the column "Weight" and as "omega_i" in the pseudocode block, but no mathematical symbol for the weights is defined anywhere in the equations or prose.

## Summary

The tutorial is well-constructed and internally consistent. The one Nevo, Turner, and Williams (2016) paper is correctly cited with accurate bibliographic details. The second reference (Lecture 18 Slides 2023) is unverifiable course material and is recorded as NOT FOUND. The main message is fully supported by the equations and results shown. Two notation gaps warrant attention: the consumer index $i$ is used in the plan-choice equation without prior definition, and the type weights $\omega_i$ are displayed numerically in the Model Setup table but are never assigned a formal symbol in the mathematical sections. The single most important fix is to introduce $i$ as the consumer type index (and optionally assign $\omega_i$ as the corresponding weight) before the plan-choice equations.
