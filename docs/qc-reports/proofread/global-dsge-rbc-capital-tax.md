# Proofread: global-dsge/rbc-capital-tax/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:10:00Z._

## Paper / Source Verification

### Chamley, C. (1986). *Optimal Taxation of Capital Income in General Equilibrium*. Econometrica.

- **Located:** https://www.econometricsociety.org/publications/econometrica/1986/05/01/optimal-taxation-capital-income-general-equilibrium-infinite
- **Tutorial claims:** Cited as background for capital income taxation in general equilibrium.
- **Source says:** Full title is "Optimal Taxation of Capital Income in General Equilibrium with Infinite Lives" (Econometrica 54(3), pp. 607-622, 1986).
- **Verdict:** MINOR
- **Note:** The tutorial omits the qualifying phrase "with Infinite Lives" from the title.

### Judd, K. (1985). *Redistributive Taxation in a Simple Perfect Foresight Model*. JPE.

- **Located:** https://ideas.repec.org/a/eee/pubeco/v28y1985i1p59-83.html
- **Tutorial claims:** Cited alongside Chamley as background for capital income taxation results.
- **Source says:** The paper appears in the Journal of Public Economics, Vol. 28, No. 1, pp. 59-83, not the Journal of Political Economy. The title is correct.
- **Verdict:** MINOR
- **Note:** "JPE" conventionally refers to the Journal of Political Economy; the correct abbreviation for Journal of Public Economics would be "JPubE" or the full name.

### Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.

- **Located:** https://ideas.repec.org/a/red/issued/22-86.html
- **Tutorial claims:** Cited for global DSGE model methodology used in the tutorial.
- **Source says:** The actual title is "Global GDSGE Models" (Review of Economic Dynamics, Vol. 51, December 2023, pp. 199-225). GDSGE is the specific algorithmic framework introduced by the paper.
- **Verdict:** MINOR
- **Note:** Title should read "Global GDSGE Models"; the GDSGE label distinguishes the paper's specific computational framework.

## Main Message Audit

> "The rebate balances the government budget while the intertemporal wedge remains. Once the household prices saving with $(1-\tau_k)MPK$, the economy carries less capital into every productivity state. The steady state gives the clean long-run comparison. The global policy functions show the same force away from steady state."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| The rebate balances the government budget while the intertemporal wedge remains | Equations: resource constraint unchanged; Euler equation retains $(1-\tau_k)$ wedge | OK |
| Once the household prices saving with $(1-\tau_k)MPK$, the economy carries less capital into every productivity state | Results: policy-by-tax figure shows lower capital policy for all current-capital values; results table confirms lower $K_{ss}$ and lower mean simulated capital at every tax rate | OK |
| The steady state gives the clean long-run comparison | Equations: closed-form $K_{ss}(\tau_k)$ derived; Results: exact vs. simulated table provided | OK |
| The global policy functions show the same force away from steady state | Results: policy-by-tax figure shows downward shift in capital policy at all grid points for every nonzero tax rate | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $K_t$ | Equations, line 1 of section | Yes, inline prose | "aggregate capital at the start of period $t$" |
| $z_t$ | Equations, line 1 of section | Yes, inline prose | "aggregate TFP" |
| $c_t$ | Equations, line 1 of section | Yes, inline prose | "consumption" |
| $K_{t+1}$ | Equations, line 1 of section | Yes, inline prose | "next-period capital" |
| $\beta$ | Equations, utility sum | Partial | Named in Model Setup table ~35 lines later; within 50-line window |
| $\sigma$ | Equations, utility exponent (with $\sigma>0$) | Partial | Named in Model Setup table; within 50-line window |
| $Y_t$ | Equations, Cobb-Douglas line | Yes, inline | "Cobb-Douglas output $Y_t = z_t K_t^\alpha$" |
| $\alpha$ | Equations, Cobb-Douglas exponent | Partial | Named in Model Setup table; within 50-line window |
| $\rho$ | Equations, AR(1) coefficient | Partial | Named in Model Setup table; within 50-line window |
| $\varepsilon_{t+1}$ | Equations, AR(1) shock | Yes, inline | Defined by $\sim N(0, \sigma_\varepsilon^2)$ immediately |
| $\sigma_\varepsilon$ | Equations, shock variance | Partial | Named in Model Setup table; within 50-line window |
| $\delta$ | Equations, resource constraint | Partial | Named in Model Setup table; within 50-line window |
| $\tau_k$ | Equations, Euler equation | Partial | Named in Model Setup table; within 50-line window |
| $K_{ss}(\tau_k)$ | Equations, steady-state formula | Yes, inline | "exact deterministic steady state" at $z=1$ |
| $Y_{ss}$, $C_{ss}$, $T_{ss}$ | Equations, steady-state expressions | Yes, inline | All defined in same paragraph as $K_{ss}$ |
| $V(z,K)$ | Solution Method pseudocode | Yes, in pseudocode output line | "value" |
| $g_K(z,K)$ | Solution Method pseudocode | Yes, in pseudocode output line | "saving rule" |
| $g_c(z,K)$ | Solution Method pseudocode | Yes, in pseudocode output line | "consumption rule" |
| $K'$ | Model Setup table and pseudocode | Yes, in pseudocode | Standard DP shorthand for $K_{t+1}$; two notations used for the same object |
| $MPK$ | Takeaway prose | Not explicitly | Standard abbreviation; target audience would recognize it as $\alpha z K^{\alpha-1}$ |

Flagged issues:
- None. All parameters without inline prose definitions appear in the Model Setup table within 50 lines of first use (Acceptable). The dual notation $K_{t+1}$ (formal equations) and $K'$ (pseudocode) is standard dynamic programming convention and not an error.

## Summary

The tutorial is in good shape. The notation is internally consistent and complete, the main message is fully supported by the equations and results shown, and the numerical claims in the Results section are dynamically generated from the computation rather than hardcoded. Three MINOR paper-verification issues were found: Chamley (1986) has the full title "Optimal Taxation of Capital Income in General Equilibrium with Infinite Lives" (the "with Infinite Lives" qualifier is missing); Judd (1985) appears in the Journal of Public Economics, not the Journal of Political Economy as implied by the "JPE" abbreviation; and the Cao, Luo, and Nie (2023) paper is titled "Global GDSGE Models," not "Global DSGE Models." The most important fix is the Judd (1985) journal label, since "JPE" is the well-established abbreviation for the Journal of Political Economy, a different publication.
