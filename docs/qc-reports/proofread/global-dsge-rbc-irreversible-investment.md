# Proofread: global-dsge/rbc-irreversible-investment/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T05:00:00Z._

## Paper / Source Verification

### Abel, A. and Eberly, J. (1996). *Optimal Investment with Costly Reversibility*. Review of Economic Studies.

- **Located:** https://academic.oup.com/restud/article-abstract/63/4/581/1520466
- **Tutorial claims:** Paper on optimal investment under costly reversibility, published in Review of Economic Studies in 1996 by Abel and Eberly.
- **Source says:** "Optimal Investment with Costly Reversibility," Review of Economic Studies, Vol. 63, No. 4, pp. 581-593, 1996, by Andrew B. Abel and Janice C. Eberly. Confirmed via Oxford Academic and NBER WP 5091.
- **Verdict:** OK
- **Note:** Title, authors, journal, and year all confirmed.

### Bertola, G. and Caballero, R. (1994). *Irreversibility and Aggregate Investment*. Review of Economic Studies.

- **Located:** https://academic.oup.com/restud/article-abstract/61/2/223/1517549
- **Tutorial claims:** Paper on irreversibility and aggregate investment, published in Review of Economic Studies in 1994 by Bertola and Caballero.
- **Source says:** "Irreversibility and Aggregate Investment," Review of Economic Studies, Vol. 61, No. 2, pp. 223-246, 1994, by Giuseppe Bertola and Ricardo Caballero. Confirmed via Oxford Academic and EconPapers.
- **Verdict:** OK
- **Note:** Title, authors, journal, and year all confirmed.

### Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.

- **Located:** https://ideas.repec.org/a/red/issued/22-86.html
- **Tutorial claims:** Paper titled "Global DSGE Models," Review of Economic Dynamics, 2023, by Cao, Luo, and Nie.
- **Source says:** The published title is "Global GDSGE Models" (GDSGE = Global Dynamic Stochastic General Equilibrium). Journal (Review of Economic Dynamics), year (2023), and authors (Dan Cao, Wenlan Luo, Guangyu Nie) are all correct.
- **Verdict:** MINOR
- **Note:** Replace "Global DSGE Models" with "Global GDSGE Models" in the reference entry.

## Main Message Audit

> "Irreversibility is a theory of bad states. It does not change the deterministic steady state here. When capital is too high for productivity, the standard model disinvests immediately. The irreversible model waits for depreciation and later low investment. Occasionally binding constraints matter because they create state-dependent kinks."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Irreversibility is a theory of bad states | Binding-region figure (Results): constraint binds only at high $K$, low $z$ | OK |
| Does not change the deterministic steady state | Equations: $I_{ss}=\delta K_{ss}>0$; stationary moments table: identical mean $K$ for both models | OK |
| Standard model disinvests immediately | Results: "standard model chooses negative investment" at low $z$, high $K$; overhang-experiment description | OK |
| Irreversible model waits for depreciation | Results: "holds investment at zero until depreciation lowers capital"; overhang-experiment figure | OK |
| Occasionally binding constraints create state-dependent kinks | Complementarity conditions $\lambda_t \geq 0$, $I_t \geq 0$, $\lambda_t I_t=0$ in Equations; policy-functions figure showing flat region at $I=0$ | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $K_t$ | Equations, opening prose | Yes - "beginning-of-period capital" | |
| $z_t$ | Equations, opening prose | Yes - "productivity" | |
| $c_t$ | Equations, opening prose | Yes - "consumption" | |
| $K_{t+1}$ | Equations, opening prose | Yes - "next-period capital" | |
| $Y_t$ | Equations, output equation | Partial - defined by $Y_t=z_tK_t^\alpha$ inline | Acceptable |
| $\alpha$ | Equations, output equation | Partial - Model Setup table within 50 lines | Acceptable |
| $\rho$ | Equations, AR(1) equation | Partial - Model Setup table within 50 lines | Acceptable |
| $\varepsilon_{t+1}$ | Equations, AR(1) equation | Yes - $\sim N(0,\sigma_\varepsilon^2)$ in same equation | |
| $\sigma_\varepsilon$ | Equations, AR(1) distribution | Partial - Model Setup table within 50 lines | Acceptable |
| $V(K,z)$ | Equations, Bellman equation | Yes - left-hand side of Bellman; implicitly defines $V$ | |
| $K'$ | Equations, Bellman choice variable | Partial - standard prime notation; not explicitly linked to $K_{t+1}$ in prose | Acceptable |
| $\Gamma(K,z)$ | Equations, Bellman constraint set | Partial - generic placeholder; resolved as $\Gamma^{std}$ and $\Gamma^{irr}$ within 15 lines | Acceptable |
| $\delta$ | Equations, Bellman utility term | Partial - Model Setup table within 50 lines | Acceptable |
| $\sigma$ | Equations, Bellman utility exponent | Partial - Model Setup table within 50 lines | Acceptable |
| $\beta$ | Equations, Bellman discount | Partial - Model Setup table within 50 lines | Acceptable |
| $P(z,z')$ | Equations, Bellman expectation sum | No - never named or defined anywhere in prose | **Undefined** |
| $z'$ | Equations, Bellman sum index | Partial - next-period productivity; implied by AR(1) context | Acceptable |
| $\Gamma^{std}(K,z)$ | Equations, standard choice set | Yes - defined explicitly with formula | |
| $\Gamma^{irr}(K,z)$ | Equations, irreversible choice set | Yes - defined explicitly with formula | |
| $I_t$ | Equations, irreversibility block | Yes - $I_t \equiv K_{t+1}-(1-\delta)K_t$ | |
| $\lambda_t$ | Equations, complementarity block | Yes - "multiplier on irreversible investment" | |
| $K_{ss}$ | Equations, steady-state remark | Partial - self-evident from "deterministic steady state" context | Acceptable |
| $Y_{ss}$, $C_{ss}$, $I_{ss}$ | Equations, calibration values | Partial - analogous subscript convention to $K_{ss}$ | Acceptable |
| $V(z,K)$ | Solution Method, pseudocode output line | - | **Drifts from Bellman's $V(K,z)$: argument order reversed** |

Flagged issues:
- $P(z,z')$: Used in the Bellman expectation sum $\sum_{z'} P(z,z')V(K',z')$ but never named or defined in prose. No sentence identifies $P(z,z')$ as the transition probability from productivity state $z$ to state $z'$.
- $V(K,z)$ vs $V(z,K)$: The Bellman equation in the Equations section writes $V(K,z)$ with capital first. The Solution Method pseudocode writes `V(z,K)` with productivity first. The argument order drifts between sections.

## Summary

The tutorial is accurate. There is 1 MINOR reference issue (Cao, Luo, and Nie 2023 title reads "Global DSGE Models" in the README but the published title is "Global GDSGE Models"), 0 MAJOR issues, and 0 NOT FOUND. The main message is fully supported by the equations, figures, and results. Two notation problems appear: the transition probability $P(z,z')$ is used in the Bellman equation without any definition or labeling, and the value function argument order drifts from $V(K,z)$ in Equations to $V(z,K)$ in the Solution Method pseudocode. Adding one defining clause for $P(z,z')$ in the Equations section is the single most important fix.
