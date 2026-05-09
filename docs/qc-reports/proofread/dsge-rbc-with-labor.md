# Proofread: dsge/rbc-with-labor/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T05:00:00Z._

## Paper / Source Verification

### King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.

- **Located:** https://ideas.repec.org/a/eee/moneco/v21y1988i2-3p195-232.html
- **Tutorial claims:** The citation is listed as a reference for the baseline RBC framework with capital and labor.
- **Source says:** Title, authors (King, Plosser, Rebelo), year (1988), journal (Journal of Monetary Economics), volume 21, issue 2-3, pages 195-232 all match.
- **Verdict:** OK
- **Note:** No discrepancy found.

### Hansen, G. (1985). Indivisible Labor and the Business Cycle. *Journal of Monetary Economics*, 16(3), 309-327.

- **Located:** https://ideas.repec.org/a/eee/moneco/v16y1985i3p309-327.html
- **Tutorial claims:** The citation is listed as a reference for the labor supply specification used in the model.
- **Source says:** Title, author (Gary D. Hansen), year (1985), journal (Journal of Monetary Economics), volume 16, issue 3, pages 309-327 all match.
- **Verdict:** OK
- **Note:** No discrepancy found.

### Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.

- **Located:** https://ideas.repec.org/a/eee/dyncon/v24y2000i10p1405-1423.html
- **Tutorial claims:** The citation is listed as the source for the Klein QZ solution method used as the primary solver.
- **Source says:** Title, author (Paul Klein), year (2000), journal (Journal of Economic Dynamics and Control), volume 24, issue 10, pages 1405-1423 all match.
- **Verdict:** OK
- **Note:** No discrepancy found.

### Villemot, S. (2011). Solving Rational Expectations Models at First Order: What Dynare Does. *Dynare Working Paper 2*, CEPREMAP.

- **Located:** https://ideas.repec.org/p/cpm/dynare/002.html
- **Tutorial claims:** The citation is listed as confirming that the generalized Schur algorithm is used by Dynare for `stoch_simul, order=1`.
- **Source says:** Title, author (Sebastien Villemot), year (2011), publisher (CEPREMAP), working paper number 2 all match.
- **Verdict:** OK
- **Note:** No discrepancy found.

## Main Message Audit

> "A positive TFP shock raises the marginal product of inputs. Capital is mostly inherited, so hours carry much of the impact response. [...] The TFP shock splits into an hours response and a capital response. The labor rule $\hat n_t=-0.1677\hat k_{t-1}+0.4612\hat a_t$ rises with productivity and falls with inherited capital. Klein QZ delivers that rule from the stable subspace."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| A positive TFP shock raises the marginal product of inputs | Equations - Cobb-Douglas MPK is $\alpha A_t K_{t-1}^{\alpha-1} N_t^{1-\alpha}$, rising with $A_t$ | OK |
| Capital is mostly inherited, so hours carry much of the impact response | Results - Capital impact 0.108%, Labor impact 0.461% at $t=0$ | OK |
| Log-linearization gives a rational-expectations system with two states and two jump variables | Solution Method - states $(\hat k_{t-1}, \hat a_t)$, jump variables $(\hat c_t, \hat n_t)$ | OK |
| Klein QZ selects the stable path | Solution Method - Blanchard-Kahn check described and confirmed with 2 stable roots | OK |
| Investment moves more than consumption because capital carries the shock forward | Results - Investment impact 4.311% vs. Consumption impact 0.387% | OK |
| The largest gaps between linear and nonlinear appear for investment and capital | Results table - max gap for Capital 1.582 pp, Investment 5.066 pp, larger than Consumption 0.373 pp | OK |
| The labor rule has a negative capital coefficient and positive TFP coefficient | Takeaway - $-0.1677$ on $\hat k_{t-1}$, $+0.4612$ on $\hat a_t$ | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $C_t$ | Equations - "consumption $C_t$" | Yes, inline | OK |
| $N_t$ | Equations - "labor $N_t$" | Yes, inline | OK |
| $\beta$ | Equations - utility sum $\beta^t$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $\sigma$ | Equations - $C_t^{1-\sigma}$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $\psi$ | Equations - $\psi N_t^{1+\chi}$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $\chi$ | Equations - $N_t^{1+\chi}$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $I_t$ | Equations - resource constraint $C_t+I_t=Y_t$ | Implicit - "investment" named in Overview | Acceptable |
| $Y_t$ | Equations - resource constraint | Implicit - "output" named in Overview | Acceptable |
| $K_t$ | Equations - capital accumulation | Implicit - "capital" named in Overview | Acceptable |
| $\delta$ | Equations - $(1-\delta)K_{t-1}$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $A_t$ | Equations - production function $A_t K_{t-1}^\alpha$ | Implicit - context and AR(1) identify it as TFP | Acceptable |
| $\alpha$ | Equations - $K_{t-1}^\alpha N_t^{1-\alpha}$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $\rho$ | Equations - AR(1) $\rho\log A_{t-1}$ | Partial - Model Setup table ~25 lines later | Within 50-line threshold; acceptable |
| $\varepsilon_t$ | Equations - AR(1) $+\varepsilon_t$ | Partial - $\sigma_\varepsilon$ in Model Setup table names its s.d. | $\varepsilon_t$ itself does not appear in the table; its nature as i.i.d. innovation is implied but not stated |
| $\bar N$ | Model Setup table | Yes, "Steady-state hours target" | OK |
| $\sigma_\varepsilon$ | Model Setup table | Yes, "Innovation s.d." | OK |
| $\mathbb{E}_0$, $\mathbb{E}_t$ | Equations - utility, Euler equation | No explicit definition | Standard; audience would know |
| $s_t$ | Solution Method | Yes, inline as $(\hat k_{t-1},\hat a_t,\hat c_t,\hat n_t)'$ | OK |
| $\hat k_{t-1}$, $\hat a_t$, $\hat c_t$, $\hat n_t$ | Solution Method | No | Hat convention (log-deviation from steady state) is never defined anywhere in the README |
| $A$, $B$ (matrices) | Solution Method - system equation | Implicit - defined by the system $A\,\mathbb{E}_t s_{t+1}=B\,s_t$ | Acceptable |
| $x_t$ | Solution Method | Yes, inline as $(\hat k_{t-1},\hat a_t)'$ | OK |
| $y_t$ | Solution Method | Yes, inline as $(\hat c_t,\hat n_t)'$ | OK |
| $F$ | Solution Method | Yes, inline as state transition $x_{t+1}=F x_t$ | OK |
| $P$ | Solution Method | Yes, inline as decision rule $y_t=P x_t$ | OK |

Flagged issues:
- Hat notation ($\hat{\cdot}$): The hat convention for log-deviations from steady state is never defined. The README uses $\hat k_{t-1}$, $\hat a_t$, $\hat c_t$, $\hat n_t$ from the Solution Method section onward, and the Overview mentions "log-linearization," but the meaning of the hat as a log-deviation from steady state is not stated anywhere in the README.
- $\varepsilon_t$: First used in the AR(1) equation in Equations section. The Model Setup table defines $\sigma_\varepsilon$ as "Innovation s.d." but $\varepsilon_t$ itself is not explicitly labeled (e.g., as an i.i.d. shock with mean zero and s.d. $\sigma_\varepsilon$).

## Summary

All four cited references are correctly attributed with accurate titles, authors, years, journals, and page ranges. The main message is fully supported by the equations, solution method, and results. The IRF table and the labor-rule coefficients in the Takeaway follow directly from the model outputs. There are no MAJOR or MINOR reference issues, no OVERREACH or UNSUPPORTED clauses, and 0 NOT FOUND entries. The only flagged issues are two notation gaps: the hat convention ($\hat{\cdot}$ for log-deviations from steady state) is used throughout the Solution Method and Takeaway without ever being defined, and $\varepsilon_t$ is introduced in the AR(1) equation but is only partially identified through the $\sigma_\varepsilon$ table entry. The single most important fix is to define the hat notation when $s_t$ is first introduced in the Solution Method section.
