# Proofread: structural-econometrics/dcegm-retirement-saving/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T08:30:00Z._

## Paper / Source Verification

### Iskhakov, F., Jorgensen, T. H., Rust, J., and Schjerning, B. (2017). The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with or without Taste Shocks. *Quantitative Economics*, 8(2), 317-365.

- **Located:** https://ideas.repec.org/a/wly/quante/v8y2017i2p317-365.html
- **Tutorial claims:** The paper introduces DC-EGM and handles discrete-continuous dynamic choice models with or without taste shocks, published in Quantitative Economics 8(2), 317-365 (2017).
- **Source says:** Authors are Fedor Iskhakov, Thomas H. Jorgensen (spelled Jorgensen on REPEC), John Rust, Bertel Schjerning. Title: "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." Quantitative Economics 8(2), 317-365, 2017. The DOI 10.3982/QE643 resolves correctly to Wiley Online Library.
- **Verdict:** MINOR
- **Note:** The second author's name should be "Jorgensen" with the ø diacritic: "Jorgensen" is missing "T. H. Jorgensen" should be "T. H. Jorgensen" - specifically, the author's name is "Jørgensen" not "Jorgensen"; and the published title uses parentheses "with (or without) taste shocks" rather than "with or without taste shocks." The DOI is correct.

### Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.

- **Located:** https://ideas.repec.org/a/eee/ecolet/v91y2006i3p312-320.html
- **Tutorial claims:** Carroll (2006) presents the EGM method for solving dynamic stochastic optimization problems, published in Economics Letters 91(3), 312-320.
- **Source says:** Author: Christopher D. Carroll. Title: "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." Economics Letters, volume 91, issue 3, pages 312-320, 2006. The DOI 10.1016/j.econlet.2005.09.013 resolves correctly.
- **Verdict:** OK
- **Note:** All bibliographic fields match. Title capitalization difference (tutorial uses title case; journal uses sentence case) is cosmetic only.

## Main Message Audit

> "A plain grid search treats every current asset and every next asset as a nested maximization. DC-EGM avoids that inner search. It solves the continuous saving problem separately for work and retirement, then keeps the upper envelope of the choice-specific value functions." (Overview) and "DC-EGM is useful when a structural labor model combines a discrete margin with continuous saving. Each branch remains an Euler-equation problem, so EGM avoids the inner root search or grid search. The discrete retirement option then enters through the upper envelope. That envelope is the economic policy boundary and the numerical source of the kink." (Takeaway)

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| DC-EGM avoids the nested inner grid search | Equations: Euler inversion $c_{t,i}^d = (\beta R \mu_{t+1}^{m'(d)}(a_i^{+}))^{-1/\gamma}$ replaces a search over next assets | OK |
| Solves the continuous saving problem separately for work and retirement | Equations: separate branch Bellman $V_t^d(a)$ for each $d$; Solution Method: SOLVE_BRANCH subroutine called once per branch | OK |
| Keeps the upper envelope of choice-specific value functions | Equations: $V_t^0(a) = \max\lbrace V_t^{\mathrm{work}}(a), V_t^{\mathrm{retire}}(a)\rbrace$; Solution Method pseudocode explicitly computes and selects the max | OK |
| Each branch remains an Euler-equation problem | Equations: the Euler condition $u'(c_t^d(a^{+})) = \beta R \,\partial V_{t+1}^{m'(d)}(a^{+})/\partial a^{+}$ is stated and inverted per branch | OK |
| The discrete retirement option enters through the upper envelope | Equations: $d_t^{\ast}(a) = \arg\max_d V_t^d(a)$ and the value envelope equation are explicit | OK |
| The envelope is the economic policy boundary and numerical source of the kink | Results: retirement-boundary figure shows the asset threshold created by the envelope; branch-consumption figure shows the kink at the switch point | OK |

Issues:
- None. All clauses are supported by the Equations, Solution Method, or Results sections.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $t$ | Equations, first line | Yes - Equations prose and Model Setup table | Period index (ages 55-70) |
| $a_t$ | Equations, first line | Yes - Equations prose and Model Setup table | Assets at start of age $t$ |
| $m_t$ | Equations, first line | Yes - Equations prose | Absorbing retirement status $\in \lbrace 0,1\rbrace$ |
| $d_t$ | Equations, first paragraph | Yes - Equations prose | Discrete labor-supply choice |
| $D(m_t)$ | Equations, first display | Yes - defined inline with $D(0)$ and $D(1)$ | Feasible choice set |
| $m'(d)$ | Equations, absorbing-status display | Yes - defined inline | Next-period retirement status transition |
| $u(c)$ | Equations, CRRA display | Yes - defined inline | CRRA utility |
| $u'(c)$ | Equations, CRRA display | Yes - defined inline | Marginal utility |
| $\gamma$ | Equations, CRRA display | Partial - Model Setup table within ~20 lines | CRRA curvature (2.0) |
| $\beta$ | Equations, branch value display | Partial - Model Setup table within ~30 lines | Discount factor (0.96) |
| $\omega_B$ | Equations, terminal value display | Partial - Model Setup table within ~5 lines | Terminal bequest weight (1.15) |
| $\bar b$ | Equations, terminal value display | Partial - Model Setup table within ~5 lines | Bequest utility floor (1.0) |
| $V_T^m(a)$ | Equations, terminal value display | Yes - defined inline | Terminal value function |
| $T$ | Equations, terminal value display | Partial - Model Setup table implies $T$ = age 70; pseudocode "ages t = 0,...,T-1" | Standard terminal-period index; acceptable |
| $\alpha_t$ | Equations, income equations | Yes - defined inline: "$\alpha_t = 55 + t$" | Calendar age |
| $y_t(\mathrm{work})$ | Equations, income display | Yes - defined by formula | Working income |
| $\bar y^R$ | Equations, income display | Partial - Model Setup table row $y_t(\mathrm{retire}) = 0.78$ within ~10 lines | Pension income constant; symbol not glossed in prose |
| $y_t(\mathrm{retire})$ | Equations, income display | Yes - equation sets it equal to $\bar y^R$ | Retirement income |
| $\psi_t(\mathrm{work})$ | Equations, income display | Yes - defined by formula | Work disutility (negative) |
| $\chi_R$ | Equations, income display | Partial - Model Setup table row $\psi_t(\mathrm{retire}) = 0.00$ within ~10 lines | Retirement amenity constant; symbol not glossed in prose |
| $\psi_t(\mathrm{retire})$ | Equations, income display | Yes - equation sets it equal to $\chi_R$ | Retirement amenity |
| $R$ | Equations, budget constraint | Partial - Model Setup table ($R = 1+r$, 1.02) within ~5 lines | Gross asset return |
| $\underline a$ | Equations, budget constraint | Partial - Model Setup table (0.0) within ~5 lines | Borrowing limit on next assets |
| $c_t$ | Equations, budget constraint | Yes - budget equation defines residual consumption | Consumption |
| $a_{t+1}$ | Equations, budget constraint | Yes - budget equation context | Next-period assets (budget constraint notation) |
| $a^{+}$ | Equations, branch value display | Introduced as optimization variable in $\max_{a^{+} \geq \underline a}$; prose definition given ~10 lines later ("Write $a_i^{+}$ for...") | **Drifts from $a_{t+1}$** - see flagged issues |
| $V_t^d(a)$ | Equations, branch value display | Yes - defined as the branch Bellman | Branch value function |
| $c_t^d(a,a^{+})$ | Equations, branch objects display | Yes - defined inline as $Ra + y_t(d) - a^{+}$ | Branch consumption (two-argument form) |
| $\widetilde V_t^d(a,a^{+})$ | Equations, branch objects display | Yes - defined inline | Branch value at candidate $(a, a^{+})$ (two-argument form) |
| $V_t^0(a)$, $V_t^1(a)$ | Equations, upper envelope display | Yes - defined as active and retired value functions | Active/retired value functions |
| $a_i^{+}$ | Equations, EGM Euler inversion | Yes - introduced by prose "Write $a_i^{+}$ for a candidate next-period asset point" | Exogenous grid point for next-period assets |
| $\mu_{t+1}^{m'(d)}(a_i^{+})$ | Equations, EGM Euler inversion | Yes - defined in display as $\partial V_{t+1}^{m'(d)}(a_i^{+})/\partial a^{+}$ | Next-period marginal value |
| $c_{t,i}^d$ | Equations, EGM consumption display | Yes - defined as inverse marginal utility | Consumption implied by Euler inversion at grid point $i$ |
| $a_{t,i}^{\mathrm{endo},d}$ | Equations, EGM endogenous asset display | Yes - defined by formula | Endogenous current asset from EGM |
| $\widetilde V_t^d(a_{t,i}^{\mathrm{endo},d})$ | Equations, EGM value curve display | Implicit - one-argument form used without noting that $a^{+} = a_i^{+}$ is fixed | **Argument change from two-argument definition** - see flagged issues |
| $g_t^d(a)$ | Equations, interpolation display | Yes - defined as interpolated saving policy | Branch saving/next-asset policy |
| $c_t^d(a)$ | Equations, constraint-binding display | Implicit - one-argument form appears without noting $a^{+} = \underline a$ | **Argument change from two-argument definition** - see flagged issues |
| $d_t^{\ast}(a)$ | Equations, final policies display | Yes - defined as $\arg\max_d V_t^d(a)$ | Optimal discrete choice |
| $g_t^0(a)$, $c_t^0(a)$ | Equations, final policies display | Yes - defined as policies from the winning branch | Active saving and consumption policies |

Flagged issues:
- **$a_{t+1}$ vs $a^{+}$ drift (Equations section):** The budget constraint uses $a_{t+1}$ for next-period assets. The branch Bellman and all EGM equations then switch to $a^{+}$ (and $a_i^{+}$ for grid-indexed points) for the same object. The README never states $a^{+} \equiv a_{t+1}$. These are used for the same quantity across sections without an explicit transition.
- **$\widetilde V_t^d$ argument change (Equations section):** The branch objects block defines $\widetilde V_t^d(a, a^{+})$ with two arguments. The EGM value-curve display then writes $\widetilde V_t^d(a_{t,i}^{\mathrm{endo},d})$ with one argument, implicitly fixing $a^{+} = a_i^{+}$. The one-argument form is used in the interpolation step without noting the collapsed argument.
- **$c_t^d$ argument change (Equations section):** The branch objects block defines $c_t^d(a, a^{+})$ with two arguments. The constraint-binding display and the final policies display then write $c_t^d(a)$ with one argument (implicitly fixing $a^{+}$ to either $\underline a$ or the policy optimum). The one-argument form is used without noting the collapsed argument.

## Summary

The tutorial is well-constructed. Both references locate correctly; the only paper-verification finding is MINOR (the second author's name should be "Jorgensen" with the ø diacritic - "Jørgensen" - and the published title includes parentheses around "or without"). The main message is fully supported: every clause in the Overview and Takeaway maps to explicit equations or results in the body. The notation has no undefined symbols and no overloading, but three drift issues appear in the Equations section: (1) next-period assets are written $a_{t+1}$ in the budget constraint then $a^{+}$ throughout the EGM derivation without an explicit equivalence statement, and (2) and (3) the branch functions $\widetilde V_t^d$ and $c_t^d$ are defined with two arguments $(a, a^{+})$ but then used with one argument in the EGM value-curve, constraint-binding, and final-policies displays without noting that the second argument is fixed. Overall verdict: 1 MINOR citation issue, 3 notation drift issues, 0 MAJOR, 0 NOT FOUND, 0 OVERREACH. The single most important fix is clarifying the $a_{t+1}$ to $a^{+}$ notation transition, since it spans the widest gap in the Equations section.
