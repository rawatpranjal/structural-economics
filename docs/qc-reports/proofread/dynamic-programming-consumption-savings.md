# Proofread: dynamic-programming/consumption-savings/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T18:00:00Z._

## Paper / Source Verification

### Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.

- **Located:** https://academic.oup.com/qje/article-abstract/112/1/1/1870884
- **Tutorial claims:** Implicit background reference for buffer-stock saving theory and the existence of a finite buffer-stock target under persistent income risk.
- **Source says:** Title, journal (QJE), year (1997), volume (112), issue (1), and pages (1-55) all confirmed from the Oxford Academic publisher record.
- **Verdict:** OK
- **Note:** Every bibliographic field is exact; no discrepancy.

---

### Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.

- **Located:** https://www.jstor.org/stable/2938366 (DOI: 10.2307/2938366, Econometric Society)
- **Tutorial claims:** Implicit background reference for the precautionary savings / liquidity-constrained household problem.
- **Source says:** Title, journal (Econometrica), year (1991), volume (59), issue (5), and pages (1221-1248) confirmed from JSTOR / Econometric Society record.
- **Verdict:** OK
- **Note:** Every bibliographic field is exact; no discrepancy.

---

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** The 4th edition (2018) is the reference for the recursive savings problem; Ch. 18 is cited.
- **Source says:** The 4th edition was published September 2018, confirming the year. However, Chapter 17 ("Self-Insurance") is the chapter that develops the income fluctuation / buffer-stock savings problem in partial equilibrium. Chapter 18 ("Incomplete Markets Models") covers the Bewley/Huggett/Aiyagari general-equilibrium extensions. The tutorial is a partial-equilibrium income fluctuation problem - the precise chapter match is Ch. 17, not Ch. 18.
- **Verdict:** MINOR
- **Note:** Edition year (2018) and publisher (MIT Press) are correct; the chapter number should be 17 (Self-Insurance) for the partial-equilibrium buffer-stock problem, not 18 (Incomplete Markets Models).

---

## Main Message Audit

> "Persistent income risk and no borrowing make saving state-contingent. Value function iteration turns the recursive choice into a policy on the asset-income grid. The computed policy has high MPCs near zero assets, positive saving after high income, and a finite buffer-stock target."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Persistent income risk and no borrowing make saving state-contingent | Equations (AR(1) log income with ρ=0.9; borrowing constraint a′≥0; policy g_a(a,z) depends on both state variables); Results (savings-policy figure shows income-state-dependent saving) | OK |
| Value function iteration turns the recursive choice into a policy on the asset-income grid | Solution Method (VFI algorithm pseudocode; convergence in 260 iterations; outputs V(a,z) and g_a(a,z)) | OK |
| The computed policy has high MPCs near zero assets | Results ("average MPC is 0.52 near zero assets" computed from numerical gradient of consumption policy) | OK |
| Positive saving after high income | Results ("High income raises saving, especially close to the borrowing limit"; savings-policy figure) | OK |
| A finite buffer-stock target | Results ("The zero crossing is the buffer-stock target for the median income state"; the net-saving figure shows a crossing) | OK |

Issues:
- No OVERREACH or UNSUPPORTED clauses. All five clauses are directly supported by sections of the README. The "finite" qualifier in the last clause is demonstrated empirically by the plotted zero crossing rather than proved analytically, but this is consistent with the tutorial's scope (numerical, not theoretical) and the parameter βR = 0.9785 < 1 shown in Model Setup implicitly satisfies the impatience condition.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $a$, $a_t$ | Equations §1 | Yes - "beginning-of-period assets" | OK |
| $z$, $z_t$ | Equations §1 | Yes - "labor income" | OK |
| $r$ | Equations §1 (inside $R=1+r$) | Implicit - named only in Model Setup table | Acceptable; $r$ is introduced by the relation $R=1+r$ even though "risk-free interest rate" label comes later |
| $R$ | Equations §1 | Yes - "gross risk-free return, $R=1+r$" | OK |
| $a'$, $a_{t+1}$ | Equations §1 | Yes - "next-period assets" | OK |
| $\underline{a}$ | Equations §2 | Yes - "no-borrowing lower bound $= 0$" | OK |
| $\bar{a}$ | Equations §2 | Partial - "upper grid bound" (numerical only) | OK; scope limited to numerics is explicit |
| $u(c)$ | Equations §3 | Yes - CRRA formula inline | OK |
| $\sigma$ | Equations §3 | Partial - constraint $\sigma>0, \sigma\neq 1$ given inline; label "CRRA risk aversion" only in Model Setup | Acceptable for a standard parameter |
| $\rho$ | Equations §4 | Partial - appears in AR(1) formula; named "persistence of log income" only in Model Setup | Same pattern as $\sigma$ |
| $\varepsilon_{t+1}$ | Equations §4 | Yes - "$\varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2)$" | OK |
| $\sigma_\varepsilon$ | Equations §4 | Yes - variance of log-income innovation | OK |
| $z_j$, $z_1,\ldots,z_J$ | Equations §5 | Yes - "income states $z_1,\ldots,z_J$" | OK |
| $P$, $P_{jk}$ | Equations §5 | Yes - "$P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$" | OK |
| $J$ | Equations §5 (upper index) | Implicit only - never assigned a name | MINOR: $J$ is the cardinality of the income grid; context makes it clear but a one-word label ("$J$ income states") would be precise |
| $\beta$ | Equations - Bellman equation | **No** - first used in Bellman equation; named "discount factor" only in Model Setup (later section) | FLAGGED: defined after first use |
| $V(a,z_j)$ | Equations - Bellman equation | Yes - defined by the Bellman equation itself | OK |
| $g_a(a,z)$ | Equations - post-Bellman | Yes - "The asset policy is $g_a(a,z)=a'$" | OK |
| $c^{\ast}(a,z)$ | Equations - post-Bellman | Yes - "The consumption policy is $c^{\ast}(a,z)=Ra+z-g_a(a,z)$" | OK |
| $T$ (operator, $TV$) | Solution Method | Yes - "The Bellman operator is $(TV)(a,z_j)=\ldots$" | OK |

Flagged issues:
- **$\beta$ defined after first use.** $\beta$ appears in the Bellman equation (Equations section) with no prior introduction. It is first labelled "Discount factor" with value 0.95 in the Model Setup table, which follows Equations. A one-line introduction in the Bellman equation paragraph ("where $\beta\in(0,1)$ is the discount factor") would resolve this.
- **$J$ never explicitly named.** The symbol $J$ serves as the count of income states throughout the Bellman and sum notation but is never introduced by name. It is inferrable from context ($z_1,\ldots,z_J$) but a brief label ("$J$ income states") would complete the definition.
- **Bellman constraint notation drifts between sections.** In the Equations section the feasible set is written $\underline{a}\leq a'\leq\bar{a},\ a'\leq Ra+z_j$ (explicit grid upper bound $\bar{a}$). In Solution Method the operator is written $\max_{0\leq a'\leq Ra+z_j}[\ldots]$ (grid bound dropped). Both are correct for different purposes but the symbol $\bar{a}$ silently disappears; a brief parenthetical in Solution Method ("the grid bound $\bar{a}$ is always slack here") would remove the ambiguity.

---

## Summary

The tutorial is clean overall: both journal citations (Carroll 1997, Deaton 1991) are bibliographically perfect, and all five clauses of the main takeaway are directly supported by the README's equations, method, and results. The single most important fix is the L&S chapter reference - **the tutorial cites Ch. 18 (Incomplete Markets Models / Bewley-Aiyagari GE) but the partial-equilibrium income fluctuation problem it solves is developed in Ch. 17 (Self-Insurance)**; this is a MINOR but misleading mismatch for a reader who turns to the book for follow-up. Two notation issues (β used before it is named "discount factor", and J never labelled) are cosmetic and easy to fix in `run.py`. The Bellman constraint drift ($\bar{a}$ in Equations, absent in Solution Method) is also minor. **Totals: 1 MINOR (citation chapter), 0 MAJOR, 0 NOT FOUND, 0 OVERREACH/UNSUPPORTED, 3 notation flags (all minor).**
