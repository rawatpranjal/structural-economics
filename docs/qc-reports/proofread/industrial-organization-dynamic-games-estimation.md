# Proofread: industrial-organization/dynamic-games-estimation/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:50:00Z._

## Paper / Source Verification

### Aguirregabiria, V. and Mira, P. (2007). Sequential Estimation of Dynamic Discrete Games. *Econometrica*, 75(1), 1-53.

- **Located:** https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2007.00731.x
- **Tutorial claims:** The tutorial references this as the foundational CCP estimation method for dynamic discrete games.
- **Source says:** The paper introduces pseudo maximum likelihood (PML) and nested pseudo likelihood (NPL) estimators for dynamic discrete games, addressing multiple equilibria and computational complexity. Authors are Victor Aguirregabiria and Pedro Mira. Volume 75, issue 1, pages 1-53, 2007.
- **Verdict:** OK
- **Note:** All bibliographic details verified correct.

### Bajari, P., Benkard, C. L., and Levin, J. (2007). Estimating Dynamic Models of Imperfect Competition. *Econometrica*, 75(5), 1331-1370.

- **Located:** https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2007.00796.x
- **Tutorial claims:** The tutorial references this as a key method for estimating dynamic models of imperfect competition.
- **Source says:** The paper proposes a two-step algorithm for estimating dynamic games under Markov Perfect Equilibrium assumptions. Authors are Patrick Bajari, C. Lanier Benkard, and Jonathan Levin. Volume 75, issue 5, pages 1331-1370, 2007.
- **Verdict:** OK
- **Note:** All bibliographic details verified correct.

### Pesendorfer, M. and Schmidt-Dengler, P. (2008). Asymptotic Least Squares Estimators for Dynamic Games. *Review of Economic Studies*, 75(3), 901-928.

- **Located:** https://academic.oup.com/restud/article-abstract/75/3/901/1556022
- **Tutorial claims:** The tutorial references this as a source for asymptotic least squares estimators for dynamic games.
- **Source says:** The paper presents asymptotic least squares estimators for dynamic games with finite actions, unifying Hotz-Miller (1993) and Aguirregabiria-Mira (2002) estimators. Authors, year, journal, volume, issue, and page range are all correct.
- **Verdict:** MINOR
- **Note:** The cited DOI `10.1111/j.1467-937X.2008.00497.x` is incorrect; the correct DOI is `10.1111/j.1467-937X.2008.00496.x` (last segment differs: 497 vs 496).

## Main Message Audit

> In this quality game, empirical investment rates show where firms try to catch up. CCP estimation turns those rates into continuation values and a choice likelihood. The second stage recovers payoffs without solving a new MPE for every parameter guess.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Empirical investment rates show where firms try to catch up | Results (CCP heatmaps show higher investment when rival leads) | OK |
| CCP estimation turns those rates into continuation values | Equations ($W_\theta = \bar\pi_\theta(\hat p) + \beta \hat P W_\theta$) and Solution Method | OK |
| CCP estimation turns those rates into a choice likelihood | Equations (pseudo likelihood $\ell(\theta)$ uses $v_\theta$ from CCP values) | OK |
| The second stage recovers payoffs without solving a new MPE for every parameter guess | Solution Method ("Each likelihood evaluation is a linear policy-evaluation solve, not a new equilibrium solve") and Results (parameter recovery table) | OK |

Issues:
- None. All clauses are supported by the README content.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $\omega$ | Equations, sentence 1 | Yes - "firm-view state $(q_i,q_j)$" | Clear |
| $q_i$ | Equations, sentence 1 | Yes - "own quality" | Clear |
| $q_j$ | Equations, sentence 1 | Yes - "rival quality" | Clear |
| $a_i$ | Equations, sentence 3 | Yes - "$a_i \in \{0,1\}$, where one means invest" | Clear |
| $\pi_i$ | Flow payoff equation | Yes - introduced as "flow payoff" in the preceding sentence | Clear |
| $\theta$ | Flow payoff equation (as $\pi_i(\omega,a_i;\theta)$) | Partial - components $\theta_q$, $\theta_c$, $\theta_g$ defined in Model Setup table ~25 lines later | Acceptable by 50-line rule |
| $\theta_q$ | Flow payoff equation | Partial - defined in Model Setup table ~25 lines after first appearance | Acceptable by 50-line rule |
| $\theta_c$ | Flow payoff equation | Partial - defined in Model Setup table ~25 lines after first appearance | Acceptable by 50-line rule |
| $\theta_g$ | Flow payoff equation | Partial - defined in Model Setup table ~25 lines after first appearance | Acceptable by 50-line rule |
| $p(\omega)$ | Equations, CCP sentence | Yes - "state-specific investment rate $\Pr(a_i=1\mid\omega)$" | Clear |
| $\hat P$ | Equations, CCP sentence | Yes - "policy transition" in same sentence | Clear |
| $\bar\pi_\theta$ | Equations, CCP sentence | Yes - "expected flow payoff" in same sentence | Clear |
| $\hat p$ | Equations, CCP sentence | Yes - identified as fixed CCPs in same context | Clear |
| $W_\theta$ | $W_\theta$ equation | Yes - "value under the first-stage policy" in preceding sentence | Clear |
| $\beta$ | $W_\theta$ equation | Partial - defined in Model Setup table ~16 lines after first appearance | Acceptable by 50-line rule |
| $v_\theta$ | Choice-value equation | Yes - "choice-specific values" in preceding sentence | Clear |
| $\hat p_j$ | Choice-value equation $E_{\hat p_j}[\cdot]$ | Partial - prose says "rival's first-stage CCP" before equation, but $j$ subscript is not formally bridged from $\hat p$ | Minor gap; context is clear |
| $\omega'$ | Choice-value equation | No - prime notation for next-period state not explicitly introduced | Not flagged as critical; standard DP convention |
| $\ell(\theta)$ | Pseudo likelihood equation | Yes - "second-stage pseudo likelihood" in preceding sentence | Clear |
| $d_{it}$ | Pseudo likelihood equation | **No** - never defined in the README | Flagged |
| $\Lambda$ | Pseudo likelihood equation | **No** - the logistic CDF is never defined or named | Flagged |
| $\omega_{it}$ | Pseudo likelihood equation | Partial - $i$ defined as firm index; $t$ as time index implicit throughout | Acceptable; standard convention |

Flagged issues:
- **$d_{it}$**: appears in the pseudo likelihood equation as the observed investment indicator for firm $i$ at time $t$, but is never defined in the README. The code uses `data["invest"]` but the README gives no corresponding definition.
- **$\Lambda[\cdot]$**: appears in the pseudo likelihood equation as the logistic CDF link function, but is never defined. The README mentions "logit action shock" in an earlier sentence but does not state that $\Lambda$ denotes the logistic function.

## Summary

The tutorial is well-constructed and internally consistent. Paper verification found 1 MINOR issue: the DOI for Pesendorfer and Schmidt-Dengler (2008) has a one-digit error (the cited suffix is `00497.x`; the correct suffix is `00496.x`). All other bibliographic details for all three references are correct. The main message is fully supported by the tutorial's equations, solution method, and results. Notation is mostly complete, with two flagged undefined symbols: $d_{it}$ (the observed investment indicator) and $\Lambda$ (the logistic CDF) both appear in the pseudo likelihood equation without definition anywhere in the README. The single most important fix is adding definitions for $d_{it}$ and $\Lambda$ when they are introduced in the likelihood equation.
