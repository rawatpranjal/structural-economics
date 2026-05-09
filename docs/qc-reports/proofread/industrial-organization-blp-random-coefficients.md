# Proofread: industrial-organization/blp-random-coefficients/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:20:00Z._

## Paper / Source Verification

### Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*, 63(4), 841-890.

- **Located:** https://ideas.repec.org/a/ecm/emetrp/v63y1995i4p841-90.html
- **Tutorial claims:** Foundational BLP paper estimating differentiated-products demand; the contraction mapping and IV/GMM approach originate here.
- **Source says:** Title "Automobile Prices in Market Equilibrium," Econometrica, vol. 63, issue 4, 1995, pp. 841-890, authors Steven Berry, James Levinsohn, Ariel Pakes. Confirmed via IDEAS/EconPapers and NBER working paper w4264.
- **Verdict:** OK
- **Note:** All bibliographic fields match exactly.

### Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2), 242-262.

- **Located:** https://ideas.repec.org/a/rje/randje/v25y1994isummerp242-262.html
- **Tutorial claims:** Berry (1994) provides the share inversion method used inside the BLP contraction.
- **Source says:** Title "Estimating Discrete-Choice Models of Product Differentiation," RAND Journal of Economics, vol. 25, Summer 1994 (issue 2), pp. 242-262, author Steven T. Berry. Confirmed via IDEAS and a Caltech PDF.
- **Verdict:** OK
- **Note:** All bibliographic fields match exactly.

### Nevo, A. (2000). "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand." *Journal of Economics & Management Strategy*, 9(4), 513-548.

- **Located:** https://ideas.repec.org/a/bla/jemstr/v9y2000i4p513-548.html
- **Tutorial claims:** Nevo (2000) gives practical implementation guidance for BLP-style random-coefficients estimation.
- **Source says:** Title "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand," Journal of Economics & Management Strategy, vol. 9, issue 4, 2000, pp. 513-548, author Aviv Nevo. DOI: 10.1111/j.1430-9134.2000.00513.x. Confirmed via IDEAS and Wiley Online Library.
- **Verdict:** OK
- **Note:** All bibliographic fields match exactly.

## Main Message Audit

> "BLP changes the estimated substitution object. The contraction lets each candidate $\sigma$ fit observed shares. IV/GMM chooses heterogeneity using moments for recovered unobserved quality. With heterogeneity, substitution no longer has to follow existing shares."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| BLP changes the estimated substitution object | Results: cross-price elasticity matrix shows different off-diagonal pattern vs. plain logit | OK |
| The contraction lets each candidate $\sigma$ fit observed shares | Equations: contraction update $\delta^{(r+1)}_{jt} = \delta^{(r)}_{jt} + \log s^{\text{obs}}_{jt} - \log s^{\text{pred}}_{jt}$ | OK |
| IV/GMM chooses heterogeneity using moments for recovered unobserved quality | Equations: identifying moments $E[Z_{jt}\xi_{jt}]=0$; Solution Method: GMM objective Q | OK |
| With heterogeneity, substitution no longer has to follow existing shares | Results: cross-price elasticity matrix figure; cross-logit IIA column uniformity vs. BLP variation | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $i$ | Equations, line 1 | Yes - "Consumer $i$" | OK |
| $t$ | Equations, line 1 | Yes - "in market $t$" | OK |
| $j$ | Equations, line 3 | Yes - "inside product $j$" | OK |
| $J$ | Equations, line 1 | Yes - "among $J$ inside goods" | OK |
| $u_{ijt}$ | Equations, utility eq | Yes - LHS of utility equation | OK |
| $\beta_0$ | Equations, utility eq | Partial - Model Setup table (~30 lines later) | Acceptable per 50-line rule |
| $\beta_x$ | Equations, utility eq | Partial - Model Setup table | Acceptable per 50-line rule |
| $x_{jt}$ | Equations, utility eq | Yes - "observed product characteristic" inline | OK |
| $\alpha$ | Equations, utility eq | Partial - Model Setup table | Acceptable per 50-line rule |
| $p_{jt}$ | Equations, utility eq | Yes - "$p_{jt}$ is price" inline | OK |
| $\xi_{jt}$ | Equations, utility eq | Yes - "unobserved quality" inline | OK |
| $\sigma_x$ | Equations, utility eq | Partial - Model Setup table | Acceptable per 50-line rule |
| $\nu_{i1}$ | Equations, utility eq | Partial - "$\nu_i \sim N(0,I)$" identifies the vector; subscript 1 denotes first component | OK |
| $\sigma_p$ | Equations, utility eq | Partial - Model Setup table | Acceptable per 50-line rule |
| $\nu_{i2}$ | Equations, utility eq | Partial - same as $\nu_{i1}$; second component of $\nu_i$ | OK |
| $\varepsilon_{ijt}$ | Equations, utility eq | Yes - "Type-I extreme value" inline | OK |
| $\delta_{jt}$ | Equations, decomposition eq | Yes - LHS of decomposition | OK |
| $\mu_{ijt}$ | Equations, decomposition eq | Yes - RHS of decomposition | OK |
| $\sigma$ | Equations, share formula | Yes - "$\sigma=(\sigma_x,\sigma_p)$" inline | OK |
| $ns$ | Equations, share formula | Late - Model Setup table (~23 lines later) | Acceptable per 50-line rule; meaning is self-evident from $\frac{1}{ns}\sum_{i=1}^{ns}$ |
| $s_{jt}$ | Equations, share formula | Yes - LHS of share equation | OK |
| $k$ | Equations, share formula | Partial - summation dummy; no explicit introduction | Acceptable; standard summation convention |
| $s^{\text{obs}}_{jt}$ | Equations, contraction eq | Yes - "equal observed shares" inline | OK |
| $s^{\text{pred}}_{jt}$ | Equations, contraction eq | Yes - "predicted shares" inline | OK |
| $r$ | Equations, contraction eq | Partial - superscript iteration counter; not introduced in text | Acceptable; standard fixed-point iteration convention |
| $X_{jt}$ | Equations, linear demand eq | Yes - "$X_{jt}=(1,x_{jt},p_{jt})$" inline | OK |
| $\theta_1$ | Equations, linear demand eq | Undefined | See flagged issues |
| $Z_{jt}$ | Equations, moments | Partial - "The instruments" in prose; role described but vector contents not listed here | Acceptable; contents described in Solution Method pseudocode |

Flagged issues:
- $\theta_1$: introduced in the linear demand equation $\delta_{jt} = X_{jt}\theta_1 + \xi_{jt}$ without an explicit statement that $\theta_1 = (\beta_0, \beta_x, \alpha)$. The decomposition is recoverable by comparing this equation with the earlier $\delta_{jt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt}$, and the code confirms it (line 446: `beta_0_hat, beta_x_hat, alpha_hat = theta_1_hat`), but the README never writes the identification out explicitly. Adding a brief parenthetical such as "$\theta_1 = (\beta_0, \beta_x, \alpha)$" after the equation would close the gap.

## Summary

All three cited references verified without error; bibliographic details are exact. The main message is fully supported by the Equations, Solution Method, and Results sections, with no overreach or unsupported clauses. The only substantive issue is a single notation gap: $\theta_1$ is introduced in the linear demand equation but its composition as $(\beta_0, \beta_x, \alpha)$ is never stated explicitly - the reader must infer it by comparing two equations. Overall verdict: 0 MAJOR, 0 MINOR paper issues, 0 NOT FOUND, 0 OVERREACH, 1 undefined symbol ($\theta_1$). The single most important fix is adding "$\theta_1 = (\beta_0, \beta_x, \alpha)$" inline when the linear demand equation is introduced.
