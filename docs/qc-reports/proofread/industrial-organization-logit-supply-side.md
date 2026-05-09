# Proofread: industrial-organization/logit-supply-side/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:20:00Z._

## Paper / Source Verification

### Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics* 25(2), 242-262.

- **Located:** https://ideas.repec.org/a/rje/randje/v25y1994isummerp242-262.html
- **Tutorial claims:** Berry inversion turns observed shares into a linear estimating equation; the paper is the source for the logit inversion and IV demand approach.
- **Source says:** Author is Steven T. Berry, year 1994, RAND Journal of Economics, volume 25, issue 2 (Summer), pages 242-262. The paper introduces methods for estimating discrete-choice demand models with endogenous prices.
- **Verdict:** OK
- **Note:** All bibliographic fields confirmed.

### Nevo, A. (2001). "Measuring Market Power in the Ready-to-Eat Cereal Industry." *Econometrica* 69(2), 307-342.

- **Located:** https://ideas.repec.org/a/ecm/emetrp/v69y2001i2p307-42.html
- **Tutorial claims:** The cereal market framing (Choco-Bombs, Fiber-Bran, etc. as fictional stand-ins) draws on Nevo's empirical cereal study as motivation.
- **Source says:** Author is Aviv Nevo, year 2001, Econometrica, volume 69, issue 2, pages 307-342. The paper estimates demand and measures market power in the ready-to-eat cereal industry using BLP-style random coefficients logit.
- **Verdict:** OK
- **Note:** All bibliographic fields confirmed.

### Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition, Ch. 3.

- **Located:** https://www.cambridge.org/core/books/discrete-choice-methods-with-simulation/49CABD00F3DDDA088A8FBFAAAD7E9546
- **Tutorial claims:** Referenced as background for discrete choice and logit methods; Ch. 3 covers logit.
- **Source says:** Author is Kenneth E. Train, publisher Cambridge University Press, 2nd edition published 2009. Chapter 3 is titled "Logit" (pages 34-75) and covers the logit model as a foundational discrete choice method.
- **Verdict:** OK
- **Note:** All bibliographic fields confirmed; Ch. 3 is indeed the logit chapter.

## Main Message Audit

> Markup recovery is only as credible as the estimated demand slope. Berry inversion and IV/2SLS estimate that slope from shares, prices, and instruments. The Bertrand-Nash FOC then converts demand derivatives and ownership into marginal costs. Simple logit makes the inversion clear but imposes rigid substitution.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Markup recovery is only as credible as the estimated demand slope | Results (OLS error in $\alpha$ carries into cost recovery; MAE 0.455 dollars) | OK |
| Berry inversion and IV/2SLS estimate that slope from shares, prices, and instruments | Equations (Berry inversion formula) and Solution Method (steps 1-3) | OK |
| The Bertrand-Nash FOC then converts demand derivatives and ownership into marginal costs | Equations ($\Omega_{jk}$ definition and $\Omega m = s$) and Solution Method (steps 4-6) | OK |
| Simple logit makes the inversion clear but imposes rigid substitution | Results (elasticity heatmap caption: "Off-diagonal columns are identical (IIA)"; cross-elasticities depend on rival shares, not product similarity) | OK |

Issues:
- None. All clauses in the takeaway are demonstrated by the README's equations, method, or results.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $t$ | Equations, sentence 1 | Yes - "Markets are indexed by $t$" | OK |
| $j$ | Equations, sentence 1 | Yes - "Products are indexed by $j$" | OK |
| $\delta_{jt}$ | Equations, equation 1 | Yes - "Mean utility collects characteristics, price, and unobserved quality" with defining equation | OK |
| $\beta_0$ | Equations, equation 1 | Partial - value and label appear in Model Setup table ("Base utility") | Within ~50 lines of first use; acceptable |
| $\beta_{\text{sugar}}$ | Equations, equation 1 | Partial - value and label in Model Setup table ("Sugar taste") | Within ~50 lines; acceptable |
| $\beta_{\text{fiber}}$ | Equations, equation 1 | Partial - value and label in Model Setup table ("Fiber taste") | Within ~50 lines; acceptable |
| $x^{\text{sugar}}_{jt}$ | Equations, equation 1 | Partial - no explicit prose gloss; identified as a product characteristic by context and Model Setup | Acceptable given subscript notation and Model Setup context |
| $x^{\text{fiber}}_{jt}$ | Equations, equation 1 | Partial - same as $x^{\text{sugar}}_{jt}$ | Acceptable |
| $\alpha$ | Equations, equation 1 | Partial - "Price sensitivity" label and value in Model Setup table | Within ~50 lines; acceptable |
| $p_{jt}$ | Equations, equation 1 | Implicit - "price" named in surrounding prose | Acceptable |
| $\xi_{jt}$ | Equations, equation 1 | Implicit - "unobserved quality" named in surrounding prose | Acceptable |
| $s_{jt}$ | Equations, logit share equation | Yes - "Simple logit shares satisfy" with defining equation | OK |
| $s_{0t}$ | Equations, logit share equation | Yes - defined by the share equation | OK |
| $k$ | Equations, logit share equation | Implicit - product summation index, same role as $j$ | Acceptable |
| $f$ / $f(j)$ | Equations, FOC sentence | Partial - "Firm $f$ chooses prices for its products"; $f(j)$ used as ownership function in FOC with no formal prose definition | Acceptable from context, but $f(j)$ as a function is never stated explicitly |
| $c_k$ | Equations, FOC equation | Undefined at first use - appears in FOC before any prose label; first labeled "cost" implicitly via $c=p-m$ two equations later | Flag: $c_k$ undefined at first use |
| $O_{jk}$ | Equations, supply section | Yes - "Let $O_{jk}=1$ when products $j$ and $k$ share an owner" | OK |
| $\Omega_{jk}$ | Equations, supply section | Yes - "Define the pricing matrix $\Omega_{jk}=-O_{jk}\frac{\partial s_k}{\partial p_j}$" | OK |
| $m$ | Equations, supply section | Yes - "The markup vector $m=p-c$ solves $\Omega m=s$" | OK |
| $p$ (vector) | Equations, FOC: $s_j(p)$ | Drift - $p_{jt}$ (scalar) in demand section; $p$ (vector) in supply section without transition note | Minor drift; economically the same object |
| $s$ (vector) | Equations, $\Omega m = s$ | Drift - $s_{jt}$ (scalar) in demand section; $s$ (vector) in supply section | Minor drift; same object |
| $c$ (vector) | Equations, $c=p-m$ | Partial definition - equation defines $c$ in terms of $p$ and $m$, but "marginal cost" label appears only in Overview prose and Model Setup without assigning $c$ | See flagged issues |

Flagged issues:
- $c_k$ appears in the Bertrand-Nash FOC equation without a prior prose definition. The Overview mentions "marginal cost vector" but does not assign $c$ to it. The equation $c = p - m$ two lines later provides a functional relationship but not an explicit label. A sentence such as "$c_k$ is the marginal cost of product $k$" is absent before the FOC.
- $p$ and $s$ drift from indexed scalars ($p_{jt}$, $s_{jt}$) in the demand equations to unindexed vectors ($p$, $s$) in the supply equations without a transition note. The drift is standard in the IO literature and causes no ambiguity here, but it is an unlabeled change of form.

## Summary

All three cited references are correctly attributed: Berry (1994) RAND 25(2) 242-262, Nevo (2001) Econometrica 69(2) 307-342, and Train (2009) Cambridge 2nd edition Ch. 3 are each verified against authoritative sources with no discrepancies. The main message is fully supported by the README's equations, method, and results, with no overreach or unsupported clauses. The one notation issue worth fixing is that $c_k$ (marginal cost) appears in the Bertrand-Nash FOC without a prior prose definition; adding a single sentence defining $c_k$ before the FOC equation would close this gap. Total issues: 0 MAJOR, 0 MINOR (references), 0 NOT FOUND, 0 OVERREACH, 1 notation flag ($c_k$ undefined at first use) and 2 minor notation drift observations ($p$ and $s$ shifting from scalar to vector form).
