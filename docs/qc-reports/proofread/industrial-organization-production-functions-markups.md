# Proofread: industrial-organization/production-functions-markups/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:30:00Z._

## Paper / Source Verification

### Olley, S., and Pakes, A. (1996). The Dynamics of Productivity in the Telecommunications Equipment Industry. *Econometrica*, 64(6), 1263-1297.

- **Located:** https://www.econometricsociety.org/publications/econometrica/1996/11/01/dynamics-productivity-telecommunications-equipment-industry
- **Tutorial claims:** Cited as background for the investment proxy approach to correcting production-function estimation for simultaneity bias.
- **Source says:** G. Steven Olley and Ariel Pakes, Econometrica, Vol. 64, No. 6, pp. 1263-1297, November 1996. The paper introduces an estimator that uses investment as a proxy for unobserved productivity to address simultaneity and selection bias.
- **Verdict:** OK
- **Note:** Bibliographic details match exactly; the tutorial's investment-proxy approach directly follows this paper's methodology.

### Levinsohn, J., and Petrin, A. (2003). Estimating Production Functions Using Inputs to Control for Unobservables. *Review of Economic Studies*, 70(2), 317-341.

- **Located:** https://academic.oup.com/restud/article-abstract/70/2/317/1586773
- **Tutorial claims:** Cited as part of the proxy-variable literature for production-function estimation.
- **Source says:** James Levinsohn and Amil Petrin, Review of Economic Studies, Vol. 70, No. 2, pp. 317-341, April 2003. The paper's core innovation is using intermediate inputs (not investment) as the proxy variable to control for productivity.
- **Verdict:** OK
- **Note:** Bibliographic details match exactly; the tutorial uses investment as the proxy (Olley-Pakes style), but the Levinsohn-Petrin citation as background for the proxy-variable literature is not a false claim.

### De Loecker, J., and Warzynski, F. (2012). Markups and Firm-Level Export Status. *American Economic Review*, 102(6), 2437-2471.

- **Located:** https://www.aeaweb.org/articles?id=10.1257%2Faer.102.6.2437
- **Tutorial claims:** Implied as the source for the markup formula $\mu_{it} = \theta^m / \alpha^m_{it}$ derived from cost minimization with a variable input.
- **Source says:** Jan De Loecker and Frederic Warzynski, American Economic Review, Vol. 102, No. 6, pp. 2437-2471, October 2012. The paper derives firm-level markups as the ratio of the output elasticity of a variable input to that input's expenditure share, using cost-minimizing behavior. The exact formula matches the tutorial's markup equation.
- **Verdict:** OK
- **Note:** Bibliographic details and the attributed markup formula both match the source.

### Lectures 10-12 Slides 2023: Production functions, proxy methods, and markups.

- **Located:** NOT FOUND
- **Tutorial claims:** Cited as supplementary lecture material on production functions, proxy methods, and markups.
- **Source says:** N/A - no author, institution, or URL is provided; the reference cannot be located or verified.
- **Verdict:** NOT FOUND
- **Note:** This is an informal internal reference with no author or institution; it cannot be verified from the information given.

## Main Message Audit

> "The object is the firm-year markup. This tutorial recovers it from a materials output elasticity and a materials revenue share. Inputs respond to productivity before output is observed. The computation uses an investment proxy to correct the materials elasticity before forming markups. The markup estimate is only as credible as the production elasticity and the materials share behind it. In this controlled panel, correcting for productivity greatly reduces markup error."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Markup is recovered from materials elasticity and revenue share | Equations: $\mu_{it} = \theta^m / \alpha^m_{it}$ | OK |
| Inputs respond to productivity before output is observed | Equations: simultaneity discussion following the Cobb-Douglas equation | OK |
| Investment proxy corrects the materials elasticity | Equations and Solution Method: productivity control $\tilde\omega_{it}$ formed from investment | OK |
| Correcting for productivity greatly reduces markup error | Results table: materials OLS bias 0.331 vs. proxy bias 0.020; quintile table shows OLS markups too high in all cells | OK |
| Markup credibility depends on both the elasticity and the revenue share | Equations: markup formula involves both $\theta^m$ and $\alpha^m_{it}$; Results discuss both sources of error | OK |

Issues:
- None. All clauses are supported by the README's equations, method description, and results.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $i$ | Equations, first sentence | Yes | "Let $i$ index firms" |
| $t$ | Equations, first sentence | Yes | "and $t$ index years" |
| $y_{it}$ | Equations, second sentence | Yes | "denoted by $y_{it}$, $l_{it}$, $k_{it}$, and $m_{it}$" |
| $l_{it}$ | Equations, second sentence | Yes | defined alongside $y_{it}$ |
| $k_{it}$ | Equations, second sentence | Yes | defined alongside $y_{it}$ |
| $m_{it}$ | Equations, second sentence | Yes | defined alongside $y_{it}$ |
| $\beta_l$, $\beta_k$, $\beta_m$ | Equations, Cobb-Douglas equation | Partial | No inline word definition; values appear in Model Setup table within 30 lines |
| $\omega_{it}$ | Equations, Cobb-Douglas equation | Yes | "productivity $\omega_{it}$" defined in the following sentence |
| $\varepsilon_{it}$ | Equations, Cobb-Douglas equation | Implicit | Standard i.i.d. error term; not named in prose but recognizable to the target audience |
| $I_{it}$ | Equations, investment paragraph | Yes | "The proxy variable is investment $I_{it}$" |
| $h(k_{it}, \omega_{it})$ | Equations, investment equation | Yes | Introduced as an investment policy with explicit monotonicity condition |
| $\nu_{it}$ | Equations, investment equation | Implicit | Standard noise term; not named in prose but recognizable to the target audience |
| $\tilde\omega_{it}$ | Equations, control-function paragraph | Yes | "productivity control $\tilde\omega_{it} = h^{-1}(k_{it}, I_{it})$" |
| $\rho$ | Equations, controlled regression | No | Coefficient on the productivity control; introduced in the regression equation without any accompanying description |
| $u_{it}$ | Equations, controlled regression | Implicit | Standard regression residual; not named in prose but recognizable to the target audience |
| $\theta^m$ | Equations, markup paragraph | Yes | "the materials elasticity is $\theta^m = \beta_m$" |
| $\alpha^m_{it}$ | Equations, markup paragraph | Yes | Defined with its own display equation and the phrase "be the materials revenue share" |
| $\mu_{it}$ | Equations, markup formula | Yes | "the gross markup $\mu_{it} = \theta^m / \alpha^m_{it}$" |

Flagged issues:
- $\rho$: appears in the controlled regression equation "$y_{it} = \beta_l l_{it} + \beta_k k_{it} + \beta_m m_{it} + \rho \tilde\omega_{it} + u_{it}$" without any definition or description. Its role as the coefficient on the productivity control is not stated in prose.

## Summary

The tutorial's three formal academic references are verified and correct: Olley-Pakes (1996), Levinsohn-Petrin (2003), and De Loecker-Warzynski (2012) all match their published bibliographic details and the attributed methods. The informal "Lectures 10-12 Slides 2023" entry cannot be located (NOT FOUND) due to missing author and institution information. All main-message claims are supported by the README's equations, solution method, and results tables - 0 OVERREACH and 0 UNSUPPORTED clauses. The single notation flag is $\rho$, which appears in the controlled regression equation without any accompanying word definition. The most important fix is adding a one-phrase description of $\rho$ alongside the controlled regression equation.
