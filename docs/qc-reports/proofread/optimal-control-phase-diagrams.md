# Proofread: optimal-control/phase-diagrams/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:45:00Z._

## Paper / Source Verification

### Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).

- **Located:** https://academic.oup.com/ej/article-abstract/38/152/543/5282967 (also at JSTOR: https://www.jstor.org/stable/2224098)
- **Tutorial claims:** Cites Ramsey (1928) as the foundational reference for the Ramsey planner's problem solved in the tutorial.
- **Source says:** F. P. Ramsey, "A Mathematical Theory of Saving," *The Economic Journal*, vol. 38, no. 152, pp. 543-559, December 1928.
- **Verdict:** MINOR
- **Note:** Year, journal, volume, and issue are correct; page range 543-559 is absent from the citation.

### Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.

- **Located:** MIT Press catalogue entry; ISBN 9780262025539.
- **Tutorial claims:** Cites Barro and Sala-i-Martin (2004) Ch. 2 as the textbook treatment of the Ramsey-Cass-Koopmans model.
- **Source says:** Chapter 2 is titled "Growth Models with Consumer Optimization (the Ramsey Model)" and covers the continuous-time Ramsey model with endogenous household saving, which is exactly the content of this tutorial. Year 2004, MIT Press, second edition are all correct.
- **Verdict:** OK
- **Note:** Citation is accurate.

## Main Message Audit

> "The phase diagram shows direction. It does not choose initial consumption. The transversality condition selects the stable arm. Local linearization gives the slope near the steady state. Backward integration draws the nonlinear path away from it."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| The phase diagram shows direction | Results - quiver/arrow field in phase-diagram figure | OK |
| It does not choose initial consumption | Results - path-selection figure and surrounding prose | OK |
| The transversality condition selects the stable arm | Equations - transversality stated; Solution Method - "Points above or below it miss transversality" | OK |
| Local linearization gives the slope near the steady state | Solution Method - Jacobian, eigenvalues, slope $dc/dk = 0.1110$ | OK |
| Backward integration draws the nonlinear path away from it | Solution Method - algorithm steps 4-6 and pseudocode | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $c(t)$ | Objective integral (Equations) | Yes - "consumption" named in Overview | OK |
| $k(t)$ | Capital accumulation constraint (Equations) | Yes - "capital" named in Overview | OK |
| $\rho$ | Objective integral (Equations) | Yes - Model Setup table, "Continuous-time discount rate" | OK |
| $\sigma$ | Objective integral (Equations) | Yes - Model Setup table, "CRRA coefficient and inverse EIS" | OK |
| $A$ | Capital accumulation constraint (Equations) | Yes - Model Setup table, "Total factor productivity" | OK |
| $\alpha$ | Capital accumulation constraint (Equations) | Yes - Model Setup table, "Capital share" | OK |
| $\delta$ | Capital accumulation constraint (Equations) | Yes - Model Setup table, "Depreciation rate" | OK |
| $f(k)$ | Two-dimensional system (Equations) | Yes - defined inline: $f(k) = Ak^{\alpha}$ | OK |
| $f'(k)$ | Two-dimensional system (Equations) | Implicit - derivative of $f$, standard notation | OK |
| $k^{\ast}$ | Consumption nullcline (Equations) | Yes - defined as $\left(\frac{\alpha A}{\rho+\delta}\right)^{1/(1-\alpha)}$ | OK |
| $c^{\ast}$ | After consumption nullcline (Equations) | Yes - defined as $f(k^{\ast}) - \delta k^{\ast}$ | OK |
| $u'(c(t))$ | Transversality condition (Equations) | No - $u$ is never named; utility appears as $\frac{c^{1-\sigma}}{1-\sigma}$ inline in the objective | Flagged |
| $J$ | Solution Method | Yes - "The Jacobian of $(\dot{k},\dot{c})$ at $(k^{\ast},c^{\ast})$" | OK |
| $f''(k^{\ast})$ | Jacobian (Solution Method) | Implicit - second derivative of $f$, standard notation | OK |
| $\lambda_s$ | Solution Method prose | Yes - stated as $-0.0710$ with label "stable eigenvalue" | OK |
| $\lambda_u$ | Solution Method prose | Yes - stated as $0.1110$ with label "unstable eigenvalue" | OK |
| $v_s$, $m_s$ | Pseudocode step 3 (Solution Method) | Yes - defined within the pseudocode block | OK |

Flagged issues:
- **$u'(c(t))$ in the transversality condition**: the letter $u$ is introduced here without being named. The utility functional is written as $\frac{c(t)^{1-\sigma}}{1-\sigma}$ inside the objective integral but is never labeled $u$. A reader must infer the identity. Replacing $u'(c(t))$ with the explicit expression $c(t)^{-\sigma}$ (which follows from differentiating the CRRA form) would make the transversality condition self-contained.

## Summary

The tutorial is clean and internally consistent. Numerical values for $k^{\ast}$, $c^{\ast}$, $\lambda_s$, $\lambda_u$, and the stable-arm slope all match the code. Both citations locate correctly; the single issue is that the Ramsey (1928) page range (543-559) is omitted from the reference list (1 MINOR). The main message is fully supported by the Equations, Solution Method, and Results sections (0 OVERREACH, 0 UNSUPPORTED). The one notation flag is that $u'(c(t))$ appears in the transversality condition but the function $u$ is never defined by name - the CRRA utility is written inline in the objective and not labeled $u$ anywhere prior to that equation. The most important fix is to define $u$ explicitly or replace $u'(c(t))$ with its CRRA form $c(t)^{-\sigma}$ in the transversality display.
