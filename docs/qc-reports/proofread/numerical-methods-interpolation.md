# Proofread: numerical-methods/interpolation/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T19:00:00Z._

## Paper / Source Verification

### Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.

- **Located:** NOT FOUND
- **Tutorial claims:** Internal lecture notes on numerical methods from Georgetown ECON 606.
- **Source says:** Toshihiko Mukoyama is a confirmed Georgetown faculty member. Public records identify ECON 606 as "Macroeconomics II"; the slides are not publicly accessible. "Basic Numerical Methods" is consistent with a topic module within that course.
- **Verdict:** NOT FOUND
- **Note:** Internal course material; non-public. No verifiable error in the citation as stated.

---

### Fritsch, F. N. and Carlson, R. E. (1980). *Monotone Piecewise Cubic Interpolation*. SIAM Journal on Numerical Analysis 17(2), 238-246.

- **Located:** https://epubs.siam.org/doi/10.1137/0717021
- **Tutorial claims:** PCHIP endpoint slopes are chosen by a monotonicity-preserving rule (Fritsch-Carlson 1980); the method is $C^1$ and never overshoots a monotone target.
- **Source says:** The paper derives necessary and sufficient conditions for piecewise cubics to be monotone and provides an algorithm constructing a visually pleasing monotone interpolant — the foundational PCHIP algorithm. Vol. 17, issue 2, pp. 238–246, April 1980.
- **Verdict:** OK
- **Note:** All bibliographic details (authors, year, journal, volume, issue, pages) verified correct.

---

### Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 3.

- **Located:** https://www.cambridge.org/numericalrecipes
- **Tutorial claims:** Cited as a general reference for interpolation methods.
- **Source says:** Chapter 3 is titled "Interpolation and Extrapolation" and covers cubic spline interpolation among other methods. 3rd edition published 2007 by Cambridge University Press.
- **Verdict:** OK
- **Note:** Authors, year, publisher, edition, and chapter subject all verified correct.

---

### Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 6.

- **Located:** https://mitpress.mit.edu/9780262547741/numerical-methods-in-economics/
- **Tutorial claims:** Cited as a general reference for interpolation methods in economics.
- **Source says:** Chapter 6 is titled "Approximation Methods" and covers piecewise polynomial interpolation, splines (including shape-preserving splines), and orthogonal polynomial approximation.
- **Verdict:** OK
- **Note:** Author, year, publisher, and chapter subject all verified correct.

---

## Main Message Audit

> "Piecewise linear is the safe default for value functions with borrowing constraints: it preserves shape, never overshoots, and requires no setup. Natural cubic spline gives the best convergence on smooth functions but rings near kinks and can violate monotonicity. PCHIP is the right middle ground for monotone-but-non-smooth policies — the case EGP and consumption-savings VFI face every period — beating both linear (more accurate) and cubic (no ringing) at the same node count."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Piecewise linear preserves shape and never overshoots | Results (figures and error table show linear tracking kinked target monotonically with no overshoot) | OK |
| Piecewise linear requires no setup | Method (convex combination, no solve step) | OK |
| Natural cubic spline gives the best convergence on smooth functions | Results: convergence figure text states cubic and PCHIP both converge at slope −4; smooth-target error table shows PCHIP sup-error 0.781 vs cubic 1.09 at N=10 | OVERREACH |
| Cubic rings near kinks | Results (error-curves figure, kinked sup-error 4.57e-02) | OK |
| Cubic can violate monotonicity | Implied by overshoot discussion in Results; no explicit monotonicity-violation demonstration | OK |
| PCHIP beats linear in accuracy on kinked target | Results table: PCHIP kinked sup 0.029 < linear 0.076 | OK |
| PCHIP beats cubic by avoiding ringing | Results table and error-curves figure | OK |

Issues:
- **OVERREACH — "cubic gives best convergence on smooth functions":** The convergence section states "cubic and PCHIP drop at roughly slope −4" (identical rate). The smooth-target error table at N=10 shows PCHIP has lower sup-error (0.781) and lower L2 error (0.171) than cubic (sup 1.09, L2 0.258). The tutorial does not demonstrate cubic being superior to PCHIP on smooth functions within the node range shown.
- **Internal inconsistency in Results text:** The second results paragraph reads "with cubic uniformly smallest" (for smooth-target pointwise errors). The error table in the same Results section shows PCHIP has strictly lower smooth sup-error and smooth L2 error than cubic at N=10. These two statements within the README contradict each other.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $V$ | Overview ("stores $V$ at a finite set of grid points") | Yes — Equations defines $V(W)$ | OK |
| $W$ | Equations ($V(W)$ formula) | Yes — Equations; Model Setup gives domain | OK |
| $\beta$ | Equations ($V(W)$ formula) | Partial — Model Setup table (next section, within 50 lines) | Acceptable |
| $a$ | Equations ($c(a)$ formula) | Yes — Equations | OK |
| $a_{\text{kink}}$ | Equations ("borrowing constraint at $a_{\text{kink}}$") | Yes — Equations + Model Setup | OK |
| $r$ | Equations ($c(a) = (1+r)\,a + y$) | No — never listed in Model Setup or prose | Flagged |
| $y$ | Equations ($c(a) = (1+r)\,a + y$) | No — never listed in Model Setup or prose | Flagged |
| $\mathrm{MPC}$ | Equations ($c(a)$ piecewise formula) | Yes — prose immediately after equation: "marginal propensity to consume" | OK |
| $\hat{f}$ | Equations (piecewise linear formula header) | Yes — context of interpolant | OK |
| $x_i,\,x_{i+1}$ | Equations ("adjacent nodes $x_i \le x \le x_{i+1}$") | Yes — defined by context | OK |
| $f(x_i),\,f(x_{i+1})$ | Equations (piecewise linear formula) | Yes — function values at nodes | OK |
| $x_0,\,x_N$ | Equations (natural spline BCs: $\hat{f}''(x_0)=\hat{f}''(x_N)=0$) | Yes — boundary-node notation consistent throughout | OK |
| $N$ | Equations ($x_N$) | Yes — Model Setup: "Display node count $N$ = 10" | OK |
| $\hat{f}',\,\hat{f}''$ | Equations (cubic spline continuity conditions) | Yes — derivative notation, standard | OK |
| $C^1,\,C^2$ | Equations (PCHIP / spline smoothness comparison) | Implicit — standard smoothness class notation | OK |
| $m_i$ | Pseudocode ("compute secant slopes $m_i$ between adjacent nodes") | Partial — defined by descriptive label in pseudocode | Acceptable |
| $h_i$ | Pseudocode ("w ← (x − x_i) / h_i") | No — never defined anywhere in README | Flagged |

Flagged issues:
- **$r$ (interest rate):** first used in $c(a) = (1+r)\,a + y$ in the Equations section; no value or definition appears in the Model Setup table or surrounding prose. The code sets `r=0.04`.
- **$y$ (income/endowment):** first used alongside $r$ in the same formula; similarly absent from Model Setup. The code sets `y=0.5`.
- **$h_i$ (node spacing):** used in the piecewise-linear pseudocode as `w ← (x − x_i) / h_i` but never defined. It is implicitly $h_i = x_{i+1} - x_i$ from the Equations formula, but this equivalence is not stated.

## Summary

Three of the four references verified correctly (Fritsch & Carlson 1980, Press et al. 2007, Judd 1998); the Mukoyama (2021) slides are internal course material and cannot be independently verified. The main substantive issues are: (1) the Takeaway claims "Natural cubic spline gives the best convergence on smooth functions," which is OVERREACH — the tutorial's own convergence figure and table show PCHIP outperforming cubic at every node count in the tested range and matching its asymptotic slope; (2) the Results prose states "with cubic uniformly smallest" for smooth-target errors, directly contradicted by the smooth-target error table in the same section (PCHIP sup-error 0.781 < cubic 1.09). Three notation symbols lack definitions in the README: $r$, $y$, and $h_i$. The single most important fix is correcting the "cubic uniformly smallest" claim in Results (and the corresponding Takeaway sentence), since it is an internal factual contradiction between the results text and the results table generated by the same code.
