# Proofread: numerical-methods/scalar-optimization/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T18:50:00Z._

## Paper / Source Verification

### Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.

- **Located:** NOT FOUND
- **Tutorial claims:** The tutorial cites these slides as a primary reference for the numerical optimization methods presented (golden section and Newton on FOC).
- **Source says:** Toshihiko Mukoyama is a confirmed faculty member at Georgetown University who teaches ECON 606; his public website lists macroeconomics lecture materials, but no document titled "Basic Numerical Methods" for ECON 606 is publicly indexed. Course slides of this type are typically distributed only through course management systems.
- **Verdict:** NOT FOUND
- **Note:** The instructor and course are real; the inaccessibility is expected for private course slides, not a citation error.

---

### Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 10.

- **Located:** https://www.cambridge.org/numericalrecipes (official Cambridge University Press page); table of contents confirmed via https://assets.cambridge.org/97805218/80688/frontmatter/9780521880688_frontmatter.pdf
- **Tutorial claims:** Cited as a reference for golden section search and Newton-on-FOC optimization methods.
- **Source says:** Third edition (2007) confirmed, ISBN 978-0-521-88068-8, Cambridge University Press. Chapter 10 is titled "Minimization or Maximization of Functions" (pp. 487–562) and includes §10.2 "Golden Section Search in One Dimension" and §10.4 "One-Dimensional Search with First Derivatives" — directly covering both methods the tutorial demonstrates.
- **Verdict:** OK
- **Note:** All bibliographic fields (authors, year, publisher, edition, chapter) are correct.

---

### Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 4.

- **Located:** https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/
- **Tutorial claims:** Cited as a reference for numerical optimization methods in an economics context.
- **Source says:** Published by MIT Press, 1998, ISBN 0-262-10071-1. Chapter 4 is titled "Optimization" and covers unconstrained optimization algorithms including Newton's method, search methods, and derivative-free optimization with economic applications.
- **Verdict:** OK
- **Note:** All bibliographic fields (author, year, publisher, chapter) are correct.

---

## Main Message Audit

> Golden section is the safe default for a one-state Bellman inner step because it only needs unimodality and a bracket: it contracts at a fixed factor regardless of where the optimum sits. Newton on the FOC is much faster when $g'$ and $g''$ are available and $x_0$ is inside the basin of attraction, but a far-off start makes the parabolic extrapolation overshoot outside the feasible interval. Cake-eating and consumption-savings VFI later in the catalog use the golden-section flavour for this reason; smooth optimum problems with cheap derivatives can graduate to Newton.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Golden section needs only unimodality and a bracket | Overview ("The first needs only unimodality and is globally safe"), Equations (no derivative required in golden section update) | OK |
| It contracts at a fixed factor regardless of where the optimum sits | Results ("golden-section iteration counts are flat across starting points (the bracket is the same)") | OK |
| Newton on the FOC is much faster when $g'$ and $g''$ are available and $x_0$ is inside the basin of attraction | Results (48 golden section iterations vs. 6 Newton iterations) | OK |
| A far-off start makes the parabolic extrapolation overshoot outside the feasible interval | Results ("3 of 9 starts overshoot outside $(0, W)$ and diverge") | OK |
| Cake-eating and consumption-savings VFI later in the catalog use the golden-section flavour | Not shown in this README's Equations, Method, or Results | OVERREACH |
| Smooth optimum problems with cheap derivatives can graduate to Newton | Equations (Newton uses $g'$ and $g''$), Results (Newton converges in 6 iterations) | OK |

Issues:
- The claim that other catalog tutorials (cake-eating, consumption-savings VFI) use golden section is a forward reference to separate tutorials that this README does not demonstrate; it is plausible as an authorial statement about the catalog but is unsupported within this document.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $c$ | Overview (control variable in $\max_{c \in [0,W]}$) | Yes, implicitly as control | Overloaded: reused as name stem $c_n$ for golden section left probe point in Equations |
| $W$ | Overview | Yes, Model Setup table ("Wealth at the inner state") | |
| $u(c)$ | Overview | Yes, Equations ($u(c) = \log c$) | |
| $\beta$ | Overview | Yes, Model Setup table ("Discount factor") | |
| $V$ | Overview | Yes, Equations (closed-form formula given) | |
| $c^{\ast}$ | Overview | Yes, Overview ($c^{\ast} = (1-\beta)W$) | |
| $g(c)$ | Equations | Yes, defined in Equations | |
| $g'(c)$ | Equations | Yes, formula given in Equations | |
| $g''(c)$ | Equations | Yes, formula given in Equations | |
| $\phi$ | Equations | Yes, $(\sqrt{5}-1)/2 \approx 0.618$ | |
| $[a, b]$ | Overview ("contracts a bracket $[a,b]$") | Yes, used as bracket in Equations; initial values in Model Setup | |
| $c_n$ | Equations | Yes, $c_n = b_n - \phi(b_n - a_n)$ | Overloads bare $c$ (control variable); subscript-n distinguishes |
| $d_n$ | Equations | Yes, $d_n = a_n + \phi(b_n - a_n)$ | |
| $a_n, b_n$ | Equations (in definition of $c_n$) | Implicitly subscript-n versions of $a, b$ | Not formally re-introduced as iteration-indexed; natural from context |
| $x_n$ | Equations (Newton update $x_{n+1} = x_n - \ldots$) | Introduced at point of use as current iterate | No prior declaration; standard iterate notation; unambiguous from context |
| $q_n(c)$ | Equations | Yes, formula given immediately | |
| $a_0, b_0$ | Model Setup table | Yes, initial bracket values | |
| $x_0$ | Model Setup table | Yes, "Starting iterate for Newton-on-FOC" | |
| $\varepsilon$ | Model Setup table | Yes, "Stopping rule on bracket width and on $g'(x_n)$" | |

Flagged issues:
- **$c$ / $c_n$ overloading:** $c$ is introduced as the general control variable throughout the problem, and $c_n$ is then defined in the Equations section as the golden section left probe point. The subscript-$n$ provides syntactic distinction, but the two objects are semantically unrelated. The issue is minor because no ambiguity arises in practice.

---

## Summary

The tutorial is mathematically sound and internally consistent: all derivative formulas, the closed-form policy $c^{\ast} = (1-\beta)W$, and the golden section update rule are verified correct against the code in `run.py`. Two of the three references are fully verifiable (Press et al. and Judd both confirm correct chapter assignments and bibliographic details); the Mukoyama lecture slides cannot be located because they are private course material, which is expected. The single OVERREACH is the Takeaway's claim that other catalog tutorials use golden section — true as a catalog-level statement but not demonstrated within this README. The one notation flag is a mild overloading of $c$ (control variable) with $c_n$ (golden section probe point), distinguishable by subscript. Overall verdict: 0 MAJOR, 0 MINOR bibliographic issues, 1 NOT FOUND (private course slides), 1 OVERREACH, 1 notation flag. The most important fix is either removing the forward reference about other tutorials or grounding it with a link, since it is the only claim this README cannot support on its own.
