# Proofread: numerical-methods/root-finding/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T17:35:00Z._

## Paper / Source Verification

### Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.

- **Located:** NOT FOUND
- **Tutorial claims:** Course lecture slides covering basic numerical methods (bisection, Newton-Raphson) from Georgetown ECON 606.
- **Source says:** Toshihiko Mukoyama is confirmed faculty at Georgetown Economics with a notes/materials page. No public index of ECON 606 numerical methods slides was found; the course is not publicly accessible. ECON 606 at Georgetown is currently titled "Macroeconomics II," not a dedicated numerical methods course.
- **Verdict:** NOT FOUND
- **Note:** Unpublished course material - correct author and institution, but chapter-level content and year cannot be independently verified.

---

### Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 9.

- **Located:** https://www.cambridge.org/numericalrecipes (publisher page); Wikipedia: https://en.wikipedia.org/wiki/Numerical_Recipes
- **Tutorial claims:** Cites this as a reference for bisection and Newton-Raphson (Ch. 9), 3rd edition, 2007, Cambridge University Press.
- **Source says:** Full title is *Numerical Recipes: The Art of Scientific Computing*, 3rd ed., Cambridge University Press, 2007. Chapter 9 is titled "Root Finding and Nonlinear Sets of Equations" and explicitly covers both bisection and Newton-Raphson.
- **Verdict:** MINOR
- **Note:** Tutorial omits the subtitle "The Art of Scientific Computing"; year, edition, publisher, and chapter content are all correct.

---

### Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 5.

- **Located:** https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/
- **Tutorial claims:** Cites this as a reference for root-finding methods (Ch. 5), MIT Press, 1998.
- **Source says:** *Numerical Methods in Economics*, MIT Press, 1998. Part II of the book covers numerical analysis basics on ℝⁿ, including a chapter explicitly on nonlinear equations; the exact chapter-5 title is not exposed in public excerpts but is consistent with root-finding content.
- **Verdict:** OK
- **Note:** Year and publisher are confirmed correct; Chapter 5 as root-finding is consistent with the book's Part II structure but the exact chapter number could not be verified from publicly available excerpts.

---

## Main Message Audit

> "Bisection is the safe default when only a sign-change bracket is available: it halves the error every step and never leaves the bracket. Newton-Raphson is much faster once the iterate is near a simple root, because the tangent extrapolation squares the residual each step. The trade-off shows up at large $x_0$, where the Newton step here overshoots into $k < 0$ and the Cobb-Douglas residual becomes undefined. Aiyagari- and Huggett-style interest-rate solves later in the catalog use bisection on $r$ for exactly this reason."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Bisection halves the error every step | Equations (bisection bracket halves each iteration) + Results (31 iterations, linear 1/2 rate) | OK |
| Bisection never leaves the bracket | Equations (sign-change invariant is maintained at each step) | OK |
| Newton is much faster near a simple root | Results table (7 vs 31 iterations) + convergence figure | OK |
| Tangent extrapolation squares the residual each step | Equations section explicitly states quadratic convergence near a simple root | OK |
| Trade-off: Newton overshoots into k < 0 at large x_0 | Results (sensitivity figure, 2 DNC starting points above k*) | OK |
| Aiyagari- and Huggett-style solves later use bisection on r for exactly this reason | Not demonstrated anywhere in this tutorial | OVERREACH |

Issues:
- **OVERREACH**: The final sentence ("Aiyagari- and Huggett-style interest-rate solves later in the catalog use bisection on $r$ for exactly this reason") is a forward reference to tutorials not contained in this README. The tutorial cannot support or demonstrate this claim from its own equations or results.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k^{\ast}$ | Overview | Yes - "the steady-state capital stock $k^{\ast}$" | |
| $F'(k^{\ast})$ | Equations, first equation | Ambiguous - see flagged issue below | Used before $F$ is defined; intended as marginal product $f'(k)$, not derivative of the residual |
| $\beta$ | Equations, first equation | Not in Equations; defined in Model Setup table | Defined after first use; acceptable given standard parameter convention |
| $\delta$ | Equations, first equation | Not in Equations; defined in Model Setup table | Defined after first use; acceptable |
| $f(k)$ | Equations, second paragraph | Yes - "$f(k) = k^{\alpha}$" in the same sentence | |
| $\alpha$ | Equations, second equation | Not in Equations; defined in Model Setup table | Defined after first use; acceptable |
| $F(k)$ | Equations, second equation | Yes - "$F(k) = \alpha k^{\alpha-1} - \bar{r}$" | Defined as the residual function; conflicts with $F'$ in the preceding equation |
| $\bar{r}$ | Equations, second equation | Yes - "$\bar{r} \equiv \tfrac{1}{\beta} - 1 + \delta$" inline | Defined at first use |
| $m_n$ | Equations, bisection display | Yes - "$m_n = \tfrac{a_n + b_n}{2}$" inline | |
| $a_n, b_n$ | Equations, bisection display | Partially - bracket $[a, b]$ introduced in the sentence above, subscripted form introduced with the equation | Acceptable |
| $x_n$ | Equations, Newton display | Yes - defined in Newton equation | |
| $\varepsilon$ | Model Setup table | Yes - "Tolerance $\varepsilon$" | Used as `eps` in pseudocode; informal but unambiguous |

Flagged issues:
- **Overloaded $F$ / $F'$**: The first equation in the Equations section writes $F'(k^{\ast}) = \tfrac{1}{\beta} - 1 + \delta$ to express the market-clearing condition (marginal product equals user cost). Here $F'$ is intended to mean $f'(k)$ - the derivative of the Cobb-Douglas production function $f(k) = k^{\alpha}$. However, in the very next equation $F(k)$ is introduced and defined as the **residual** $\alpha k^{\alpha-1} - \bar{r}$. This makes $F$ refer to two different objects within a few lines: (1) the production function (whose derivative $F'$ appears in the first equation) and (2) the residual function defined in the second equation. The derivative of the residual $F'(k) = \alpha(\alpha-1)k^{\alpha-2}$ (as computed in `run.py` as `Fprime`) is NOT equal to $\bar{r}$, so the first equation is incorrect if $F$ is interpreted as the residual. The fix is to use lowercase $f'(k^{\ast})$ in the first equation, consistent with the production function $f(k) = k^{\alpha}$ already introduced in the following sentence.
- **$\alpha, \beta, \delta$ defined after first use**: These three parameters first appear in the Equations section but are only given values in the Model Setup table (which follows). This is a standard economics convention and is unlikely to confuse readers, but strictly speaking the symbols are undefined on first appearance.

---

## Summary

The tutorial is clean and internally consistent with one notable exception. The most important fix is the notation overload in the Equations section: the opening equation uses $F'(k^{\ast})$ to express the marginal product of capital ($f'(k^{\ast})$), but $F$ is immediately thereafter defined as the residual function - making $F'$ mean two distinct objects within four lines of the same section. This should be corrected to $f'(k^{\ast}) = \bar{r}$, consistent with the production function $f(k) = k^{\alpha}$ introduced in the subsequent sentence and with the code's own `Fprime` function. Beyond that: one citation is NOT FOUND (Mukoyama 2021 lecture slides - unpublished, unverifiable), one is MINOR (Numerical Recipes subtitle truncated), and the Takeaway contains one OVERREACH (forward reference to Aiyagari/Huggett tutorials that this README cannot support). All numerical values in the Model Setup table are consistent with the code. Issue count: 1 notation overload (MAJOR concern), 1 MINOR citation, 1 NOT FOUND, 1 OVERREACH.
