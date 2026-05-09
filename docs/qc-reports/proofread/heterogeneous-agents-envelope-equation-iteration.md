# Proofread: heterogeneous-agents/envelope-equation-iteration/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T04:30:00Z._

## Paper / Source Verification

### Maliar, L. and Maliar, S. (2013). Envelope Condition Method with an Application to Default Risk Models. *Journal of Economic Dynamics and Control*, 37(7), 1439-1459.

- **Located:** https://ideas.repec.org/a/eee/ecolet/v120y2013i2p262-266.html (Maliar & Maliar 2013) and https://ideas.repec.org/a/eee/dyncon/v69y2016icp436-459.html (Arellano, Maliar, Maliar & Tsyrennikov 2016)
- **Tutorial claims:** Maliar and Maliar (2013) introduced the Envelope Condition Method with an application to default risk models, published in JEDC 37(7), pp. 1439-1459.
- **Source says:** Two distinct real papers exist. (1) Maliar, L. and Maliar, S. (2013). "Envelope condition method versus endogenous grid method for solving dynamic programming problems." *Economics Letters*, 120(2), 262-266. (2) Arellano, C., Maliar, L., Maliar, S. and Tsyrennikov, V. (2016). "Envelope condition method with an application to default risk models." *Journal of Economic Dynamics and Control*, 69, 436-459.
- **Verdict:** MAJOR
- **Note:** The citation conflates two papers - the title belongs to the 2016 Arellano, Maliar, Maliar & Tsyrennikov paper (JEDC 69, 436-459), while the listed authors (Maliar & Maliar only) and year (2013) belong to the Economics Letters paper with a different title. The correct citation for the paper the title describes is: Arellano, C., Maliar, L., Maliar, S. and Tsyrennikov, V. (2016), JEDC 69, 436-459.

### Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.

- **Located:** https://ideas.repec.org/a/eee/ecolet/v91y2006i3p312-320.html
- **Tutorial claims:** Carroll (2006) introduced the endogenous gridpoints method, published in Economics Letters 91(3), pp. 312-320.
- **Source says:** Carroll, C. D. (2006). "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." *Economics Letters*, 91(3), 312-320.
- **Verdict:** OK
- **Note:** Title, author, year, journal, volume, issue, and pages all match.

### Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.

- **Located:** https://www.jstor.org/stable/2938366
- **Tutorial claims:** Deaton (1991) published a study on saving and liquidity constraints in Econometrica 59(5), pp. 1221-1248.
- **Source says:** Deaton, A. (1991). "Saving and Liquidity Constraints." *Econometrica*, 59(5), 1221-1248.
- **Verdict:** OK
- **Note:** All bibliographic fields match exactly.

### Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.

- **Located:** https://academic.oup.com/qje/article-abstract/112/1/1/1870884
- **Tutorial claims:** Carroll (1997) published the buffer-stock saving theory in QJE 112(1), pp. 1-55.
- **Source says:** Carroll, C. D. (1997). "Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis." *Quarterly Journal of Economics*, 112(1), 1-55.
- **Verdict:** OK
- **Note:** All bibliographic fields match exactly.

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** Ljungqvist and Sargent (2018) cover buffer-stock/incomplete-markets models in Chapter 18 of the 4th edition of Recursive Macroeconomic Theory.
- **Source says:** The 4th edition (2018, MIT Press, ISBN 9780262038669) is confirmed. Chapter 18 content could not be independently verified from open sources; the 4th edition is known to cover incomplete-markets models in this part of the book.
- **Verdict:** OK
- **Note:** Book-level citation (authors, publisher, year, edition) is confirmed; chapter number is unverified but plausible.

## Main Message Audit

> "EEI is a fixed point for the same buffer-stock household. It iterates $W_a(a)$ instead of the value level. Low-wealth households consume more of a transfer. High-wealth households smooth toward the perfect-foresight MPC $\kappa^{\ast}\approx0.041$. The computational lesson is simple. The envelope condition can be an update rule. EGP is faster here because it uses an analytic inverse. All three methods agree up to the fine-grid gap."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| EEI is a fixed point for the same buffer-stock household | Equations (Bellman, Euler, envelope) and Solution Method (algorithm) | OK |
| It iterates $W_a(a)$ instead of the value level | Solution Method algorithm - envelope step produces $W_a$, no value level stored | OK |
| Low-wealth households consume more of a transfer | Results table (mean MPC 0.220 vs. PF limit 0.041) and borrowing-limit discussion | OK |
| High-wealth households smooth toward $\kappa^{\ast}\approx0.041$ | Results table (Perfect-foresight MPC limit 0.0413) | OK |
| The envelope condition can be an update rule | Solution Method - algorithm iterates via envelope step directly | OK |
| EGP is faster here because it uses an analytic inverse | Equations / Method | OVERREACH |
| All three methods agree up to the fine-grid gap | Results (convergence figure and table show EEI vs. fine-grid gap; VFI convergence shown) | OK |

Issues:
- "EGP is faster here" is an OVERREACH. The Results section explicitly states "This is a fixed-point comparison, not a timing claim," yet the Takeaway asserts a timing advantage for EGP. The tutorial demonstrates that EGP avoids bisection (code confirms `u_prime_inv` is used analytically), but no wall-clock timing comparison is reported. The claim is plausible but goes beyond what the tutorial's own results demonstrate.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $a$ | Equations, line 1: "enters with assets $a$" | Yes, at first use | |
| $y_j$ | Equations, line 1: "IID income $y_j$" | Yes, at first use | |
| $\pi_j$ | Equations, line 2: "probabilities $\pi_j$" | Yes, at first use | |
| $n_y$ | Equations, line 2: "$\{y_1,\dots,y_{n_y}\}$" | Yes, at first use | |
| $R$ | Equations, line 3: "$R = 1+r$" | Yes, defined inline | |
| $r$ | Equations, line 3 (via $R=1+r$); Model Setup table | Partial - introduced implicitly, defined in table | Acceptable; within 50 lines |
| $V(a,y_j)$ | Equations, Bellman equation | Implicit (standard value function notation) | |
| $a'$ | Equations, Bellman equation | Standard next-period notation | |
| $\underline a$ | Equations, Bellman constraint | Yes, "borrowing limit" in surrounding text | |
| $\beta$ | Equations, Bellman equation | Late - defined in Model Setup table | Acceptable; ~44 lines from first use |
| $W(a')$ | Equations, Bellman equation | Yes, defined inline as $\sum \pi_\ell V(a',y_\ell)$ | |
| $\pi_\ell$ | Equations, definition of $W(a')$ | Yes - same probabilities as $\pi_j$, $\ell$ is dummy index | |
| $g(a,y_j)$ | Equations: "The policy is $g(a,y_j)$" | Yes, at first use | |
| $c(a,y_j)$ | Equations: "Consumption is $c(a,y_j) = Ra+y_j-g(a,y_j)$" | Yes, at first use | |
| $\gamma$ | Equations, CRRA formula $u(c) = \frac{c^{1-\gamma}-1}{1-\gamma}$ | Late - defined in Model Setup table | Acceptable; ~33 lines from first use |
| $u(c)$ | Equations, CRRA block | Yes, defined by formula | |
| $u'(c)$ | Equations, CRRA block | Yes, defined as $c^{-\gamma}$ | |
| $(u')^{-1}(\mu)$ | Equations, CRRA block | Yes, defined as $\mu^{-1/\gamma}$; $\mu$ is local dummy | |
| $W_a(a)$ | Overview: "marginal continuation value $W_a(a)$" | Yes, described in Overview; formula in envelope condition | |
| $V_a(a,y_\ell)$ | Equations, envelope condition | Implicit (partial derivative of $V$); standard notation | |
| $\mu_y$ | Model Setup table: "Income mean $\mu_y$" | Yes, in table | $\mu$ used earlier as dummy in $(u')^{-1}(\mu)$; different symbol, no clash |
| $\sigma_y$ | Model Setup table: "Income s.d. $\sigma_y$" | Yes, in table | |
| $\bar a$ | Model Setup table: "Upper grid bound $\bar a$ \| 50.0" | Yes - upper grid bound | **Overloaded**: Results reuses $\bar a$ for mean assets (0.41) |
| $\kappa^{\ast}$ | Takeaway: "perfect-foresight MPC $\kappa^{\ast}\approx0.041$" | No | Never defined in equations or setup |

Flagged issues:
- $\bar a$ is overloaded: Model Setup table defines it as the upper asset grid bound (50.0), but the Results section uses it for mean assets ("Mean assets are $\bar a = 0.41$"). These are different quantities sharing the same symbol.
- $\kappa^{\ast}$ is used in the Takeaway ("perfect-foresight MPC $\kappa^{\ast}\approx0.041$") without definition anywhere in the tutorial. The value appears in the Results table as "Perfect-foresight MPC limit | 0.0413" but the symbol is never introduced.

## Summary

The tutorial has 1 MAJOR citation error, 1 OVERREACH in the main message, and 2 notation flags. The MAJOR issue is the first reference: the cited paper "Maliar, L. and Maliar, S. (2013), JEDC 37(7), 1439-1459" does not exist as written - the title belongs to Arellano, Maliar, Maliar & Tsyrennikov (2016), JEDC 69, 436-459, which is the paper that should be cited. The OVERREACH is the Takeaway's claim "EGP is faster here" when the Results section explicitly states the comparison is not a timing claim. The two notation issues are: $\bar a$ is overloaded between the Model Setup (upper grid bound, 50.0) and the Results (mean assets, 0.41), and $\kappa^{\ast}$ is used in the Takeaway without prior definition. The single most important fix is correcting the first reference to cite Arellano, Maliar, Maliar & Tsyrennikov (2016) with the correct journal, year, and page numbers.
