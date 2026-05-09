# Proofread: industrial-organization/bertrand-logit-demand/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:51:00Z._

## Paper / Source Verification

### Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2).

- **Located:** https://ideas.repec.org/a/rje/randje/v25y1994isummerp242-262.html
- **Tutorial claims:** The tutorial cites Berry (1994) as the source for the logit demand framework, including the share equation $s_j = \exp(\delta_j)/(1+\sum \exp(\delta_k))$ and the calibration of mean utilities from observed market shares.
- **Source says:** Steven T. Berry, "Estimating Discrete-Choice Models of Product Differentiation," *RAND Journal of Economics*, vol. 25, no. 2 (Summer 1994), pp. 242-262. The paper develops the inversion of market shares to recover mean utilities, which is exactly what the tutorial implements in step 3 of its algorithm.
- **Verdict:** MINOR
- **Note:** The citation omits the page range; correct pages are pp. 242-262.

### Werden, G. and Froeb, L. (1994). "The Effects of Mergers in Differentiated Products Industries." *Journal of Law, Economics, & Organization*, 10(2).

- **Located:** https://academic.oup.com/jleo/article-abstract/10/2/407/842179
- **Tutorial claims:** The tutorial cites Werden and Froeb (1994) for the merger simulation methodology: calibrating logit demand from observed data and resolving Bertrand pricing FOCs after changing the ownership matrix.
- **Source says:** Gregory J. Werden and Luke M. Froeb, "The Effects of Mergers in Differentiated Products Industries: Logit Demand and Merger Policy," *The Journal of Law, Economics, and Organization*, vol. 10, no. 2 (1994), pp. 407-426. The paper introduces exactly the ownership-matrix merger simulation approach the tutorial implements.
- **Verdict:** MINOR
- **Note:** The citation truncates the title (omits subtitle ": Logit Demand and Merger Policy") and omits the page range; correct pages are pp. 407-426.

## Main Message Audit

> Merger simulation here is an ownership-matrix exercise inside a pricing FOC. Diversion gives the merged firm an upward pricing incentive. Cost reductions push in the other direction.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| "Merger simulation here is an ownership-matrix exercise inside a pricing FOC" | Equations section (the $\Omega$ matrix definition and FOC equation) and Solution Method step 6 ("Replace Omega and c when ownership or efficiencies change") | OK |
| "Diversion gives the merged firm an upward pricing incentive" | Results section: merged products raise prices; figure captions explain that a lost sale to the partner product no longer fully leaves the firm | OK |
| "Cost reductions push in the other direction" | Results table: "Merger 1+2, lower costs" shows avg price 1.0241 vs. 1.0456 for the plain merger scenario | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $J$ | "There are $J$ inside products" | Yes - inline | |
| $p_j$ | "Product $j$ has price $p_j$" | Yes - inline | |
| $c_j$ | "marginal cost $c_j$" | Yes - inline | |
| $\xi_j$ | "mean non-price utility $\xi_j$" | Yes - inline | |
| $\alpha$ | "With $\alpha<0$, mean utility is" | Partial - sign stated; calibrated value appears in the Model Setup table | Acceptable |
| $\delta_j(p)$ | Equation $\delta_j(p)=\xi_j+\alpha p_j$ | Yes - by equation | |
| $s_j(p)$ | Logit share equation | Yes - by equation | |
| $s_0(p)$ | Alongside $s_j(p)$ in the share equation | Yes - by equation | |
| $\Omega_{jk}$ | "Let $\Omega_{jk}=1$ when products $j$ and $k$ are controlled by the same firm" | Yes - inline | |
| $\Delta_{jk}$ | "Let $\Delta_{jk}=\partial s_j/\partial p_k$" | Yes - inline | |
| $\circ$ | Markup equation $(\Omega\circ\Delta')^{-1}s$ | No | Hadamard (element-wise) product; never defined in the README |
| $D_{j\to k}$ | "The diversion ratio records where a lost sale goes. For $j\neq k$," | Yes - by equation | |

Flagged issues:
- $\circ$: used in the markup equation $p-c=-(\Omega\circ\Delta')^{-1}s$ without definition. The symbol denotes the Hadamard (element-wise) product of two matrices. No definition or description appears anywhere in the README.

## Summary

The tutorial is internally sound and the cited papers are correctly attributed to the methodologies they implement. There are 0 MAJOR issues, 2 MINOR citation issues, and 0 NOT FOUND references. Berry (1994) omits the page range (pp. 242-262). Werden and Froeb (1994) truncates the title (missing the subtitle ": Logit Demand and Merger Policy") and omits the page range (pp. 407-426). All three clauses of the takeaway are directly supported by the Equations, Solution Method, and Results sections. There is one notation issue: the operator $\circ$ (Hadamard element-wise product) appears in the markup equation without any definition, and defining it at first use is the single most important fix.
