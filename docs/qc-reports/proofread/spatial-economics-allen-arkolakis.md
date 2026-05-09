# Proofread: spatial-economics/allen-arkolakis/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:43:00Z._

## Paper / Source Verification

### Allen, T. and Arkolakis, C. (2014). *Trade and the Topography of the Spatial Economy*. Quarterly Journal of Economics 129(3), 1085-1140. https://doi.org/10.1093/qje/qju016.

- **Located:** https://academic.oup.com/qje/article-abstract/129/3/1085/1818077
- **Tutorial claims:** This is the primary published source for the spatial equilibrium framework combining gravity trade and free labor mobility across locations.
- **Source says:** Authors, year (2014), journal (QJE), volume 129, issue 3, pages 1085-1140, and DOI all confirmed correct. The paper combines gravity trade with labor mobility to determine how economic activity distributes spatially.
- **Verdict:** OK
- **Note:** The tutorial's reference to "equations (11) and (12)" and the "Hammerstein reduction" could not be verified from publicly accessible abstracts; however, this description is consistent with the paper's known methodology (balanced trade and mobility conditions reduced to a Hammerstein integral equation under symmetry).

### Allen, T. and Arkolakis, C. (2013). *Trade and the Topography of the Spatial Economy*. NBER Working Paper 19181. https://www.nber.org/papers/w19181.

- **Located:** https://www.nber.org/papers/w19181
- **Tutorial claims:** NBER working paper version of the same paper, issued 2013 before the 2014 QJE publication.
- **Source says:** Authors, year (2013), working paper number (w19181), and title all confirmed correct via the NBER page.
- **Verdict:** OK
- **Note:** None.

### Redding, S. J. and Rossi-Hansberg, E. (2017). *Quantitative Spatial Economics*. Annual Review of Economics 9, 21-58.

- **Located:** https://ideas.repec.org/a/anr/reveco/v9y2017p21-58.html
- **Tutorial claims:** Survey reference cited in the References block without a specific claim in the body of the tutorial.
- **Source says:** Authors (Redding and Rossi-Hansberg), year (2017), journal (Annual Review of Economics), volume (9), and page range (21-58) all confirmed correct.
- **Verdict:** OK
- **Note:** None.

## Main Message Audit

> "For policy, the model is useful because it keeps three outcomes visible at the same time. Lower trade costs improve access and raise real utility, but they can also shift activity across space. Agglomeration can raise productivity when workers cluster. It can also create congestion and concentration risk. When dispersion forces are strong, the spatial outcome is more predictable. When agglomeration is strong, history and initial conditions matter more. A transport policy should therefore be judged by welfare, concentration, and geographic redistribution together. A higher common utility number does not by itself say which locations gain population or how exposed the economy becomes to concentration."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Lower trade costs improve access and raise real utility | Results (counterfactual table: +5.38% and +2.88% welfare) | OK |
| Lower trade costs can shift activity across space | Results (counterfactual table: center share changes of -1.0 pp and -7.4 pp) | OK |
| Agglomeration raises productivity when workers cluster | Equations ($A_i = \bar A_i L_i^\alpha$ with $\alpha > 0$) | OK |
| Congestion and concentration risk from agglomeration | Equations ($u_i = \bar u_i L_i^\beta$ with $\beta < 0$) and diagnostics (HHI rises from 0.099 to 0.297) | OK |
| Dispersion-dominant case yields predictable spatial outcome | Results (relocation diagnostic shows convergence from multiple starts) | OK |
| Strong agglomeration makes history and initial conditions matter more | Results (path-dependence figure; left vs. right starts differ by 0.304 in final labor) | OK |
| Policy should be judged by welfare, concentration, and redistribution together | Results (counterfactual table reports all three) | OK |
| Higher common utility alone does not reveal distributional effects | Results (agglomeration-strong case: welfare +2.88% while HHI and center share both fall) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $N$ | Equations, first sentence | Yes - inline prose | "number of locations" |
| $i$ | Equations, index sentence | Yes - inline prose | location index |
| $j$ | Equations, destinations sentence | Yes - inline prose | destination index |
| $L_i$ | Equations, location description | Yes - inline prose | "workers" at location $i$ |
| $w_i$ | Equations, location description | Yes - inline prose | "wage" at location $i$ |
| $A_i$ | Equations, location description | Yes - inline prose | "productivity" at location $i$ |
| $u_i$ | Equations, location description | Yes - inline prose | "amenity" at location $i$ |
| $T_{ij}$ | Equations, shipping sentence | Yes - inline prose | "units required at the origin per unit shipped" |
| $\bar A_i$ | Equations, fundamentals sentence | Yes - inline prose | "exogenous productivity fundamental" |
| $\bar u_i$ | Equations, fundamentals sentence | Yes - inline prose | "exogenous amenity fundamental" |
| $\alpha$ | Equations, productivity equation ($A_i = \bar A_i L_i^\alpha$) | Yes - 2 lines after first use | "raises productivity when workers crowd in" |
| $\beta$ | Equations, amenity equation ($u_i = \bar u_i L_i^\beta$) | Yes - 2 lines after first use | "captures congestion" |
| $\sigma$ | Equations, CES price index equation ($P_j^{1-\sigma}$) | Partial - defined in Model Setup table ~46 lines later | Late-defined; falls just within the 50-line acceptable threshold |
| $P_j$ | Equations, CES price index equation | Yes - described as "CES price index in destination $j$" | OK |
| $k$ | Equations, trade share denominator ($\sum_k$) | Implicit - standard dummy index | No flag; conventional summation dummy |
| $\pi_{ij}$ | Equations, trade share definition | Yes - "share of destination $j$'s spending that goes to origin $i$" | OK |
| $V$ | Equations, mobility condition ($w_i u_i / P_i = V$) | No - implied by surrounding prose but never named | Undefined; see flagged issues |
| $x_i$ | Model Setup table | Yes - in table | "Geographic position" |
| $\gamma_1, \gamma_2$ | Model Setup table | Yes - in table | "Allen-Arkolakis stability and uniqueness summary terms" |

Flagged issues:
- $V$ is undefined. The mobility equation $w_i u_i / P_i = V$ appears in the Equations section with surrounding prose saying "workers move until real utility is the same in every inhabited location," which implies $V$ is the common real utility level, but the symbol is never explicitly named or defined. A one-line prose definition (e.g., "where $V$ is the common real utility level") would close the gap.
- $\sigma$ is first used in the CES price index equation in the Equations section and defined only in the Model Setup table, roughly 46 lines later. This is borderline under the 50-line threshold and is classified as Partial rather than a hard flag.

## Summary

All three cited references are verified correct with no bibliographic errors. The tutorial's main message is fully supported by its equations, results, and counterfactual tables - no OVERREACH or UNSUPPORTED clauses were found. The single substantive notation issue is that $V$ (the common real utility level in the mobility condition) is used in the Equations section without being named or defined; the most important fix is to add an inline definition for $V$ immediately after its first appearance in the mobility equation. The late definition of $\sigma$ (used in the CES price index before being defined in the Model Setup table) is borderline acceptable under the 50-line rule and is not a hard flag. Overall verdict: 0 MAJOR, 0 MINOR paper issues, 0 OVERREACH/UNSUPPORTED message issues, 1 undefined symbol ($V$), 1 borderline late-defined symbol ($\sigma$).
