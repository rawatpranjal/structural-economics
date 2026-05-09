# Proofread: structural-econometrics/auction-valuation-recovery/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T08:30:00Z._

## Paper / Source Verification

### Perrigne, I. and Vuong, Q. (2019). Econometrics of Auctions and Nonlinear Pricing. *Annual Review of Economics*, 11, 27-54.

- **Located:** https://doi.org/10.1146/annurev-economics-080218-025702
- **Tutorial claims:** Survey reference for the econometrics of auctions; listed in references without a specific textual claim attached.
- **Source says:** Authors Isabelle Perrigne and Quang Vuong; Annual Review of Economics, Volume 11, 2019, pages 27-54.
- **Verdict:** OK
- **Note:** All bibliographic fields match the authoritative source exactly.

### Guerre, E., Perrigne, I., and Vuong, Q. (2000). Optimal Nonparametric Estimation of First-Price Auctions. *Econometrica*, 68(3), 525-574.

- **Located:** https://doi.org/10.1111/1468-0262.00123
- **Tutorial claims:** Foundational paper for the GPV nonparametric inversion applied in this tutorial; the inversion formula is attributed to this work.
- **Source says:** Authors Emmanuel Guerre, Isabelle Perrigne, Quang Vuong; Econometrica, Vol. 68, No. 3, May 2000, pages 525-574.
- **Verdict:** OK
- **Note:** All bibliographic fields match; the GPV inversion in the README is correctly attributed.

### Gentry, M., Komarova, T., and Schiraldi, P. (2023). Preferences and Performance in Simultaneous First-Price Auctions: A Structural Analysis. *Review of Economic Studies*, 90(2), 852-878.

- **Located:** https://doi.org/10.1093/restud/rdac030
- **Tutorial claims:** Listed as a reference for structural auction analysis; no specific result is attributed to it in the prose.
- **Source says:** Authors Matthew Gentry, Tatiana Komarova, Pasquale Schiraldi; Review of Economic Studies, Volume 90, Issue 2, March 2023, pages 852-878.
- **Verdict:** OK
- **Note:** All bibliographic fields match the Oxford University Press record.

### Hickman, B. R., Hubbard, T. P., and Saglam, Y. (2012). Structural Econometric Methods in Auctions: A Guide to the Literature. *Journal of Econometric Methods*, 1(1), 67-106.

- **Located:** https://doi.org/10.1515/2156-6674.1019
- **Tutorial claims:** Listed as a guide to the structural auction estimation literature; no specific result is attributed in the prose.
- **Source says:** Authors Brent R. Hickman, Timothy P. Hubbard, Yigit Saglam; Journal of Econometric Methods, Volume 1, Issue 1, August 2012, pages 67-106.
- **Verdict:** OK
- **Note:** All bibliographic fields match the De Gruyter record.

### Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.

- **Located:** https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1
- **Tutorial claims:** Textbook reference for auction theory foundations.
- **Source says:** Author Vijay Krishna; 2nd edition, 2009; published under the Academic Press imprint (an Elsevier imprint), ISBN 978-0-12-374507-1.
- **Verdict:** OK
- **Note:** "Academic Press" is the correct imprint on the book's title page; citing it as Academic Press is standard bibliographic practice.

## Main Message Audit

> "The tutorial uses equilibrium structure to undo that shading. It estimates the bid distribution, applies the GPV inversion, and checks recovered pseudo-values against simulated truth."
> "Observed first-price bids mix values with strategic shading. Under symmetric IPV assumptions and monotone bidding, the equilibrium first-order condition turns the bid CDF and density into pseudo-values. The exercise also shows the cost of the method. Recovery depends on a density estimate, so the edges of the bid support are fragile. Trimming is not cosmetic; it is part of making the structural inversion usable."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Equilibrium structure is used to undo bid shading | Equations (FOC derivation leading to GPV formula) | OK |
| Bid distribution is estimated | Solution Method (empirical CDF and KDE steps 3-4) | OK |
| GPV inversion is applied | Equations (inversion formula), Solution Method (steps 5-7) | OK |
| Recovered pseudo-values checked against simulated truth | Results (figure 3, diagnostics table with RMSE and correlation) | OK |
| Observed bids mix values with strategic shading | Results (figure 1 shows bid distribution shifted left of values) | OK |
| Symmetric IPV and monotone bidding are the key assumptions | Equations (stated at outset of derivation) | OK |
| FOC turns bid CDF and density into pseudo-values | Equations (derivation from FOC through change of variables) | OK |
| Recovery depends on a density estimate | Solution Method (KDE step 4), Equations (density estimate is least stable near boundaries) | OK |
| Edges of bid support are fragile | Results (figure 3 shows larger errors near remaining boundaries after trimming) | OK |
| Trimming is part of making the inversion usable, not cosmetic | Equations (trimming motivation stated), Model Setup (5% trim row), Results (error pattern) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $n$ | Equations, "There are $n$ risk-neutral bidders" | Yes, at first use | |
| $v_i$ | Equations, "$v_i \sim F_v$ with density $f_v$" | Yes, at first use | |
| $F_v$ | Equations, same sentence as $v_i$ | Yes, at first use | CDF of value distribution |
| $f_v$ | Equations, same sentence as $v_i$ | Yes, at first use | Density of value distribution |
| $v$ | Equations, "type $v$ submits $b=s(v)$" | Yes, contextual (general type variable) | Consistent with $v_i$ as specific draw |
| $b$ | Equations, "$b=s(v)$" | Yes, at first use | Generic bid |
| $s(\cdot)$ | Equations, "$b=s(v)$" | Yes, as equilibrium bidding strategy | |
| $x$ | Equations, "deviate by bidding $s(x)$, as if its type were $x$" | Yes, defined in surrounding prose | Deviation type |
| $\pi(v,x)$ | Equations, "Expected payoff from the deviation is $\pi(v,x)$..." | Yes, at first use | |
| $s'(v)$ | Equations, derivative of $\pi$ w.r.t. $x$ evaluated at $x=v$ | Yes, implicitly from differentiation context | |
| $G$ | Equations, "Let $G$ and $g$ be the CDF and density of bids" | Yes, at first use | |
| $g$ | Equations, same sentence as $G$ | Yes, at first use | |
| $b_i$ | Equations, "For $b_i=s(v_i)$, the GPV inversion is" | Yes, at first use | |
| $\hat{v}_i$ | Equations, GPV inversion formula | Yes, as recovered pseudo-value | |
| $\hat{G}$ | Equations, GPV inversion formula (line 74) | Partial - described in Solution Method (~line 94), within 50 lines | |
| $\hat{g}$ | Equations, GPV inversion formula (line 74) | Partial - described in Solution Method (~line 94), within 50 lines | |
| $N$ | Solution Method, "Let $N$ be the number of observed bids" | Yes, at first use | |
| $I_q$ | Solution Method pseudocode, step 5 | Yes, defined at first use in pseudocode | |
| $Q_q$ | Solution Method pseudocode, step 5 | Partial - not explicitly named, but "trim q" in Input line and context make meaning clear | q-th quantile of bids |

Flagged issues:
- None. $\hat{G}$ and $\hat{g}$ are used in the inversion formula before their algorithmic definitions in Solution Method, but the definitions appear within approximately 20 lines and $G$, $g$ are defined immediately before the formula. Classified as Partial/Acceptable per the 50-line rule. $Q_q$ is self-evident from context within the same pseudocode block and is not a flagged issue.

## Summary

The tutorial is in excellent shape. All five cited references verified against authoritative sources with no bibliographic errors. The main message is fully supported by the Equations, Solution Method, and Results sections - every clause of the Overview and Takeaway is demonstrated within the tutorial rather than asserted without backing. Notation is consistent and complete: every symbol is either defined at first use or defined within 20 lines of first use, and no symbol is used for two distinct objects. There are 0 MAJOR issues, 0 MINOR issues, 0 NOT FOUND references, 0 OVERREACH clauses, and 0 UNSUPPORTED clauses. The tutorial requires no corrective action.
