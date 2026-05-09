# Proofread: industrial-organization/theory-of-the-firm/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T02:50:00Z._

## Paper / Source Verification

### Williamson, O. (1975). *Markets and Hierarchies*. Free Press.

- **Located:** https://openlibrary.org/books/OL5062431M/Markets_and_hierarchies_analysis_and_antitrust_implications
- **Tutorial claims:** Cited as a foundational reference for firm boundaries, transaction costs, and vertical integration driven by asset specificity.
- **Source says:** Full title is "Markets and Hierarchies: Analysis and Antitrust Implications - A Study in the Economics of Internal Organization." Publisher is Free Press (The Free Press), year 1975. Covers asset specificity, opportunism, and governance as the deciding factors between markets and hierarchies.
- **Verdict:** OK
- **Note:** The tutorial uses the standard shortened title; "Free Press" is the correct abbreviated publisher name.

### Grossman, S., and Hart, O. (1986). The Costs and Benefits of Ownership. *Journal of Political Economy*, 94(4), 691-719.

- **Located:** https://www.journals.uchicago.edu/doi/abs/10.1086/261404
- **Tutorial claims:** Cited for the analysis of ownership costs and benefits in the context of vertical integration.
- **Source says:** Full title is "The Costs and Benefits of Ownership: A Theory of Vertical and Lateral Integration." Volume 94, Issue 4, pages 691-719, 1986. All bibliographic details confirmed correct.
- **Verdict:** MINOR
- **Note:** The subtitle "A Theory of Vertical and Lateral Integration" is omitted from the citation.

### Hart, O., and Moore, J. (1990). Property Rights and the Nature of the Firm. *Journal of Political Economy*, 98(6), 1119-1158.

- **Located:** https://ideas.repec.org/a/ucp/jpolec/v98y1990i6p1119-58.html
- **Tutorial claims:** Cited for property rights theory and the nature of the firm.
- **Source says:** Volume 98, Issue 6, pages 1119-1158, year 1990. All bibliographic details confirmed correct.
- **Verdict:** OK
- **Note:** All details are correct.

### Lecture 6 Slides 2023: Theory of the Firm and incomplete contracts.

- **Located:** NOT FOUND
- **Tutorial claims:** Cited as supplementary material on the theory of the firm and incomplete contracts.
- **Source says:** N/A
- **Verdict:** NOT FOUND
- **Note:** This is an informal, unattributed reference with no author, publisher, or DOI; it cannot be located or verified externally.

## Main Message Audit

> "Firm boundaries follow the hold-up tradeoff. Market exchange works when assets are easy to redeploy. Integration pays when stronger control rights offset hierarchy cost. Long-term contracts fill the middle range."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Firm boundaries follow the hold-up tradeoff | Equations: $b_g(s)$ captures the fraction of the marginal return retained by the investor, and specificity reduces this share for spot exchange, modeling hold-up directly | OK |
| Market exchange works when assets are easy to redeploy | Results: governance regions show spot exchange wins for $s \lesssim 0.21$; figure and table both confirm spot contract is chosen at $s=0$ | OK |
| Integration pays when stronger control rights offset hierarchy cost | Equations and Results: $b_{\text{integration}}(s)$ stays near first-best while $F_{\text{integration}}(s)$ decreases with $s$; table shows integration chosen at $s=0.5$ and $s=1$ | OK |
| Long-term contracts fill the middle range | Results: contracts win for $0.21 \lesssim s \lesssim 0.37$; figure and table both confirm contract chosen at $s=0.3$ | OK |

Issues:
- None. All four clauses of the takeaway are supported by the README's equations and results.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $s$ | Equations, prose before eq. 1 | Yes - "Let $s$ denote asset specificity" | Clear |
| $g$ | Equations, prose before eq. 1 | Yes - "Let $g \in \mathcal{G}$ index spot exchange..." | Clear |
| $\mathcal{G}$ | Equations, prose before eq. 1 | Yes - implicitly with $g$, enumerated in same sentence | Clear |
| $x$ | Equations, prose before eq. 1 | Yes - "Relationship-specific investment $x$" | Clear |
| $V(x)$ | Equations, eq. 1 | Yes - defined by formula $\theta x - \frac{1}{2}x^2$ | Clear |
| $\theta$ | Equations, eq. 1 | Partial - appears in $V(x)$ without prior prose introduction; defined in Model Setup table 37 lines later as "Marginal productivity scale, $\theta=4$" | Within 50-line threshold; acceptable |
| $x^{\ast}$ | Equations, eq. 2 | Yes - labeled "First-best investment" in surrounding prose | Clear |
| $b_g(s)$ | Equations, prose after eq. 2 | Yes - "share of marginal value captured by the investor under governance $g$" | Clear |
| $W_g(s)$ | Equations, eq. 4 | Yes - labeled "Total surplus" in surrounding prose | Clear |
| $F_g(s)$ | Equations, prose before eq. 4 | Yes - "governance cost $F_g(s)$" | Clear |
| $b_{\text{spot}}(s)$, $b_{\text{contract}}(s)$, $b_{\text{integration}}(s)$ | Equations, eq. 5 | Yes - explicit linear formulas given | Clear |
| $F_{\text{spot}}(s)$, $F_{\text{contract}}(s)$, $F_{\text{integration}}(s)$ | Equations, eq. 6 | Yes - explicit linear formulas given | Clear |
| $g^{\ast}(s)$ | Equations, eq. 7 | Yes - labeled "selected governance form" | Clear |
| $W^{\ast}$ | Equations, eq. 8 | Yes - labeled "first-best surplus benchmark" | Clear |

Flagged issues:
- None. All symbols are defined at or near first use. $\theta$ is used in the Equations section before its explicit label appears in the Model Setup table, but the gap is 37 lines - within the 50-line threshold, so this is acceptable rather than a flagged issue.

## Summary

The tutorial is clean and internally consistent. Among the three formal references, all bibliographic details are correct except that the Grossman-Hart (1986) citation omits the subtitle "A Theory of Vertical and Lateral Integration" (1 MINOR, 0 MAJOR, 0 NOT FOUND for formal papers). The informal "Lecture 6 Slides 2023" entry cannot be verified and counts as 1 NOT FOUND. All four clauses of the takeaway are directly supported by the equations and results sections (0 OVERREACH, 0 UNSUPPORTED). Every notation symbol is defined at or near first use with no overloading or drift. The single most important fix is replacing or removing the unverifiable "Lecture 6 Slides 2023" citation with a public, citable reference.
