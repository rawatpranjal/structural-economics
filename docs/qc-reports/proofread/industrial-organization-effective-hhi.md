# Proofread: industrial-organization/effective-hhi/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:42:00Z._

## Paper / Source Verification

### U.S. Department of Justice & Federal Trade Commission (2023). *Merger Guidelines*.

- **Located:** https://www.justice.gov/atr/2023-merger-guidelines
- **Tutorial claims:** The 2023 DOJ/FTC Merger Guidelines treat HHI above 1,800 as highly concentrated; an HHI increase above 100 points is significant for the structural presumption; below 1,000 is unconcentrated; between 1,000 and 1,800 is moderately concentrated.
- **Source says:** The 2023 Merger Guidelines (released December 18, 2023) establish HHI > 1,800 as highly concentrated, confirm a structural presumption when delta-HHI > 100 in a highly concentrated market, and set the unconcentrated threshold at HHI < 1,000.
- **Verdict:** OK
- **Note:** All four HHI thresholds stated in the tutorial are consistent with the 2023 guidelines.

---

### Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.

- **Located:** https://mitpress.mit.edu/9780262200714/the-theory-of-industrial-organization/
- **Tutorial claims:** Cited as a supporting reference for the industrial-organization treatment underlying the tutorial.
- **Source says:** Published 1988 by MIT Press. Chapter 5 covers short-run price competition (Bertrand competition and related pricing models), which is the game-theoretic foundation for the Bertrand Nash pricing model used in the tutorial.
- **Verdict:** OK
- **Note:** Year, publisher, and chapter subject are all correct.

---

## Main Message Audit

> HHI is transparent: it converts ownership shares into a concentration number and gives a closed-form delta for mergers. Ownership aggregation can raise HHI even when the demand model implies no price effect. Once products substitute, the same ownership change moves prices through the Bertrand FOC.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| HHI converts ownership shares into a concentration number | Equations: $\text{HHI}=10{,}000\sum_{f}s_f^2$ defined explicitly | OK |
| Gives a closed-form delta for mergers | Equations: $\Delta\text{HHI}=20{,}000 s_a s_b$ shown explicitly | OK |
| Ownership aggregation can raise HHI even when the demand model implies no price effect | Results: segmented case shows delta-HHI = 2,812 with 0.00% merged-price change | OK |
| Same ownership change moves prices through the Bertrand FOC when products substitute | Results: differentiated case shows 1.69% price increase; Equations: Bertrand FOC stated | OK |

Issues:
- None. All clauses are demonstrated by equations or results shown in the tutorial.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $f$ | Equations, "Let firms be indexed by $f=1,\ldots,F$" | Yes | Defined at first use |
| $F$ | Equations, same sentence as $f$ | Yes | Defined at first use |
| $s_f$ | Equations, "market shares $s_f$ measured as fractions that sum to one" | Yes | Defined at first use |
| $N_{\text{eff}}$ | Equations, formula block | Yes | Defined by formula |
| $a$ (firm label) | Equations, "If firms $a$ and $b$ merge" | Partial | Used as a firm-index label; defined contextually |
| $b$ | Equations, "If firms $a$ and $b$ merge" | Partial | Same as $a$ above |
| $\Delta\text{HHI}$ | Equations, formula block | Yes | Defined by formula |
| $j$ | Equations, "product $j$ belongs to firm $f(j)$" | Yes | Defined at first use |
| $f(j)$ | Equations, same sentence as $j$ | Yes | Defined at first use |
| $q_j$ | Equations, "sells quantity $q_j$" | Yes | Defined at first use |
| $\ell$ | Equations, share-aggregation denominator | Implicit | Standard dummy summation index; acceptable |
| $a$ (demand intercept) | Equations, "$q(p)=a+Dp$" | Partial | **Overloads** firm-label $a$ introduced three lines earlier |
| $D$ | Equations, "$q(p)=a+Dp$" | Partial | Matrix structure given via $D_{jj}$, $D_{jk}$; named only in code |
| $\alpha$ | Equations, "$D_{jj}=\alpha<0$" | Yes | Defined at first use |
| $\beta$ | Equations, "$D_{jk}=\beta\geq 0$" | Yes | Defined at first use |
| $p$ | Equations, argument of $q(p)$ | Implicit | Inferable as price vector from context; not stated explicitly |
| $\Omega_{jk}$ | Equations, "Let $\Omega_{jk}=1$ if products $j$ and $k$ are commonly owned" | Yes | Defined at first use |
| $\Omega$ | Equations, Bertrand FOC $(\Omega\circ D^{\top})$ | Implicit | Matrix $\Omega_{jk}$ is defined; matrix form implicitly follows |
| $\circ$ | Equations, Bertrand FOC $(\Omega\circ D^{\top})$ | No | Hadamard (element-wise) product; not defined anywhere in the README |
| $c$ | Equations, Bertrand FOC $(p-c)$ | No | Appears in the FOC without definition; cost vector is named only in pseudocode ("costs c") |
| $N$ | Results, "HHI is exactly $10{,}000/N$" | Implicit | Used for the equal-sized firm count; related to but distinct from $N_{\text{eff}}$ |

Flagged issues:
- **$a$ overloaded**: In the Equations section, $a$ is first introduced as a firm-index label ("If firms $a$ and $b$ merge"), then reused four lines later as the demand intercept vector ("$q(p)=a+Dp$"). These are distinct objects: a scalar firm identifier and a vector of demand intercepts.
- **$c$ undefined**: The cost vector $c$ appears in the Bertrand FOC without any prior definition in the Equations section. It is mentioned only in pseudocode as "costs c".
- **$\circ$ undefined**: The Hadamard (element-wise) product operator $\circ$ in $(\Omega\circ D^{\top})$ is not defined or named anywhere in the README.

## Summary

Both references check out: the DOJ/FTC (2023) Merger Guidelines are correctly cited with accurate HHI thresholds, and Tirole (1988) is correctly identified as MIT Press with Chapter 5 covering Bertrand price competition. The main message is fully supported by the tutorial's equations and results tables, with no overreach or unsupported clauses. The notation audit finds 0 MAJOR reference issues, 0 MINOR reference issues, 0 NOT FOUND references, 0 OVERREACH claims, and 3 notation flags: $a$ is overloaded (firm label then demand intercept vector), $c$ (cost vector) is undefined in the Equations section, and $\circ$ (Hadamard product) is used but never defined. The single most important fix is disambiguating $a$: the demand intercept vector should use a different letter (such as $\mathbf{a}$ or a distinct symbol like $\mu$) to avoid collision with the firm-index label introduced in the same section.
