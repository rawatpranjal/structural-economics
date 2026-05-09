# Proofread: dynamic-programming/optimal-growth/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T18:42:00Z._

## Paper / Source Verification

### Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 2 & 4.

- **Located:** https://www.hup.harvard.edu/books/9780674750968
- **Tutorial claims:** Foundational reference for recursive methods; cited as the primary source for Ch. 2 (overview of deterministic optimal growth) and Ch. 4 (dynamic programming under certainty, Bellman equation).
- **Source says:** Harvard University Press confirmed 1989 publication. Chapter 2 ("An Overview") includes section 2.1 "A Deterministic Model of Optimal Growth." Chapter 4 ("Dynamic Programming under Certainty") covers the Principle of Optimality and Euler equations.
- **Verdict:** OK
- **Note:** All bibliographic details confirmed; chapter content matches the tutorial's use.

---

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** Supplementary reference for recursive methods in macroeconomics, 4th edition, Chapter 3.
- **Source says:** MIT Press confirms the 4th edition published 2018. Chapter 3 is a foundational chapter on dynamic programming theory (the Bellman equation). Optimal growth *applications* are treated in Chapter 15 ("Economic Growth"), not Chapter 3.
- **Verdict:** OK
- **Note:** The chapter reference is correct for the Bellman-equation theory the tutorial teaches; no claim is made that Ch. 3 covers optimal growth applications.

---

### Ramsey, F. P. (1928). A Mathematical Theory of Saving. *Economic Journal*, 38(152), 543-559.

- **Located:** https://academic.oup.com/ej/article-abstract/38/152/543/5282967
- **Tutorial claims:** Seminal paper on optimal saving; implicitly provides the consumption-savings problem that this model solves.
- **Source says:** Oxford Academic and JSTOR confirm: *Economic Journal*, volume 38, issue 152, pages 543-559, 1928. The paper addresses how much of its income a nation should save.
- **Verdict:** OK
- **Note:** All bibliographic details (journal, volume, issue, pages, year) are correct.

---

### Cass, D. (1965). Optimum Growth in an Aggregative Model of Capital Accumulation. *Review of Economic Studies*, 32(3), 233-240.

- **Located:** https://academic.oup.com/restud/article-abstract/32/3/233/1551001
- **Tutorial claims:** Foundational optimal growth paper contributing the Ramsey-Cass-Koopmans framework.
- **Source says:** *Review of Economic Studies*, volume 32, issue 3, pages 233-240, 1965. The paper establishes turnpike properties of optimal capital accumulation paths in an aggregated model.
- **Verdict:** OK
- **Note:** All bibliographic details confirmed; one of the defining papers in optimal growth theory.

---

### Koopmans, T. C. (1965). On the Concept of Optimal Economic Growth. In *The Econometric Approach to Development Planning*. North-Holland.

- **Located:** https://elischolar.library.yale.edu/cowles-discussion-paper-series/392/ (Cowles Foundation Discussion Paper 163; volume confirmed via Stanford catalog and Pontifical Academy of Sciences records)
- **Tutorial claims:** Koopmans' contribution to the Ramsey-Cass-Koopmans framework on optimal growth.
- **Source says:** The chapter appears in "The Econometric Approach to Development Planning" (1965), North-Holland Publishing Company, Amsterdam, pages 225-287. Based on a 1963 Study Week organized by the Pontifical Academy of Sciences.
- **Verdict:** OK
- **Note:** Publisher, year, and volume title all confirmed.

---

## Main Message Audit

> "The one-capital growth problem makes saving productive. In the log Cobb-Douglas case, the exact policy saves αβ = 0.27 of output. VFI recovers that rule to grid accuracy. The example shows how to audit a Bellman solver when an exact benchmark exists."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| "The one-capital growth problem makes saving productive" | Equations (resource constraint, diminishing-returns production) | OK |
| "In the log Cobb-Douglas case, the exact policy saves αβ = 0.27 of output" | Equations (Euler equation derivation yields s = αβ; Model Setup: α=0.3, β=0.9) | OK |
| "VFI recovers that rule to grid accuracy" | Results (policy gap outside bottom decile is 2.87e-02; table shows k' errors ~0.005-0.020 on a 500-point grid) | OK |
| "The example shows how to audit a Bellman solver when an exact benchmark exists" | Solution Method + Results (full pointwise comparison of numerical vs closed-form) | OK |

Issues:
- None. All clauses are directly supported by material in the README.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k$ | Overview | Implicit | Described as "capital" in context before formal definition in Equations |
| $g(k)$ | Overview | Yes (Overview) | Policy rule for next-period capital; re-stated more formally in Equations |
| $c^{\ast}(k)$ | Overview | Yes (Overview) | Defined as $Ak^{\alpha}-g(k)$ in Overview |
| $A$ | Overview ("$Ak^{\alpha}$") | No (Overview) | First used in Overview expression "$Ak^{\alpha}$"; formally defined ("Total factor productivity, $A>0$") only in Equations |
| $\alpha$ | Overview ("$Ak^{\alpha}$") | No (Overview) | Same as above; formally defined ("$\alpha\in(0,1)$") only in Equations |
| $\beta$ | Overview ("saving rate $\alpha\beta$") | No (Overview) | First used in Overview; formally defined ("$\beta\in(0,1)$") only in Equations |
| $k_t$, $y_t$, $c_t$, $k_{t+1}$ | Equations | Yes (Equations) | All defined in their introducing sentence |
| $V(k)$ | Equations | Yes (Equations) | Value function; introduced in the Bellman equation |
| $k'$ | Equations | Yes (Equations) | Choice variable for next-period capital |
| $u'(\cdot)$ | Equations (Euler equation) | Partial | $u$ specified as log utility above; $u'$ used without explicit "$u'(c)=1/c$" step |
| $f'(k)$ | Equations (Euler equation) | Yes | Defined inline as "$\alpha A k^{\alpha-1}$" |
| $s$ | Equations (conjecture) | Yes (inline) | Saving rate; introduced and solved for in the same paragraph |
| $E$ | Equations | Yes (Equations) | Value function intercept; formula given |
| $B$ | Equations | Yes (Equations) | Value function slope; $B=\alpha/(1-\alpha\beta)$ |
| $k_{ss}$ | Equations | Yes (Equations) | Steady-state capital; formula given |
| $c_{ss}$ | Equations | Yes (Equations) | Steady-state consumption; formula given |
| $\varepsilon$ | Model Setup (table) | Yes (table) | Convergence tolerance |
| $N_k$, $N_{k'}$ | Model Setup (table) | Yes (table) | Grid sizes |
| $T_{sim}$ | Model Setup (table) | Yes (table) | Simulation horizon |
| $k_0$ | Model Setup (table) | Yes (table) | Initial capital; stated as $0.1\,k_{ss}\approx0.9952$ |
| $TV$ | Solution Method | Yes (Solution Method) | Bellman operator; defined in its introducing equation |
| $k_{min}$ | Solution Method (pseudocode) | Partial | Used in pseudocode as lower bound on $k'$; value 0.01 implicit from the Model Setup "k domain" row, but the symbol $k_{min}$ is not introduced before the pseudocode |
| $N_{kp}$ (pseudocode) | Solution Method (pseudocode) | - | Notation drift: the pseudocode uses "$N_{kp}$" but the Model Setup table uses "$N_{k'}$" for the same quantity |

Flagged issues:
- **$A$, $\alpha$, $\beta$ used in Overview before formal definition in Equations.** The Overview writes "$Ak^{\alpha}$" and "saving rate $\alpha\beta$" before any of the three are introduced or constrained. For a tutorial-style document this is a low-severity issue, but readers who pause on the Overview will encounter undefined symbols.
- **$k_{min}$ in pseudocode is not defined as a symbol.** Its value (0.01) is inferrable from the Model Setup table's "$k$ domain" row, but $k_{min}$ is never introduced by name or formula.
- **Notation drift: $N_{kp}$ (pseudocode) vs $N_{k'}$ (Model Setup table).** Both refer to the inner choice-grid size. Using two different names for the same quantity across adjacent sections is a minor inconsistency.
- **Figure alt text rounds $k_0$ to 1.00.** The simulation figure alt text reads "$k_0=1.00$" (produced by Python's `:.2f` format on 0.9952), while the Model Setup table explicitly states $k_0\approx0.9952$. The rounded value is not wrong, but it creates a visible discrepancy within the README.

---

## Summary

The tutorial is accurate and internally consistent. All five references were located and verified: publication years, publishers, journal volumes, issue numbers, and page ranges are all correct. The main message is fully supported by the equations and results presented. The most significant notation issue is that $A$, $\alpha$, and $\beta$ appear in the Overview before their formal definitions in the Equations section; this is a cosmetic concern (0 MAJOR, 3 MINOR, 0 NOT FOUND, 0 OVERREACH). The single most important fix is to either introduce $A$, $\alpha$, $\beta$ informally in the Overview (e.g., "with productivity $A$ and capital share $\alpha$") or defer those symbols to the Equations section.
