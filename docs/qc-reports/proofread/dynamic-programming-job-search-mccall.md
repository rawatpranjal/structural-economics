# Proofread: dynamic-programming/job-search-mccall/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T18:05:00Z._

## Paper / Source Verification

### McCall, J.J. (1970). "Economics of Information and Job Search." *Quarterly Journal of Economics*, 84(1), 113-126.

- **Located:** https://econpapers.repec.org/RePEc:oup:qjecon:v:84:y:1970:i:1:p:113-126.
- **Tutorial claims:** Foundational paper on sequential job search; implicitly cited as the source of the model solved in the tutorial.
- **Source says:** Correct title, author, year, journal, volume 84, issue 1, pages 113-126.
- **Verdict:** OK
- **Note:** All bibliographic fields match the EconPapers/OUP record exactly.

---

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 6.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** Standard graduate textbook treatment; Chapter 6 covers the McCall search model.
- **Source says:** 4th edition, 2018, MIT Press. Chapter 6 is titled "Search and Unemployment," confirming job-search content.
- **Verdict:** OK
- **Note:** Edition, year, publisher, and chapter topic all confirmed correct.

---

### Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.

- **Located:** https://www.hup.harvard.edu/books/9780674750968
- **Tutorial claims:** Foundational reference for recursive methods; cited as background for VFI theory.
- **Source says:** Correct title, authors (Nancy L. Stokey, Robert E. Lucas Jr., Edward C. Prescott), year 1989, Harvard University Press.
- **Verdict:** OK
- **Note:** All details confirmed via Harvard University Press catalog.

---

### Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.

- **Located:** https://mitpress.mit.edu/9780262533980/equilibrium-unemployment-theory/
- **Tutorial claims:** Cited as background for equilibrium unemployment theory building on the McCall framework.
- **Source says:** Correct title, author (Christopher A. Pissarides), year 2000, MIT Press, 2nd edition.
- **Verdict:** OK
- **Note:** All details confirmed via MIT Press catalog.

---

## Main Message Audit

> "McCall search turns unemployment duration into a reservation wage. The worker accepts only offers that beat this price of waiting. A higher benefit or more patience raises the cutoff and extends unemployment duration. Computationally, the Bellman problem is nearly scalar because rejection has one continuation value. The scalar fixed point gives a clear check on finite-grid VFI."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| McCall search turns unemployment duration into a reservation wage | Overview; Equations: cutoff policy derived from Bellman | OK |
| The worker accepts only offers that beat this price of waiting | Equations: indifference condition $A(w^{\ast})=C$ | OK |
| A higher benefit raises the cutoff | Results: figure `wstar-vs-benefits.png`; parameter table rows with increasing $b$ | OK |
| More patience raises the cutoff | Results: figure `wstar-vs-beta.png`; parameter table rows with increasing $\beta$ | OK |
| Higher benefit/patience extends unemployment duration | Results: parameter table shows E[duration] increasing with $b$ and $\beta$ | OK |
| Bellman problem is nearly scalar because rejection has one continuation value | Equations: $C = b + \beta \mathbb{E}_F[V(W')]$ is a single scalar; Solution Method pseudocode | OK |
| Scalar fixed point gives a clear check on finite-grid VFI | Solution Method: continuous benchmark solved via Brent; table compares grid vs. continuous $w^{\ast}$ | OK |

Issues:
- None. All clauses are directly demonstrated by equations, figures, or the parameter table in the README.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $b$ | Overview ("benefit $b$") | Yes — informal introduction in Overview, context clear | Used consistently throughout |
| $W$ | Equations ("Let $W$ be a wage offer") | Yes | Explicitly defined |
| $F$ | Equations ("with distribution $F$") | Yes | Explicitly defined alongside $W$ |
| $w$ | Equations ("The current offer is $w$") | Yes | Explicitly defined |
| $\beta$ | Equations ("The worker discounts at $\beta\in(0,1)$") | Yes | Explicitly defined |
| $A(w)$ | Equations (displayed equation) | Yes — named "Accepting gives a permanent income stream" | |
| $C$ | Equations ("rejection has one value: $C = \ldots$") | Yes | Explicitly named |
| $V(w)$ | Equations (Bellman equation) | Yes — introduced in Bellman equation | |
| $w^{\ast}$ | Overview ("reservation wage $w^{\ast}$") | Yes | Introduced in Overview |
| $W'$ | Equations ($\mathbb{E}_F[V(W')]$) | No explicit definition | Used as next-period wage draw; meaning is implied by context but never stated |
| $T$ | Solution Method ("The Bellman operator $T$") | Yes | Explicitly named |
| $r$ | Solution Method (scalar fixed-point equation) | No — introduced without statement | Plays the role of $w^{\ast}$ in the continuous case; not explicitly equated or linked to $w^{\ast}$ in prose |
| $m(r)$ | Solution Method (scalar fixed-point equation) | Yes — "$m(r)=\mathbb{E}_F[\max(W,r)]$" | Defined inline |
| $\mu$ | Model Setup table ("$e^\mu$ for the lognormal") | No — appears in the Role column of the table only | Value 0.0 is shown in the wage law notation but $\mu$ is never introduced as a named parameter in the prose |
| $\sigma$ | Model Setup table ("$\log W\sim N(0.0,1.0^2)$") | Implicit only | $\sigma=1.0$ can be read from the table but $\sigma$ is never named in the README text |
| $n_w$ | Solution Method ("$n_w=50$ equal-probability bins") | Yes — value given inline | |

Flagged issues:
- **$W'$ (Equations):** Used in $C = b + \beta\,\mathbb{E}_F[V(W')]$ without explicit definition. A one-word gloss ("where $W'$ is an i.i.d. draw from $F$") would close the gap. Low severity — meaning is unambiguous from context.
- **$r$ (Solution Method):** Introduced in the scalar fixed-point equation without connecting it to $w^{\ast}$. The reader must infer that $r$ is the continuous analog of the grid reservation wage. A phrase like "where $r$ denotes the reservation wage" would complete the link.
- **$\mu$ (Model Setup table):** Appears in the Role column ("$e^\mu$ for the lognormal") but is never named as the lognormal location parameter in the README prose. The table shows the numeric value 0.0 inside the wage-law expression, which is $\mu$, but the symbol $\mu$ itself goes undefined.

---

## Summary

All four cited references are verified correct — titles, authors, years, publishers, and edition numbers match authoritative sources, and Ljungqvist-Sargent Chapter 6 is confirmed as "Search and Unemployment." The main message is fully supported: every clause in the takeaway is demonstrated by the equations, figures, or parameter table. The only issues are three notation gaps of low-to-moderate severity: $W'$ is used in the Bellman equation without a one-line definition, $r$ is introduced in the continuous fixed-point equation without being explicitly linked to $w^{\ast}$, and $\mu$ appears in the Model Setup table role column without ever being named in the prose. The single most important fix is defining $r$ in the Solution Method section, since readers who skip back from the pseudocode to the scalar fixed-point equation may not immediately recognize it as the reservation wage.
