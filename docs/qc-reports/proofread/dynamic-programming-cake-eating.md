# Proofread: dynamic-programming/cake-eating/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T17:10:00Z._

## Paper / Source Verification

### Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 4.

- **Located:** https://www.hup.harvard.edu/books/9780674750968
- **Tutorial claims:** The book is cited as the canonical reference for the cake-eating / finite-resource allocation problem solved via value function iteration.
- **Source says:** Chapter 4 is titled "Applications of Dynamic Programming under Certainty" and explicitly covers the cake-eating / finite-resource problem as one of its canonical examples; Chapter 3 ("Dynamic Programming under Certainty") provides the Bellman equation theory. Author order Stokey–Lucas–Prescott is confirmed correct.
- **Verdict:** OK
- **Note:** Year (1989), publisher (Harvard University Press), author order, and chapter are all verified correct.

---

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** The book is cited as a secondary reference for the recursive/Bellman equation framework underlying the cake-eating model.
- **Source says:** Chapter 3 is titled "Dynamic Programming" and covers the Bellman equation and its mathematical properties (existence, uniqueness, contraction mappings). The 4th edition was published in 2018 by MIT Press (ISBN 9780262038669). Author order Ljungqvist–Sargent is confirmed.
- **Verdict:** OK
- **Note:** Year (2018), edition (4th), publisher (MIT Press), and author order are all verified correct. Ch. 3 is the appropriate reference for Bellman equation foundations.

---

## Main Message Audit

> "Cake eating isolates Bellman logic in a one-state resource problem. The computed policy should consume a constant share of remaining cake. In this log case, the share is $1-\beta$. The closed form makes value function iteration easy to inspect. The remaining errors are interpolation and choice-grid error."

and from the Overview:

> "Value function iteration solves the Bellman equation on a grid. Log utility gives a closed-form Euler rule. That rule lets us check the computed value and policy directly."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Cake eating is a one-state problem | Equations (single state $W$, Bellman $V(W)=\max\{u(c)+\beta V(W-c)\}$) | OK |
| The computed policy consumes a constant share | Equations ($c^{\ast}(W)=(1-\beta)W$), Results (table, figures) | OK |
| In the log case, the share is $1-\beta$ | Equations, Model Setup ($\sigma=1$, $\beta=0.9$), Results ("10% of remaining cake") | OK |
| The closed form makes VFI easy to inspect | Results (pointwise table, figures showing numerical vs. analytical) | OK |
| Remaining errors are interpolation and choice-grid error | Solution Method (describes both interpolation and discrete $N_c$-point grid); code comment in run.py confirms this attribution | OK |
| "Log utility gives a closed-form Euler rule" | Equations derive the closed form via a guess-and-verify step, not from the Euler equation alone | OVERREACH |

Issues:
- **OVERREACH — "Log utility gives a closed-form Euler rule":** The Euler equation $u'(c_t)=\beta u'(c_{t+1})$ alone only pins down the *ratio* $c_{t+1}/c_t = \beta$; it does not determine the consumption *level*. The full closed-form policy $c^{\ast}(W) = (1-\beta)W$ requires the additional guess-and-verify argument described later in the Equations section. The phrase conflates the Euler equation with the complete closed-form derivation. The Equations section correctly describes the derivation; the Overview description is loose but not seriously misleading.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $W_t$ | Overview ("the state is remaining cake $W_t$") | Yes — Equations: "Let $W_t$ be remaining cake at the start of period $t$" | |
| $c_t$ | Overview ("the control is consumption $c_t$") | Yes — Equations: "The household chooses $c_t \in [0, W_t]$" | |
| $W_0$ | Equations ("$W_0$ given") | Implicitly — never receives a dedicated definition sentence; context makes it clear | Minor: could state "where $W_0$ is the initial endowment" |
| $W_{t+1}$ | Equations (resource constraint) | Yes — defined by $W_{t+1} = W_t - c_t$ | |
| $\beta$ | Equations ("discount factor $\beta \in (0,1)$") | Yes — also tabulated in Model Setup | |
| $u(c)$ | Equations | Yes — CRRA formula given immediately | |
| $\sigma$ | Equations (inside CRRA formula) | Yes — called "CRRA curvature" in Model Setup table | "CRRA" is used without expansion; acceptable standard jargon |
| $V(W)$ | Equations ("The value function solves…") | Yes | |
| $c^{\ast}(W)$ | Equations ("closed-form policy") | Yes | |
| $g(W)$ | Equations | Yes — $g(W)=W-c^{\ast}(W)=\beta W$ | Used only in Equations; not referenced again |
| $V'(W)$ | Equations | Yes — stated after $V(W)$ formula | Prime used for both $u'$ and $V'$; standard convention, not ambiguous |
| $\varepsilon$ | Model Setup table | Yes | |
| $N_W$ | Model Setup table | Yes | |
| $N_c$ | Model Setup table | Yes | |
| $T_{sim}$ | Model Setup table | Yes | |
| $T$ (Bellman operator) | Solution Method ("Define the Bellman operator $(TV)(W)$") | Yes — defined at first and only use | |
| $W$ (function argument) | Equations (Bellman) | Yes — consistent with $W_t$ as state; stationary notation standard in DP | Dual use: $W$ is both the function argument in $V(W)$ and the grid range $[0.01,1.0]$ in Model Setup; not confusing in context |

Flagged issues:
- **$W_0$ undefined as a symbol:** Used in Equations ("$W_0$ given") and Results ("Starting from $W_0=1$") without a dedicated definition sentence. The value $w_max = 1.0$ appears in Model Setup only as the grid upper bound, not as $W_0$. Low-severity: the meaning is unambiguous from context, but a single phrase like "initial endowment $W_0 = 1$" in Model Setup would close the gap.
- **Trailing comma after the $c^{\ast}(W)$, $g(W)$ display equation:** The equation ends with `\beta\, W,` (comma before `$$`) suggesting the sentence continues, but the next sentence is a new paragraph. Minor typographic inconsistency.
- **$g(W)$ is defined but not referenced after Equations:** The next-period wealth function $g(W) = \beta W$ is introduced but the Results and Solution Method refer to "next-period wealth" $W-c$ rather than $g$. Not an error, but the symbol is orphaned after its definition.

---

## Summary

Both references are correctly cited with accurate authors, years, publishers, and chapters. Equations are mathematically correct and internally consistent: the closed-form value function $V(W)$, derivative $V'(W)$, and consumption rule $c^{\ast}(W)=(1-\beta)W$ are all verified. The main message is well-supported by the Equations, Solution Method, and Results sections. The single OVERREACH ("Log utility gives a closed-form Euler rule") is a loose phrase in the Overview that attributes the full closed-form policy to the Euler equation alone, when the Equations section correctly shows it requires an additional guess-and-verify step; this is the most important fix. Notation is largely complete and consistent, with one minor gap ($W_0$ never formally introduced as a symbol) and one orphaned symbol ($g(W)$). Overall: **1 OVERREACH, 0 MAJOR, 0 MINOR reference issues, 0 NOT FOUND, 2 minor notation gaps**.
