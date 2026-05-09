# Proofread: industrial-organization/dynamic-entry-exit/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T08:00:00Z._

## Paper / Source Verification

### Ericson, R. and Pakes, A. (1995). Markov-perfect industry dynamics: A framework for empirical work. *Review of Economic Studies*, 62(1):53-82.

- **Located:** https://academic.oup.com/restud/article-abstract/62/1/53/1568000
- **Tutorial claims:** The paper is cited as the framework behind Markov-perfect industry dynamics with free entry and exit; the Takeaway names "Ericson-Pakes style IO models" as the context for sunk-cost-driven firm-count persistence.
- **Source says:** Title, authors, journal, volume, issue, and page range all match the authoritative Oxford Academic record (DOI: 10.2307/2297841).
- **Verdict:** OK
- **Note:** Published title capitalizes both words "Markov-Perfect"; the tutorial uses "Markov-perfect" (lowercase second word). Citation style is consistent with the rest of the repo - not a substantive error.

### Hopenhayn, H. (1992). Entry, exit, and firm dynamics in long run equilibrium. *Econometrica*, 60(5):1127-1150.

- **Located:** https://econpapers.repec.org/article/ecmemetrp/v_3a60_3ay_3a1992_3ai_3a5_3ap_3a1127-50.htm
- **Tutorial claims:** The paper is cited alongside Ericson-Pakes as a foundational reference for the entry/exit framework.
- **Source says:** Title, author initial (Hugo A. Hopenhayn), journal, volume, issue, and page range all match the EconPapers/IDEAS record (DOI: 10.2307/2951541).
- **Verdict:** OK
- **Note:** No discrepancies found.

## Main Message Audit

> "The entry and exit conditions separate. Static profits show whether a firm covers the operating cost. Dynamic values show whether keeping the incumbency option is worthwhile. A sunk entry cost creates a band where incumbents stay and entrants wait. That band makes firm counts persistent in Ericson-Pakes style IO models."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Entry and exit conditions separate | Equations: distinct conditions $\Delta(N)\geq 0$ for exit and $V(\bar S+m)\geq K$ for entry; Results: separate policy columns in table | OK |
| Static profits show whether a firm covers the operating cost | Results table: "Net profit pi-f" column and "Zero-profit N (static)" row in Equilibrium Statistics | OK |
| Dynamic values show whether keeping the incumbency option is worthwhile | Equations: $V(N)$ includes $\beta\mathbb{E}[V(N_{t+1})]$; Results table: V(N) remains positive (e.g., 2.48) at N=15 where net static profit is -0.25 | OK |
| A sunk entry cost creates a band where incumbents stay and entrants wait | Results: entry drops to zero at N=10 while exit probability is only 0.022; Results figure description calls this "the hysteresis region created by sunk entry" | OK |
| That band makes firm counts persistent | Results: stationary std. dev. of N is 0.15; stationary distribution figure shows tight concentration around mode 8 | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $N_t$ | Overview ("the active firm count $N_t$") | Yes - Equations ("Let $N_t\in\{1,\ldots,N_{\max}\}$ denote...") | OK |
| $f$ | Overview ("fixed cost $f$") | Partial - Overview gives context; Model Setup table gives value | OK |
| $K$ | Overview ("sunk cost $K$") | Model Setup table | OK |
| $P$ | Equations ("inverse demand $P=a-bQ$") | Yes - same sentence, as price | **Collision:** $P$ reused for transition matrix in Solution Method ("$\mu=\mu P$") |
| $a$ | Equations ("$P=a-bQ$") | Model Setup table (~52 lines after first use) | Partial - acceptable for intended audience |
| $b$ | Equations ("$P=a-bQ$") | Model Setup table (~52 lines after first use) | Partial - acceptable |
| $Q$ | Equations ("$P=a-bQ$") | Not explicitly defined | Standard total-quantity notation in Cournot; acceptable for intended audience |
| $c$ | Equations ("constant marginal cost $c$") | Model Setup table | Partial - acceptable |
| $\pi(N)$ | Equations | Yes - formula given immediately | OK |
| $\Delta(N)$ | Equations | Yes - "deterministic surplus" with formula | OK |
| $\beta$ | Equations (in $\Delta(N)$ formula) | Model Setup table | Partial - acceptable |
| $\mathbb{E}$ | Equations (in $\Delta(N)$) | Standard expectation operator | OK |
| $\sigma_\varepsilon$ | Equations | Yes - "logistic scale $\sigma_\varepsilon$" in same paragraph | OK |
| $V(N)$ | Equations | Yes - "log-sum inclusive value" with formula | OK |
| $p_{\mathrm{exit}}(N)$ | Equations | Yes - formula given | OK |
| $\bar S(N_t)$ | Equations | Yes - "expected survivor count $\bar S(N_t)=\mathrm{round}\{...\}$" in same sentence | OK |
| $m$ | Equations (entry condition) | Yes - "Entrant $m$ enters only if..." in same paragraph | OK |
| $e(N_t)$ | Equations | Yes - formula given as set-builder max | OK |
| $S_t$ | Equations (transition law) | Yes - transition law equation | OK |
| $N_{\max}$ | Equations | Model Setup table | Partial - acceptable; used with contextual meaning throughout |
| $\mu$ | Solution Method pseudocode | Yes - "The invariant distribution solves $\mu=\mu P$" in same paragraph | OK |
| $P$ (transition matrix) | Solution Method ("transition matrix P" in pseudocode, "$\mu=\mu P$" in text) | Pseudocode label | Collision with $P$ = price in Equations |

Flagged issues:
- $P$ is used for two distinct objects. In Equations it is the inverse demand (price): "$P=a-bQ$". In Solution Method it is the Markov transition matrix: "$\mu=\mu P$". The two sections are non-overlapping and a careful reader can infer the intended meaning, but the symbol is formally overloaded.

## Summary

Both citations are correct and verified against authoritative sources; no bibliographic errors were found. All clauses of the main message are directly supported by the tutorial's equations, model setup, and results. The only notation issue is a symbol collision: $P$ is introduced as the inverse demand price in the Equations section and then reused for the Markov transition matrix in the Solution Method section. The tutorial has 0 MAJOR issues, 0 MINOR issues on citations, 0 OVERREACH clauses, and 1 notation collision ($P$ overloaded). The single most important fix is to rename the transition matrix to a distinct symbol (such as $\mathbf{T}$ or $\Pi$) so that $P$ unambiguously refers to the demand price throughout.
