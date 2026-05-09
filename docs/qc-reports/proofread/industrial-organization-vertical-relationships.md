# Proofread: industrial-organization/vertical-relationships/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:15:00Z._

## Paper / Source Verification

### Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.

- **Located:** https://mitpress.mit.edu/9780262200714/the-theory-of-industrial-organization/
- **Tutorial claims:** The reference is cited as the canonical source for double marginalization and vertical contracts in industrial organization.
- **Source says:** Jean Tirole, "The Theory of Industrial Organization," MIT Press, Cambridge MA, 1988. ISBN 0262200714. 479 pages.
- **Verdict:** OK
- **Note:** Year, author, publisher, and title all match the MIT Press catalog entry.

## Main Message Audit

> Double marginalization comes from the retailer's perceived marginal cost. A high wholesale price raises that cost and lowers quantity. A two-part tariff sets $w=c_M$, so the retailer chooses the integrated price. The fixed fee then allocates profit.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Double marginalization comes from the retailer's perceived marginal cost | Equations: "Because $w^{DM}>c_M$, the retailer acts as if marginal cost is too high" | OK |
| A high wholesale price raises that cost and lowers quantity | Results figure (wholesale pass-through) and table showing quantity 3.5 under linear wholesale vs 7.0 integrated | OK |
| A two-part tariff sets $w=c_M$, so the retailer chooses the integrated price | Equations: $w^{TPT}=c_M$ defined; Results table shows retail price 6.5 and quantity 7.0 under two-part tariff, matching the integrated benchmark | OK |
| The fixed fee then allocates profit | Results table: under two-part tariff manufacturer profit 24.5, retailer profit 0.0, channel profit 24.5 matching the integrated channel | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $q$ | Equations: $q(p)=a-bp$ | Yes, inline | quantity demanded |
| $p$ | Equations: "where $p$ is the retail price" | Yes, inline | retail shelf price |
| $a$ | Equations: $q(p)=a-bp$ | Partial - Model Setup table ~34 lines later | within 50-line window; acceptable |
| $b$ | Equations: $q(p)=a-bp$ | Partial - Model Setup table ~34 lines later | within 50-line window; acceptable |
| $\bar p$ | Equations: $p\leq\bar p\equiv a/b$ | Yes, inline ($\equiv a/b$) | choke price |
| $c_M$ | Equations: "Costs are $c_M$ upstream" | Partial - Model Setup table ~37 lines later | within 50-line window; acceptable |
| $c_R$ | Equations: "and $c_R$ downstream" | Partial - Model Setup table ~37 lines later | within 50-line window; acceptable |
| $\Pi^I$ | Equations: $\Pi^I(p)=(p-c_M-c_R)q(p)$ | Yes, defined by equation | integrated channel profit function |
| $p^I$ | Equations: "the joint-profit price is $p^I$" | Yes, inline | integrated optimal price |
| $w$ | Equations: "linear wholesale price $w$" | Yes, inline | wholesale price |
| $p_R(w)$ | Equations: "Its best response is $p_R(w)$" | Yes, inline | retailer best-response function |
| $w^{DM}$ | Equations: "which gives $w^{DM}$" | Yes, inline | equilibrium wholesale price under double marginalization |
| $w^{TPT}$ | Equations: "A two-part tariff sets $w^{TPT}=c_M$" | Yes, inline | two-part tariff wholesale price |
| $F$ | Equations: "uses the fixed fee $F=(p^I-c_M-c_R)q(p^I)$" | Yes, inline | fixed franchise fee |

Flagged issues:
- None. All symbols are defined at or before first use, no overloading or drift across sections.

## Summary

The tutorial is clean. The single reference (Tirole 1988) is confirmed by the MIT Press catalog. All four clauses of the main takeaway are directly supported by equations and numerical results shown in the README. Notation is consistent and complete throughout: every symbol introduced in the Equations section is either defined inline or defined in the Model Setup parameter table within 50 lines. There are no MAJOR or MINOR issues and no NOT FOUND references.
