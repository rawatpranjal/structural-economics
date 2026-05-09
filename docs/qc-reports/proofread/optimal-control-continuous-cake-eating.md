# Proofread: optimal-control/continuous-cake-eating/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T22:10:00Z._

## Paper / Source Verification

### Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 7.

- **Located:** https://press.princeton.edu/books/hardcover/9780691132921/introduction-to-modern-economic-growth
- **Tutorial claims:** Cited as the primary reference for the Pontryagin maximum-principle treatment of continuous-time optimal control.
- **Source says:** Published 2009 by Princeton University Press, 990 pp. Chapter 7 covers "An Introduction to the Theory of Optimal Control," including Pontryagin's maximum principle and costate equations.
- **Verdict:** OK
- **Note:** Year 2009, publisher Princeton UP, and chapter scope all confirmed.

### Kamien, M. and Schwartz, N. (2012). *Dynamic Optimization*. Dover, 2nd edition.

- **Located:** https://store.doverpublications.com/products/9780486488561
- **Tutorial claims:** Cited as a reference on dynamic optimization methods.
- **Source says:** Dover published the second edition in November 2012 (ISBN 9780486488561), reprinting the 1991 North-Holland second edition.
- **Verdict:** OK
- **Note:** The Dover 2012 book is the 2nd edition; citation is accurate.

### Chiang, A. (1992). *Elements of Dynamic Optimization*. Waveland Press.

- **Located:** McGraw-Hill 1992 edition - https://www.abebooks.com/9780070109117; Waveland Press reprint - https://www.amazon.com/Elements-Dynamic-Optimization-Alpha-Chiang/dp/157766096X
- **Tutorial claims:** Cited as a general reference on dynamic optimization.
- **Source says:** The 1992 edition was published by McGraw-Hill (ISBN 9780070109117). Waveland Press issued a reprint in 1999-2000 (ISBN 9781577660965).
- **Verdict:** MINOR
- **Note:** Year and publisher do not correspond to the same edition: cite either "McGraw-Hill, 1992" or "Waveland Press, 1999".

---

## Main Message Audit

> "The costate is the intertemporal price of the remaining resource. Here the present-value price is constant. Optimal consumption declines at rate $\rho/\sigma$ and keeps discounted marginal utility equal across dates. Higher impatience raises the depletion rate. Higher risk aversion slows it through the smoothing motive."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Costate is the intertemporal price of the remaining resource | Solution Method prose and shadow-price figure | OK |
| Present-value price is constant | Equations: $\dot{\lambda}=-\partial\mathcal{H}/\partial W=0$; Results prose | OK |
| Optimal consumption declines at rate $\rho/\sigma$ | Equations: $\dot{c}/c=-\rho/\sigma$; closed-form $c(t)=(\rho/\sigma)W_0 e^{-\rho t/\sigma}$ | OK |
| Keeps discounted marginal utility equal across dates | Equations: FOC $e^{-\rho t}c^{-\sigma}=\lambda$ (constant) | OK |
| Higher impatience raises the depletion rate | Equations: $c(0)=(\rho/\sigma)W_0$ and rate $\rho/\sigma$ both increase with $\rho$ | OK |
| Higher risk aversion slows depletion through smoothing motive | Equations: $c(0)=(\rho/\sigma)W_0$ and rate $\rho/\sigma$ both decrease with $\sigma$ | OK |

Issues:
- None.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $W_0$ | Overview | Yes | "fixed stock $W_0$ of a consumption good" |
| $c(t)$ | Overview | Yes | "continuous-time consumption path $c(t)$" |
| $W(t)$ | Overview | Yes | "remaining stock $W(t)$" |
| $\sigma$ | Equations | Partial | Mathematical constraint ($\sigma>0$, $\sigma\neq1$) stated at first use; economic label "Relative risk aversion" given in Model Setup table ~30 lines later |
| $u(c)$, $u'(c)$ | Equations | Yes | Defined inline: $u(c)=\frac{c^{1-\sigma}}{1-\sigma}$, $u'(c)=c^{-\sigma}$ |
| $\rho$ | Equations | Partial | Used in objective $e^{-\rho t}$ without prose label; defined "Continuous discount rate" in Model Setup table ~28 lines later |
| $\mathcal{H}$ | Equations | Yes | "The present-value Hamiltonian is $\mathcal{H}$…" |
| $\lambda(t)$ | Equations (Hamiltonian argument) | Partial | Appears as argument of $\mathcal{H}$ and in FOC before being labeled; named "present-value costate" in prose 6 lines later |
| $\mu(t)$ | Equations | Yes | "The current-value shadow price is $\mu(t)=e^{\rho t}\lambda=c(t)^{-\sigma}$" |
| $T$ | Model Setup | Yes | "Plotting horizon" in parameter table |

Flagged issues:
- None. The partial definitions of $\sigma$, $\rho$, and $\lambda$ are all resolved within 30 lines of first use and the audience would recognise these as standard symbols; no flag warranted.

---

## Summary

The tutorial is mathematically sound and internally consistent. All equations in the Equations section are correct, the closed-form solution is consistent with the code in `run.py`, and all main-message clauses are supported by the shown derivations. The single issue is a MINOR bibliographic mismatch in the Chiang citation: the year 1992 belongs to the McGraw-Hill first edition, while Waveland Press issued its reprint in 1999-2000; the citation should use one consistent year-publisher pair. No MAJOR issues, no unsupported claims, and no undefined notation.
