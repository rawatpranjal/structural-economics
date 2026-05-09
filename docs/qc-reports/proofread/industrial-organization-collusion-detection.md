# Proofread: industrial-organization/collusion-detection/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:45:00Z._

## Paper / Source Verification

### Stigler, G. (1964). A Theory of Oligopoly. *Journal of Political Economy*, 72(1), 44--61.

- **Located:** https://www.journals.uchicago.edu/doi/10.1086/258853
- **Tutorial claims:** cites as foundational reference for oligopoly and collusion detection.
- **Source says:** Volume 72, No. 1, February 1964, pp. 44-61. Correct on all bibliographic fields.
- **Verdict:** OK
- **Note:** Citation is accurate.

### Porter, R. (1983). A Study of Cartel Stability: The Joint Executive Committee, 1880--1886. *Bell Journal of Economics*, 14(2), 301--314.

- **Located:** https://econpapers.repec.org/article/rjebellje/v_3a14_3ay_3a1983_3ai_3aautumn_3ap_3a301-314.htm
- **Tutorial claims:** cites as an empirical study of cartel stability using historical price data.
- **Source says:** Volume 14, Issue 2 (Autumn 1983), pp. 301-314. Journal name "Bell Journal of Economics" is correct for the 1983 publication; the journal was renamed RAND Journal of Economics only in 1984.
- **Verdict:** OK
- **Note:** Citation is accurate.

### Harrington, J. (2008). Detecting Cartels. In *Handbook of Antitrust Economics*. MIT Press.

- **Located:** https://mitpress.mit.edu/9780262524773/handbook-of-antitrust-economics/
- **Tutorial claims:** cites as the methodological reference for cartel detection methods.
- **Source says:** Chapter 6, "Detecting Cartels," contributed by Joseph E. Harrington Jr. to the edited volume *Handbook of Antitrust Economics*, MIT Press, 2008. Editor is Paolo Buccirossi. MIT Press confirms the title as "Handbook of Antitrust Economics."
- **Verdict:** OK
- **Note:** Book title and year are correct; the editor is not cited but that is not required.

### Igami, M. and Sugaya, T. (2021). Measuring the Incentive to Collude: The Vitamin Cartels, 1990--1999. *Review of Economic Studies*, 89(3), 1460--1494.

- **Located:** https://academic.oup.com/restud/article-abstract/89/3/1460/6354371
- **Tutorial claims:** cites as a structural measurement of collusion incentives using the vitamin cartel episode 1990-1999.
- **Source says:** Review of Economic Studies, Volume 89, Issue 3, pp. 1460-1494. The IDEAS/RePEC record reports the year as 2022, matching the print volume. The tutorial cites 2021, which is one year early.
- **Verdict:** MINOR
- **Note:** The formal publication year for Volume 89(3) is 2022, not 2021; advance-access availability in 2021 does not change the volume year.

---

## Main Message Audit

> "A price break is a lead, not a cartel finding. The repeated-game check asks whether future cartel rents can deter one-period cheating. In the duopoly, $\delta^{\ast}=0.5294$. With ten symmetric firms, $\delta^{\ast}=0.7516$. At $\delta=0.9$, the exact firm-count cutoff is 33. Cost and demand evidence still matters."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| A price break is a lead, not a cartel finding | Results: "This is a clean benchmark, not a detection test" | OK |
| The repeated-game check asks whether future cartel rents can deter one-period cheating | Equations: incentive constraint $V^M \geq V^D$ | OK |
| In the duopoly, $\delta^{\ast}=0.5294$ | Results table: $n=2$, delta_star = 0.5294 | OK |
| With ten symmetric firms, $\delta^{\ast}=0.7516$ | Results table: $n=10$, delta_star = 0.7516 | OK |
| At $\delta=0.9$, the exact firm-count cutoff is 33 | Results table: $n=33$ sustained, $n=34$ not | OK |
| Cost and demand evidence still matters | Overview and Results acknowledge this is a stylized benchmark | OK |

Issues:
- None. All clauses are directly supported.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $i$ | Equations | Yes - "Firms $i=1,\ldots,n$" | OK |
| $n$ | Equations | Yes - firm count in "Firms $i=1,\ldots,n$" | OK |
| $q_i$ | Equations | Yes - "choose quantities $q_i$" | OK |
| $Q$ | Equations | Yes - "$Q=\sum_i q_i$" | OK |
| $P(Q)$ | Equations | Yes - "inverse demand is $P(Q)=a-Q$" | OK |
| $a$ | Equations | Yes - demand intercept in $P(Q)=a-Q$ | OK |
| $c$ | Equations | Yes - "constant marginal cost $c<a$" | OK |
| $\delta$ | Equations | Yes - "common discount factor" $\delta\in(0,1)$ | OK |
| $q^N$, $\pi^N$ | Equations table | Yes - defined in regime table | OK |
| $q^M$, $\pi^M$ | Equations table | Yes - defined in regime table | OK |
| $q^D$, $\pi^D$ | Equations table | Yes - defined in regime table | OK |
| $V^M$ | Equations | Yes - defined as $\pi^M/(1-\delta)$ immediately | OK |
| $V^D$ | Equations | Yes - defined as $\pi^D + \delta\pi^N/(1-\delta)$ immediately | OK |
| $\delta^{\ast}$ | Equations | Yes - defined in the incentive-constraint display equation | OK |
| $P_t$ | Equations (simulation) | Yes - left-hand side of $P_t=P^{r_t}+\eta_t$ | OK |
| $r_t$ | Equations (simulation) | Partial - described as $r_t\in\{N,M,N\}$ with prose clarification | See note |
| $P^N$, $P^M$ | Equations (simulation) | Yes - defined in the sentence immediately following the display equation | OK |
| $\eta_t$ | Equations (simulation) | Undefined - no distribution or parametrization given | Flagged |
| $\sigma$ | Model Setup table | Partial - value 1.5 given as "Price noise" but never connected to $\eta_t$ | Flagged |
| $m_t$ | Equations (simulation) | Yes - "The reported margin is $m_t=(P_t-c)/P_t$" | OK |

Flagged issues:

- **$\eta_t$ undefined distribution.** The noise term $\eta_t$ appears in the price equation $P_t = P^{r_t} + \eta_t$ but no distribution is stated. The Model Setup table introduces $\sigma=1.5$ as "Price noise" without writing $\eta_t \sim N(0,\sigma^2)$ or equivalent. The reader cannot reconstruct the simulation from the equations alone.

- **$r_t \in \{N,M,N\}$ uses set notation with a repeated element.** Mathematically, $\{N,M,N\} = \{N,M\}$, which loses the temporal ordering (N then M then N). The surrounding prose makes the intended meaning clear, but the notation itself is inconsistent: a set cannot represent a sequence with a repeated value.

---

## Summary

The tutorial is in good shape. There is 1 MINOR citation issue: the Igami and Sugaya paper's formal publication year is 2022 (Review of Economic Studies Volume 89), not 2021 as cited. There are 0 MAJOR issues and 0 NOT FOUND references. The main-message audit finds all takeaway clauses directly supported by the equations and results. Two notation gaps exist: $\eta_t$ is introduced in the price equation but its distribution and connection to $\sigma$ are never stated, and $r_t \in \{N,M,N\}$ applies set braces to an ordered sequence, technically collapsing the repeated N. The single most important fix is connecting $\eta_t$ and $\sigma$ in the Equations section so the simulation is fully specified from the equations alone.
