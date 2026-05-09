# Proofread: industrial-organization/nash-in-nash/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T04:00:00Z._

## Paper / Source Verification

### Horn, H. and Wolinsky, A. (1988). "Bilateral Monopolies and Incentives for Merger." RAND Journal of Economics, 19(3).

- **Located:** https://ideas.repec.org/a/rje/randje/v19y1988iautumnp408-419.html
- **Tutorial claims:** Listed as a reference with no attributed result or claim.
- **Source says:** The paper combines a bargaining model with duopoly analysis to study how merger incentives differ when input prices are negotiated bilaterally rather than set exogenously. RAND Journal of Economics, Volume 19, Issue 3, Autumn 1988, pages 408-419.
- **Verdict:** OK
- **Note:** All bibliographic fields confirmed: title, authors, year, journal, volume, and issue.

### Crawford, G. and Yurukoglu, A. (2012). "The Welfare Effects of Bundling in Multichannel Television Markets." American Economic Review, 102(2).

- **Located:** https://www.aeaweb.org/articles?id=10.1257/aer.102.2.643
- **Tutorial claims:** Listed as a reference with no attributed result or claim.
- **Source says:** The paper develops a structural model of multichannel television markets integrating viewership, consumer demand, and input-market bargaining, and estimates the welfare effects of unbundled channel pricing. American Economic Review, Volume 102, Issue 2, April 2012, pages 643-685.
- **Verdict:** OK
- **Note:** All bibliographic fields confirmed: title, authors, year, journal, volume, and issue.

### Ho, K. and Lee, R. (2017). "Insurer Competition in Health Care Markets." Econometrica, 85(2).

- **Located:** https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA13570
- **Tutorial claims:** Listed as a reference with no attributed result or claim.
- **Source says:** The paper develops and estimates a structural Nash-in-Nash bargaining model of employer-insurer-hospital negotiations to study the effects of insurer competition on premiums and hospital reimbursement rates. Econometrica, Volume 85, Issue 2, 2017, pages 379-417.
- **Verdict:** OK
- **Note:** All bibliographic fields confirmed: title, authors, year, journal, volume, and issue.

## Main Message Audit

> Nash-in-Nash turns each contract into a counterfactual network problem. The key object is what the insurer loses if a specific agreement fails. Hospital quality, substitution across insurers, and ownership determine that outside option.

(From the Overview: "Health insurers sell hospital networks. A plan loses value when it drops a high-quality hospital. It loses more when it drops the whole system. The object is a per-enrollee hospital transfer. The transfer depends on the enrollment an insurer would lose if a contract failed.")

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| A plan loses value when it drops a high-quality hospital | Equations: $Q(G_d) = \max_{h \in G_d} a_h + \eta(\|G_d\|-1)$; Results: "Dropping Hospital 1 hurts more because it has higher network value" | OK |
| It loses more when it drops the whole system | Results: Ownership Counterfactual table shows demand without system (10.4 and 8.6) is far below single-hospital disagreement demand | OK |
| Transfer depends on enrollment lost if contract failed | Equations: $\Delta_{hd} = m_d[q_d(G) - q_d(G^{-hd})]$ feeds directly into $w_{hd}$ formula | OK |
| Nash-in-Nash turns each contract into a counterfactual network problem | Solution Method: algorithm enumerates disagreement networks $G^{-hd}$ for each pair | OK |
| Hospital quality determines the outside option | Equations: $a_h$ in $Q(G_d)$; Model Setup table; Bilateral table shows H1 demand loss exceeds H2 demand loss | OK |
| Substitution across insurers determines the outside option | Equations: logit demand with $\sigma_\varepsilon$ controls cross-insurer substitution; Model Setup table | OK |
| Ownership determines the outside option | Results: merger section replaces $G^{-hd}$ with $G^{-Hd}$, producing higher system transfer | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $d$ | Equations, first sentence | Yes | Insurer index |
| $h$ | Equations, first sentence | Yes | Hospital index |
| $G_d$ | Equations, second sentence | Yes | Insurer d's hospital network |
| $G$ | Equations, third sentence | Yes | Full agreement network |
| $M$ | Equations, demand equation | Partial | Defined in Model Setup table within 50 lines |
| $v_d(G_d)$ | Equations, demand equation | Yes | Defined as "deterministic utility of an insurer" in next sentence |
| $\sigma_\varepsilon$ | Equations, demand equation | Partial | Defined in Model Setup table within 50 lines |
| $D$ | Equations, demand equation ($\sum_{\ell=1}^{D}$) | No | Never defined; context implies total number of insurers |
| $\ell$ | Equations, demand equation denominator | No | Standard summation dummy; no definition needed |
| $Q(G_d)$ | Equations, utility equation | Yes | Defined as "network value" with formula immediately following |
| $P_d$ | Equations, utility equation | Yes | Defined as "the premium" |
| $a_h$ | Equations, network value equation | Yes | Defined in next sentence as "hospital quality" |
| $\eta$ | Equations, network value equation | Yes | Defined in next sentence as "value of a second in-network hospital" |
| $m_d$ | Equations, margin definition | Yes | Defined at first appearance: $m_d = P_d - c_d^D$, "insurer margin" |
| $c_d^D$ | Equations, margin definition | Partial | Introduced in $m_d$ formula; labeled in Model Setup table |
| $G^{-hd}$ | Equations, link failure sentence | Yes | Defined as "disagreement network" at first appearance |
| $\Delta_{hd}$ | Equations, incremental value | Yes | Defined as "gross incremental value of hospital $h$ to insurer $d$" |
| $c_h^H$ | Equations, surplus formula | Partial | Introduced as "hospital cost $c_h^H$" in surplus sentence; labeled in Model Setup table |
| $S_{hd}$ | Equations, surplus formula | Yes | Defined as "bilateral surplus" at first appearance |
| $w_{hd}$ | Equations, Nash bargain | Yes | Defined as "per-enrollee hospital transfer" |
| $\tau$ | Equations, Nash bargain | Partial | Defined in Model Setup table within 50 lines |
| $H$ | Equations, merger section | Yes | Defined as "merged hospital system $H$" |
| $G^{-Hd}$ | Equations, merger section | Yes | Defined contextually: "relevant disagreement removes all system hospitals from insurer $d$" |
| $C_H$ | Equations, merger section | Yes | Defined at first appearance as $C_H = \sum_h c_h^H$ |
| $W_{Hd}$ | Equations, merger section | Yes | Defined as "system-level per-enrollee transfer" following the formula |

Flagged issues:
- $D$: used as the upper summation limit in the demand equation ($\sum_{\ell=1}^{D}$) but never defined. The correct value (2) is implied by the Model Setup table, but no sentence states "where $D$ is the number of insurers."

## Summary

All three cited references resolve to correct sources with matching bibliographic fields (title, authors, year, journal, volume, issue). The code in run.py implements every equation exactly as written in the README, and all numerical values in the diagnostic tables are consistent with the stated parameters. The main message is fully supported by the equations, solution method, and results shown. The single flagged issue is that $D$ - used as the upper summation limit in the logit demand equation - is never defined in the README prose or tables, leaving one symbol without an explicit definition. Count: 0 MAJOR, 0 MINOR reference issues, 0 OVERREACH/UNSUPPORTED main-message clauses, 1 undefined symbol ($D$ in $\sum_{\ell=1}^{D}$).
