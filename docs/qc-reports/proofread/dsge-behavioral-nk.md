# Proofread: dsge/behavioral-nk/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T04:15:00Z._

## Paper / Source Verification

### Gabaix, X. (2020). A Behavioral New Keynesian Model. *American Economic Review*, 110(8), 2271-2327. https://doi.org/10.1257/aer.20162005.

- **Located:** https://www.aeaweb.org/articles?id=10.1257%2Faer.20162005
- **Tutorial claims:** The paper introduces cognitive discounting into a New Keynesian model, weakening the pull of future output and inflation on current conditions.
- **Source says:** Correct. The AEA page confirms author Xavier Gabaix, volume 110, issue 8, pages 2271-2327, year 2020, DOI 10.1257/aer.20162005.
- **Verdict:** OK
- **Note:** All bibliographic fields verified against the AEA journal page.

### Gabaix, X. (2016). A Behavioral New Keynesian Model. NBER Working Paper 22954. https://doi.org/10.3386/w22954.

- **Located:** https://www.nber.org/papers/w22954
- **Tutorial claims:** Listed as a companion working-paper version of the 2020 AER article.
- **Source says:** Correct. NBER page confirms working paper 22954, author Xavier Gabaix, issue date December 2016, DOI 10.3386/w22954.
- **Verdict:** OK
- **Note:** The working paper was revised in June 2019 before the 2020 journal publication; the December 2016 original date is the correct citation year.

### Gabaix, X. (2020). Replication Code for: A Behavioral New Keynesian Model. AEA Data and Code Repository. https://doi.org/10.3886/E117842V1.

- **Located:** https://www.openicpsr.org/openicpsr/project/117842/version/V1/view
- **Tutorial claims:** Replication code deposited in the AEA Data and Code Repository alongside the 2020 AER article.
- **Source says:** Correct. openICPSR entry E117842V1, author Xavier Gabaix, year 2020, DOI 10.3886/E117842V1, repository is the AEA Data and Code Repository.
- **Verdict:** OK
- **Note:** Entry title on the repository is "Replication Code for: 'A Behavioral New Keynesian Model'" (with quotation marks around the article title); the README omits the inner quotes, which is a cosmetic difference only.

### Gali, J. (2015). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press, 2nd edition.

- **Located:** https://press.princeton.edu/books/hardcover/9780691164786/monetary-policy-inflation-and-the-business-cycle
- **Tutorial claims:** Standard New Keynesian textbook reference used for the rational three-equation block.
- **Source says:** Princeton University Press page confirms author Jordi Gali, full title "Monetary Policy, Inflation, and the Business Cycle: An Introduction to the New Keynesian Framework and Its Applications," second edition, published June 2015, ISBN 9780691164786.
- **Verdict:** MINOR
- **Note:** The author's name is spelled "Gali" in the README but the published name is "Gali" with an acute accent on the i (Galí). The first edition year is 2008; the 2015 date is correct for the second edition.

### Woodford, M. (2003). *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.

- **Located:** https://press.princeton.edu/books/hardcover/9780691010496/interest-and-prices
- **Tutorial claims:** Foundational monetary theory reference for the New Keynesian framework.
- **Source says:** Princeton University Press page confirms author Michael Woodford, title "Interest and Prices: Foundations of a Theory of Monetary Policy," publisher Princeton University Press, copyright 2003, ISBN 9780691010496.
- **Verdict:** OK
- **Note:** All bibliographic fields match.

## Main Message Audit

> Cognitive discounting changes the transmission channel, not the static equations. When $M$ and $M_f$ fall below one, future output and inflation matter less for today's choices. Current monetary shocks have smaller cumulative effects, and distant forward-guidance shocks lose much of their current bite.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Cognitive discounting changes the transmission channel, not the static equations | Equations: $M$ and $M_f$ appear only in the expectational terms of the IS curve and Phillips curve; the Taylor rule is identical in both versions | OK |
| When $M$ and $M_f$ fall below one, future output and inflation matter less for today's choices | Equations: $M < 1$ reduces the weight on $\mathbb{E}_t x_{t+1}$; $M_f < 1$ reduces the weight on $\mathbb{E}_t \pi_{t+1}$ | OK |
| Current monetary shocks have smaller cumulative effects | Results table: Behavioral cumulative output = -2.292 vs Rational = -2.430; Behavioral cumulative inflation = -0.396 vs Rational = -0.481 | OK |
| Distant forward-guidance shocks lose much of their current bite | Results table: Behavioral FG output at H=8 is 0.024 vs Rational 0.171; forward-guidance figure shows monotone attenuation with horizon | OK |

Issues:
- None. All four clauses are directly supported by the README's equations or results.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $x_t$ | Equations | Yes | "Let $x_t$ be the output gap" |
| $\pi_t$ | Equations | Yes | "$\pi_t$ inflation" |
| $i_t$ | Equations | Yes | "$i_t$ the policy rate" |
| $r^n_t$ | Equations | Yes | "$r^n_t$ the natural real rate" |
| $v_t$ | Equations | Yes | "$v_t$ a policy-rate wedge" |
| $M$ | Equations (IS curve) | Partial | Named as part of "the pair $(M, M_f)$"; individual role as household attention follows from position in IS curve but is never stated |
| $M_f$ | Equations (Phillips curve) | Partial | Named as part of "the pair $(M, M_f)$"; individual role as firm attention follows from position in Phillips curve but is never stated |
| $u_t$ | Equations (Phillips curve) | No | Undefined; cost-push shock; never described in the README and silently dropped in the solution without explanation |
| $\sigma$ | Equations (IS curve) | Yes | Defined in Model Setup table within 5 lines: "Interest sensitivity in the IS curve" |
| $\beta$ | Equations (Phillips curve) | Yes | Defined in Model Setup table: "Quarterly discount factor" |
| $\kappa$ | Equations (Phillips curve) | Yes | Defined in Model Setup table: "Slope of the Phillips curve" |
| $\phi_\pi$ | Equations (Taylor rule) | Yes | Defined in Model Setup table: "Taylor-rule response to inflation" |
| $\phi_x$ | Equations (Taylor rule) | Yes | Defined in Model Setup table: "Taylor-rule response to the output gap" |
| $\rho_v$ | Equations (AR(1)) | Yes | Defined in Model Setup table: "Persistence of the current policy wedge" |
| $\varepsilon^v_t$ | Equations (AR(1)) | Partial | Model Setup table lists "Shock innovation \| 0.010" without assigning the symbol; the symbol itself is not introduced in text |
| $\varepsilon^v$ | Equations (forward guidance) | Partial | Used without subscript in the forward-guidance sentence, inconsistent with $\varepsilon^v_t$ in the AR(1) equation above |
| $H$ | Equations (forward guidance) | Yes | "one future quarter $H$"; terminal condition $x_{H+1} = \pi_{H+1} = 0$ |
| $s_t$ | Solution Method | Yes | "Let the active current shock be $s_t = v_t$" |
| $\psi_x$ | Solution Method | Yes | "Guess $x_t = \psi_x s_t$" |
| $\psi_\pi$ | Solution Method | Yes | "Guess $\pi_t = \psi_\pi s_t$" |
| $\psi_i$ | Solution Method | Yes | "$\psi_i = \phi_\pi \psi_\pi + \phi_x \psi_x + 1$" |

Flagged issues:
- **$u_t$ undefined.** The cost-push shock $u_t$ appears in the Phillips curve $\pi_t = \beta M_f \mathbb{E}_t \pi_{t+1} + \kappa x_t + u_t$ but is never defined or described anywhere in the README, and the solution section drops it without explanation (it is zero for the experiments being run).
- **$\varepsilon^v_t$ vs $\varepsilon^v$ inconsistency.** In the AR(1) equation the innovation carries a time subscript ($\varepsilon^v_t$), but the forward-guidance sentence uses the bare form ($\varepsilon^v$) to name the shock amount. The two forms are consistent in meaning but the notation shifts without acknowledgment.

## Summary

The tutorial is accurate and internally consistent. All five references verified correctly, with one MINOR bibliographic issue: the author Jordi Gali's name appears without the acute accent on the final i (should be Gali with accent, Galí). The main message is fully supported by the equations and results. The single most important fix is adding a definition for $u_t$ in the Equations section (even a one-phrase parenthetical such as "where $u_t$ is a cost-push shock, set to zero in these experiments") because the symbol appears in the Phillips curve but is never named or accounted for in the solution. The notation inconsistency between $\varepsilon^v_t$ and $\varepsilon^v$ is a secondary issue. Overall: 1 MINOR reference issue, 1 undefined symbol, 1 notation inconsistency, 0 MAJOR issues, 0 NOT FOUND.
