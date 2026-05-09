# Proofread: dsge/nkdsge/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T05:35:00Z._

## Paper / Source Verification

### Gali, J. (2015). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press, 2nd edition.

- **Located:** https://press.princeton.edu/books/hardcover/9780691164786/monetary-policy-inflation-and-the-business-cycle
- **Tutorial claims:** Book by Jordi Gali, second edition, Princeton University Press, 2015, titled "Monetary Policy, Inflation, and the Business Cycle."
- **Source says:** Full title is "Monetary Policy, Inflation, and the Business Cycle: An Introduction to the New Keynesian Framework and Its Applications - Second Edition," published June 2015 by Princeton University Press.
- **Verdict:** OK
- **Note:** The tutorial cites the short title only; the subtitle is omitted, which is standard practice.

### Woodford, M. (2003). *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.

- **Located:** https://press.princeton.edu/books/hardcover/9780691010496/interest-and-prices
- **Tutorial claims:** Book by Michael Woodford, 2003, Princeton University Press, titled "Interest and Prices: Foundations of a Theory of Monetary Policy."
- **Source says:** Identical title, author, year, and publisher.
- **Verdict:** OK
- **Note:** No discrepancies.

### Clarida, R., Gali, J., and Gertler, M. (1999). The Science of Monetary Policy: A New Keynesian Perspective. *Journal of Economic Literature*, 37(4), 1661-1707.

- **Located:** https://www.aeaweb.org/articles?id=10.1257%2Fjel.37.4.1661
- **Tutorial claims:** Article by Clarida, Gali, and Gertler, 1999, in Journal of Economic Literature, volume 37, issue 4, pages 1661-1707.
- **Source says:** Same title, authors, journal, volume 37, issue 4, pages 1661-1707, December 1999.
- **Verdict:** OK
- **Note:** No discrepancies.

### Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.

- **Located:** https://ideas.repec.org/a/eee/dyncon/v24y2000i10p1405-1423.html
- **Tutorial claims:** Article by Paul Klein, 2000, in Journal of Economic Dynamics and Control, volume 24, issue 10, pages 1405-1423.
- **Source says:** Title as indexed is "Using the generalized Schur form to solve a multivariate linear rational expectations model" (all lowercase after the first word). Author, journal, volume, issue, and pages match exactly.
- **Verdict:** MINOR
- **Note:** The tutorial capitalizes interior words of the title ("Generalized Schur Form," "Multivariate Linear Rational Expectations Model"); the published record uses lowercase throughout.

## Main Message Audit

> "The three-equation New Keynesian model shows how sticky prices make nominal policy matter. A policy wedge raises the real rate and contracts demand. A natural-rate shock expands demand and inflation, with the Taylor rule leaning back. Coefficient matching is enough because the model is log-linear. The Klein QZ check confirms the same stable equilibrium. Determinacy is economic here: inflation feedback selects one forward-looking path."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Sticky prices make nominal policy matter | Equations section (NKPC with $\kappa > 0$ encodes sticky prices; IS curve shows real-rate channel) | OK |
| A policy wedge raises the real rate and contracts demand | Results (output gap and inflation are negative on impact after monetary shock) | OK |
| A natural-rate shock expands demand and inflation | Results (positive output gap and inflation impact for demand shock) | OK |
| The Taylor rule leans back against a demand shock | Results (nominal rate rises by 1.715 pp on impact of demand shock) | OK |
| Coefficient matching is enough because the model is log-linear | Solution Method (undetermined-coefficients derivation closes in one scalar equation) | OK |
| Klein QZ check confirms the same stable equilibrium | Solution Method (states "differ by at most 1.4e-15") | OK |
| Inflation feedback selects one forward-looking path | Model Setup (notes $\phi_\pi = 1.5 > 1$ keeps "the Taylor rule active enough to select one stable path"; Blanchard-Kahn is the standard criterion) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $y_t$ | Overview | Yes - Equations prose ("Let $y_t$ be the output gap") | OK |
| $\pi_t$ | Overview | Yes - Equations prose ("$\pi_t$ inflation") | OK |
| $i_t$ | Overview | Yes - Equations prose ("$i_t$ the policy rate") | OK |
| $v_t$ | Overview | Yes - Overview ("Taylor-rule wedge $v_t$") | OK |
| $d_t$ | Overview | Yes - Overview ("natural-rate demand shock $d_t$") | OK |
| $r^n_t$ | Equations (IS curve) | Yes - Equations prose ("$r^n_t$ the natural real rate") | OK |
| $\mathbb{E}_t$ | Equations (IS curve) | Not explicit | Standard conditional-expectations operator; audience would know |
| $\sigma$ | Equations (IS curve) | Acceptable - Model Setup table within 30 lines | "Inverse EIS in the IS curve" |
| $\beta$ | Equations (NKPC) | Acceptable - Model Setup table within 30 lines | "Quarterly discount factor" |
| $\kappa$ | Equations (NKPC) | Acceptable - Model Setup table within 30 lines | "Slope of the New Keynesian Phillips curve" |
| $\phi_\pi$ | Equations (Taylor rule) | Acceptable - Model Setup table within 30 lines | "Taylor-rule response to inflation" |
| $\phi_y$ | Equations (Taylor rule) | Acceptable - Model Setup table within 30 lines | "Taylor-rule response to the output gap" |
| $\rho_v$ | Equations (policy wedge AR(1)) | Acceptable - Model Setup table within 30 lines | "Persistence of the policy shock" |
| $\varepsilon^v_t$ | Equations (policy wedge AR(1)) | Partial - not named in prose | Implicit as innovation in AR(1); standard notation |
| $\rho_d$ | Equations (demand shock AR(1)) | Acceptable - Model Setup table within 30 lines | "Persistence of the demand shock" |
| $\varepsilon^d_t$ | Equations (demand shock AR(1)) | Partial - not named in prose | Implicit as innovation in AR(1); standard notation |
| $s_t$ | Solution Method | Yes - Solution Method ("Let the active shock be $s_t$") | Unified notation for either shock |
| $\rho_s$ | Solution Method | Yes - defined with $s_t$ | Persistence of generic shock |
| $\varepsilon_t$ | Solution Method | Yes - defined with $s_t$ | Generic shock innovation |
| $\psi_y$ | Solution Method | Yes - Solution Method ("They map the shock state into output and inflation") | Generic output coefficient |
| $\psi_\pi$ | Solution Method | Yes - alongside $\psi_y$ | Generic inflation coefficient |
| $b_s$ | Solution Method (scalar equation) | Yes - defined immediately after equation | "$b_s = -1/\sigma$ for a policy wedge and $b_s = 1$ for a demand shock" |

Flagged issues:
- None. The two innovation terms $\varepsilon^v_t$ and $\varepsilon^d_t$ are not named in prose, but they appear only inside AR(1) law-of-motion equations where their role is unambiguous, and the target audience (quantitative economists) would recognize them as i.i.d. innovations without a label.

## Summary

The tutorial is clean. Paper verification found 1 MINOR issue (interior-word capitalization in the Klein 2000 title), 0 MAJOR issues, and 0 NOT FOUND citations. The main message audit found no OVERREACH or UNSUPPORTED clauses; every claim in the Overview and Takeaway is directly supported by the Equations, Solution Method, or Results sections. The notation is complete and internally consistent; no symbol is undefined, late-defined, overloaded, or used for two distinct objects across sections. The single most important fix is the Klein (2000) title capitalization, which is cosmetic.
