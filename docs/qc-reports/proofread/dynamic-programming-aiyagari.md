# Proofread: dynamic-programming/aiyagari/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T19:10:00Z._

## Paper / Source Verification

### Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.

- **Located:** https://academic.oup.com/qje/article-abstract/109/3/659/1838287
- **Tutorial claims:** Aiyagari (1994) introduces incomplete-markets economy where households face idiosyncratic income risk, cannot borrow, and save in a risk-free asset; the equilibrium interest rate clears the capital market.
- **Source says:** Paper title, author (S. Rao Aiyagari), journal (The Quarterly Journal of Economics), volume 109, issue 3, pages 659–684, year 1994 all match the Oxford Academic publisher record.
- **Verdict:** OK
- **Note:** All bibliographic fields are correct.

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** The 4th edition (MIT Press, 2018) covers incomplete-markets models in Ch. 18.
- **Source says:** MIT Press catalog confirms 4th edition, 2018, ISBN 9780262038669; Ch. 18 is titled "Incomplete Markets Models" within Part IV (Savings Problems and Bewley Models), covering the Aiyagari economy.
- **Verdict:** OK
- **Note:** Year, publisher, edition, and chapter topic all match.

## Main Message Audit

> "Precautionary saving turns a household policy into an aggregate capital supply curve. In this calibration, capital-market clearing gives $r^{\ast}=0.0260$, below $1/\beta-1=0.0417$. The lower rate is the price of incomplete insurance. The computation shows how VFI, a stationary distribution, and bisection close the model."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Precautionary saving turns a household policy into an aggregate capital supply curve | Equations (K^s(r) derived from asset policy and stationary distribution); Figure 1 (capital-market picture showing upward-sloping K^s(r)) | OK |
| Capital-market clearing gives r*=0.0260 | Results table (r* = 0.025959) | OK |
| Below 1/β-1=0.0417 | Model Setup table (1/β-1 = 0.0417 with β=0.96); Equations ("A standard result is r*<1/β-1") | OK |
| The lower rate is the price of incomplete insurance | Equations (precautionary saving becomes unbounded at r ≥ 1/β-1 under no-borrowing constraint) | OK |
| The computation shows how VFI, a stationary distribution, and bisection close the model | Solution Method (pseudocode and prose describe exactly these three components) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $a_t$ | Equations, "beginning-of-period assets" | Yes | State variable |
| $\underline a$ | Equations, "no-borrowing constraint $\underline a=0$" | Yes | Borrowing floor |
| $\bar a$ | Equations, "Let $a_t\in[\underline a,\bar a]$" | Yes, as asset upper bound | **Overloaded**: reused as mean wealth in Results ("Mean wealth ($\bar a=6.76$)") and in the diagnostics table |
| $z_t$ | Equations, "idiosyncratic labor efficiency" | Yes | |
| $r$, $w$ | Equations, "With prices $(r,w)$" | Implicitly (standard notation) | Fully defined as firm FOCs in firm subsection |
| $a'$ | Equations, "$a_{t+1}=a'$" | Yes | |
| $c_t$ | Equations, budget constraint | Yes | |
| $U_0$ | Equations, utility sum | Yes | |
| $\beta$ | Equations, "$\beta\in(0,1)$" | Yes | |
| $u(c)$ | Equations, CRRA definition | Yes | |
| $\sigma$ | Equations, "$\sigma>0$" | Yes | CRRA curvature |
| $\rho$ | Equations, AR(1) equation | Yes | |
| $\varepsilon_{t+1}$ | Equations, AR(1) equation | Yes | |
| $\sigma_\varepsilon$ | Equations, AR(1) variance | Yes | |
| $N$ | Equations, "N-state Rouwenhorst chain" | Yes | |
| $\{z_j\}$, $P_{jk}$ | Equations, Rouwenhorst chain | Yes | |
| $V(a,z_j)$ | Equations, Bellman equation | Implicit (standard value function) | |
| $g_a(a,z_j)$ | Equations, "asset policy" | Yes | |
| $c^{\ast}(a,z_j)$ | Equations, "consumption policy" | Yes | |
| $\mu$ | Equations, "long-run distribution $\mu$ over $(a,z)$" | Yes | |
| $K^s(r)$ | Equations, "Aggregate household assets" | Yes | |
| $K$, $L$, $Y$ | Equations (firm), Cobb-Douglas | Yes; $L=1$ stated explicitly | |
| $\alpha$ | Equations (firm), "capital share $\alpha$" | Yes | |
| $\delta$ | Equations (firm), "depreciation $\delta$" | Yes | |
| $K^d(r)$ | Equations (firm), "capital demand at $r$" | Yes | |
| $r^{\ast}$, $w^{\ast}$ | Equations, stationary equilibrium definition | Yes | |
| $\tilde a$ | Results, "median wealth ($\tilde a=4.47$)" | Yes, at first use | |
| $G$ | Results, "Gini $G=0.526$" | Yes, at first use | |
| $K^{\ast}$ | Results table, "Aggregate capital $K^{\ast}$" | Implicit (K at equilibrium) | |
| $Y^{\ast}$ | Results table, "Output $Y^{\ast}$" | Implicit (Y at equilibrium; Y defined in Equations) | |

Flagged issues:
- **$\bar a$ overloaded**: In the Equations section, $\bar a$ denotes the upper bound of the asset bracket (confirmed as 50 in the Model Setup table). In the Results section and the diagnostics table, the same symbol is reused for mean wealth (6.76). The two uses refer to distinct quantities.

## Summary

Both cited references are bibliographically correct and the tutorial's description of each aligns with the source. All five clauses of the stated takeaway are directly supported by the README's equations, solution method, and results; no overreach was found. The sole flagged issue is a notation conflict: $\bar a$ is first introduced in the Equations section as the upper bound of the asset grid (= 50) and later reused in the Results section and the equilibrium diagnostics table as mean wealth (= 6.76). This yields 0 MAJOR, 0 MINOR paper issues, 0 NOT FOUND, 0 OVERREACH, and 1 notation flag. The single most important fix is to replace one of the two uses of $\bar a$ with a distinct symbol — for example, using $a_{\max}$ for the grid upper bound or $\mathbb{E}[a]$ / $\mu_a$ for mean wealth — to eliminate the overloading.
