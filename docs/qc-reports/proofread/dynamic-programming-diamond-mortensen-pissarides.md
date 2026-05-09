# Proofread: dynamic-programming/diamond-mortensen-pissarides/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T19:15:00Z._

## Paper / Source Verification

### Diamond, P. (1982). "Aggregate Demand Management in Search Equilibrium." *Journal of Political Economy*, 90(5), 881-894.

- **Located:** https://ideas.repec.org/a/ucp/jpolec/v90y1982i5p881-94.html
- **Tutorial claims:** The tutorial cites this as the foundational search-equilibrium paper; the title, journal, volume, issue, and page range are as stated.
- **Source says:** Title, journal (*Journal of Political Economy*), volume 90, issue 5, pages 881-894, year 1982 all confirmed via IDEAS/RePEC and the University of Chicago Press DOI 10.1086/261099.
- **Verdict:** OK
- **Note:** Citation is exact.

### Mortensen, D. and Pissarides, C. (1994). "Job Creation and Job Destruction in the Theory of Unemployment." *Review of Economic Studies*, 61(3), 397-415.

- **Located:** https://academic.oup.com/restud/article-abstract/61/3/397/1589192
- **Tutorial claims:** The tutorial cites this as the core DMP job-creation/destruction paper; the title, journal, volume, issue, and page range are as stated.
- **Source says:** Title, journal (*Review of Economic Studies*), volume 61, issue 3, pages 397-415, year 1994 all confirmed via Oxford Academic publisher page.
- **Verdict:** OK
- **Note:** Citation is exact.

### Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.

- **Located:** https://mitpress.mit.edu/9780262533980/equilibrium-unemployment-theory/
- **Tutorial claims:** The tutorial cites this as the standard textbook treatment of DMP theory; author, title, publisher, year, and edition are as stated.
- **Source says:** Author (Christopher A. Pissarides), title (*Equilibrium Unemployment Theory*), publisher (MIT Press), 2nd edition, year 2000 all confirmed via MIT Press catalog and Open Library.
- **Verdict:** OK
- **Note:** Citation is exact.

### Shimer, R. (2005). "The Cyclical Behavior of Equilibrium Unemployment and Vacancies." *American Economic Review*, 95(1), 25-49.

- **Located:** https://www.aeaweb.org/articles?id=10.1257/0002828053828572
- **Tutorial claims:** The tutorial references Shimer's finding that tightness is "near 19" times as volatile as productivity, and frames the Shimer puzzle as the baseline DMP model generating far too little amplification.
- **Source says:** Title, journal (*American Economic Review*), volume 95, issue 1, pages 25-49, year 2005 confirmed via AEA publisher page. The paper documents that the V/U ratio is approximately 20 times as volatile as productivity; the tutorial's characterization as "near 19" is a slight understatement of the ~20× ratio.
- **Verdict:** OK
- **Note:** "Near 19" vs. "approximately 20" is within normal rounding for a tutorial context; the directional claim and framing of the puzzle are accurate.

### Hagedorn, M. and Manovskii, I. (2008). "The Cyclical Behavior of Equilibrium Unemployment and Vacancies Revisited." *American Economic Review*, 98(4), 1692-1706.

- **Located:** https://www.aeaweb.org/articles?id=10.1257/aer.98.4.1692
- **Tutorial claims:** The tutorial implicitly references the Hagedorn-Manovskii mechanism through the sensitivity table showing how a high flow value of unemployment (small surplus) raises tightness elasticity.
- **Source says:** Title, journal (*American Economic Review*), volume 98, issue 4, pages 1692-1706, year 2008 confirmed via AEA publisher page.
- **Verdict:** OK
- **Note:** Citation is exact.

---

## Main Message Audit

> "DMP links productivity to vacancies and unemployment through free entry. The local rule and nonlinear fixed point give almost the same volatility. The Shimer puzzle therefore comes from the large baseline surplus, not from the numerical method. The sensitivity table shows how a smaller surplus raises tightness amplification."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| DMP links productivity to vacancies and unemployment through free entry | Equations: free-entry condition pins θ_t; Stock dynamics map θ_t to u_t and v_t | OK |
| The local rule and nonlinear fixed point give almost the same volatility | Results table: nonlinear Std./Std. z = 1.72, log-linear = 1.55; both far from data | OK |
| The Shimer puzzle comes from the large baseline surplus, not from the numerical method | Results: both solvers underamplify by similar amounts; sensitivity table shows C rises sharply as surplus shrinks | OK |
| The sensitivity table shows how a smaller surplus raises tightness amplification | Results: amplification table shows C from 1.55 (b=0.40) to 18.65 (b=0.95) | OK |

Issues:
- None.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $z_t - b$ | Overview (surplus phrase) | Partial | Both $z_t$ and $b$ defined formally in Equations within ~25 lines; acceptable |
| $\theta_t = v_t/u_t$ | Overview | Yes | Defined inline at first use |
| $u_t$ | Equations - Matching technology | Yes | Defined inline |
| $v_t$ | Equations - Matching technology | Yes | Defined inline |
| $\chi$ | Equations - Matching technology | Yes | Defined by role in $m(u,v)$ equation |
| $\eta$ | Equations - Matching technology | Yes | Defined by role in $m(u,v)$ equation |
| $f(\theta_t)$ | Equations - Matching technology | Yes | Defined as worker job-finding rate |
| $q(\theta_t)$ | Equations - Matching technology | Yes | Defined as firm vacancy-filling rate |
| $\hat{z}_t$ | Equations - Productivity | Yes | Defined as log-productivity deviation |
| $\rho$ | Equations - Productivity | Yes | Defined as AR(1) coefficient |
| $\epsilon_{t+1}$ | Equations - Productivity | Yes | Defined as i.i.d. productivity shock |
| $\sigma_\epsilon$ | Equations - Productivity | Yes | Defined as innovation standard deviation |
| $z_t = \bar{z}\exp(\hat{z}_t)$ | Equations - Productivity | Yes | Level productivity defined here |
| $\bar{z}$ | Equations - Productivity | Partial | Introduced without explicit value; value $\bar{z}=1$ only derivable from the Model Setup table row "Surplus $\bar{z}-b$ = 0.60" combined with $b=0.40$ |
| $w_t$ | Equations - Wage rule | Yes | Defined as equilibrium wage |
| $\gamma$ | Equations - Wage rule | Yes | Defined as worker bargaining weight |
| $b$ | Equations - Wage rule | Yes | Defined as flow value of unemployment |
| $k$ | Equations - Wage rule | Yes | Defined as per-period vacancy cost |
| $J_t$ | Equations - Job value | Yes | Defined as filled-job value |
| $\beta$ | Equations - Job value | Partial | First use in Job value equation; defined in Model Setup table (~30 lines later, within 50-line threshold) |
| $\sigma$ | Equations - Job value | Yes | Defined as exogenous separation rate |
| $u_{ss}$ | Equations - Stock dynamics | Yes | Defined as deterministic steady-state unemployment |
| $\theta_{ss}$ | Equations - Stock dynamics | Partial | First use in Stock dynamics; value $\theta_{ss}=1$ stated explicitly only in Local linearization (~8 lines later) |
| $\hat\theta_t$ | Equations - Local linearization | Yes | Defined as log deviation of tightness |
| $C$ | Equations - Local linearization | Yes | Defined as tightness elasticity w.r.t. log productivity |
| $A$, $B$ | Equations - Local linearization | Yes | Defined by closed-form expressions in same display equation |
| $P_{ij}$ | Solution Method - Nonlinear Bellman | Partial | Not defined in the display equation; defined as "transition matrix P" in Algorithm 2 inputs two lines below |
| $N_z$ | Model Setup table | Yes | Defined as number of Rouwenhorst nodes |
| $w_{ss}$ | Model Setup table | Yes | Defined as steady-state wage in table |

Flagged issues:
- **$\bar{z}$ value undefined in table**: The symbol is introduced in the Equations section as part of the level-productivity formula $z_t = \bar{z}\exp(\hat{z}_t)$ but the Model Setup table never lists $\bar{z}$ with an explicit numerical value. Its value ($\bar{z}=1$) must be inferred from the "Surplus $\bar{z}-b$ = 0.60" row together with $b=0.40$. Adding a dedicated row "Mean productivity $\bar{z}$ | 1.00 | Normalization" would make the table self-contained.

---

## Summary

All five references are bibliographically correct (5 OK, 0 MINOR, 0 MAJOR, 0 NOT FOUND). The main message is fully supported by the tutorial's own equations and results tables (4 clauses, all OK, 0 OVERREACH, 0 UNSUPPORTED). The single notation issue is that $\bar{z}$ is introduced in the Equations section without an explicit value entry in the Model Setup table; the value 1.00 is inferable but not stated. No other symbols are undefined, overloaded, or drift across sections. The most important fix is adding a $\bar{z} = 1.00$ row (or equivalent explicit statement) to the Model Setup table.
