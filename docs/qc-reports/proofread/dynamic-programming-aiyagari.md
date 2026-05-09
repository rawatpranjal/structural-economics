# Proofread: dynamic-programming/aiyagari/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T22:30:00Z._

## Paper / Source Verification

### Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.

- **Located:** https://academic.oup.com/qje/article-abstract/109/3/659/1838287
- **Tutorial claims:** Aiyagari (1994) introduces an incomplete-markets economy where households face persistent idiosyncratic income risk, cannot borrow, and save in a risk-free asset; the equilibrium interest rate clears the capital market.
- **Source says:** Title, author (S. Rao Aiyagari), journal (The Quarterly Journal of Economics), volume 109, issue 3, pages 659-684, year 1994 all match the Oxford Academic publisher record. DOI: 10.2307/2118417.
- **Verdict:** OK
- **Note:** All bibliographic fields verified.

### Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.

- **Located:** https://mitpress.mit.edu/9780262038669/recursive-macroeconomic-theory/
- **Tutorial claims:** The 4th edition (MIT Press, 2018) provides the textbook treatment; Chapter 18 covers the relevant incomplete-markets material.
- **Source says:** MIT Press confirms 4th edition, 2018, ISBN 9780262038669, authors Lars Ljungqvist and Thomas J. Sargent. Chapter 18 is titled "Incomplete Markets Models" (Part IV: Savings Problems and Bewley Models), covering the Aiyagari heterogeneous-agent economy.
- **Verdict:** OK
- **Note:** Year, publisher, edition, and chapter topic all verified.

## Main Message Audit

> "Precautionary saving turns a household policy into an aggregate capital supply curve. In this calibration, capital-market clearing gives $r^{\ast}=0.0260$, below $1/\beta-1=0.0417$. The lower rate is the price of incomplete insurance. The computation shows how VFI, a stationary distribution, and bisection close the model."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Precautionary saving turns a household policy into an aggregate capital supply curve | Equations ($K^s(r)$ derived from asset policy and stationary distribution); Results (capital-market figure showing upward-sloping $K^s(r)$) | OK |
| Capital-market clearing gives $r^{\ast}=0.0260$ | Results table ($r^{\ast}=0.025959$) | OK |
| Below $1/\beta-1=0.0417$ | Model Setup ($1/\beta-1=0.0417$ with $\beta=0.96$); Equations ("A standard result is $r^{\ast}<1/\beta-1$") | OK |
| The lower rate is the price of incomplete insurance | Equations (precautionary saving becomes unbounded at $r \ge 1/\beta-1$ under no-borrowing constraint) | OK |
| The computation shows how VFI, a stationary distribution, and bisection close the model | Solution Method (pseudocode and prose describe exactly these three components) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|-----------------|----------|-------|
| $a_t$ | Equations | Yes | "beginning-of-period assets" |
| $\underline a$ | Equations | Yes | "no-borrowing constraint $\underline a=0$" |
| $\bar a$ | Equations | Partial | Introduced as upper bound of asset bracket $[\underline a,\bar a]$; reused as mean wealth in Results - two distinct objects |
| $z_t$ | Equations | Yes | "idiosyncratic labor efficiency" |
| $r$, $w$ | Equations | Yes | prices $(r,w)$; fully characterized as firm FOCs in the firm subsection |
| $a'$ | Equations | Yes | "$a_{t+1}=a'$" |
| $c_t$ | Equations | Yes | budget constraint |
| $U_0$ | Equations | Yes | lifetime utility |
| $\beta$ | Equations | Yes | "$\beta\in(0,1)$" |
| $u(c)$ | Equations | Yes | CRRA formula |
| $\sigma$ | Equations | Yes | "$\sigma>0$" |
| $\rho$ | Equations | Yes | AR(1) persistence |
| $\varepsilon_{t+1}$ | Equations | Yes | AR(1) innovation |
| $\sigma_\varepsilon$ | Equations | Yes | "$\mathcal{N}(0,\sigma_\varepsilon^2)$" |
| $N$ | Equations | Yes | "$N$-state Rouwenhorst chain" |
| $\{z_j\}$ | Equations | Yes | "The grid is $\{z_j\}$" |
| $P_{jk}$ | Equations | Yes | "$P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$" |
| $j$, $k$ | Equations | Yes | income-state indices from $P_{jk}$ definition |
| $V(a,z_j)$ | Equations | Yes | Bellman equation (standard value-function notation) |
| $g_a(a,z_j)$ | Equations | Yes | "asset policy $g_a(a,z_j)$" |
| $c^{\ast}(a,z_j)$ | Equations | Yes | "consumption policy" defined after Bellman |
| $\mu(a',z_k)$ | Equations | Yes | "long-run distribution $\mu$ over $(a,z)$" |
| $i$ | Equations (distribution) | No | Used as asset-grid index in $\sum_{i:\,g_a(a_i,z_j)=a'}$ and $\sum_{i,j} a_i\,\mu(a_i,z_j)$ without definition; income indices $j$ and $k$ are defined via $P_{jk}$ but the asset index $i$ is not |
| $a_i$ | Equations (distribution) | Partial | Implied asset-grid node; no explicit "$\{a_i\}$ is the asset grid" |
| $K^s(r)$ | Equations | Yes | "Aggregate household assets are $K^s(r)$" |
| $Y$, $K$, $L$ | Equations (firm) | Yes | Cobb-Douglas; $L=1$ stated explicitly |
| $\alpha$ | Equations (firm) | Yes | "capital share $\alpha$" |
| $\delta$ | Equations (firm) | Yes | "depreciation $\delta$" |
| $K^d(r)$ | Equations (firm) | Yes | "capital demand at $r$" |
| $r^{\ast}$, $w^{\ast}$ | Equations | Yes | stationary equilibrium prices |
| $r_H$, $r_L$ | Model Setup table | Partial | Used in "Backup stop on $r_H-r_L$" before the pseudocode introduces them as bracket endpoints; meaning is clear from context |
| $\tilde a$ | Results | Yes | "median wealth ($\tilde a=4.47$)" - defined at first use |
| $\bar a$ (mean) | Results | Partial | Defined inline as "Mean wealth ($\bar a=6.76$)" but collides with Equations usage (see above) |
| $G$ | Results | Yes | "Gini $G=0.526$" - defined at first use |

Flagged issues:
- **$\bar a$ overloaded**: In the Equations section $\bar a$ is the upper bound of the asset bracket $[\underline a,\bar a]$ (= 50 in the calibration). In the Results section and the equilibrium diagnostics table, the same symbol denotes mean household wealth (= 6.76). These are two distinct quantities.
- **$i$ undefined**: The index $i$ runs over asset-grid nodes in the stationary-distribution operator and the aggregate-capital sum. Income indices $j$ and $k$ are explicitly introduced via $P_{jk}$, but $i$ is not defined anywhere in the README.

## Summary

Both cited references are bibliographically correct and the tutorial's description of each matches the source. All five clauses of the takeaway are directly supported by the README's equations, solution method, and results, with no overreach or unsupported claims. The report finds 0 MAJOR, 0 MINOR paper issues, 0 NOT FOUND, 0 OVERREACH, and 2 notation flags: the most important is that $\bar a$ is used for two distinct objects (upper asset-grid bound in Equations, mean wealth in Results), and secondarily that the asset-grid index $i$ is used in the stationary-distribution equations without explicit definition.
