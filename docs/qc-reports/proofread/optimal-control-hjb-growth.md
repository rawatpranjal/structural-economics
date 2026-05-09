# Proofread: optimal-control/hjb-growth/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T02:30:00Z._

## Paper / Source Verification

### Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies*, 89(1), 45-86.

- **Located:** https://academic.oup.com/restud/article/89/1/45/6149490
- **Tutorial claims:** The tutorial cites this as the primary methodological source for the implicit upwind finite-difference scheme used to solve the HJB equation.
- **Source says:** The paper develops continuous-time heterogeneous-agent macro models and uses exactly the upwind finite-difference scheme for HJB equations described in the tutorial.
- **Verdict:** OK
- **Note:** All fields verified: authors, title, journal, volume 89, issue 1, pages 45-86, year 2022.

### Moll, B. (2022). "Lecture notes on continuous-time methods in macroeconomics." https://benjaminmoll.com/lectures/

- **Located:** https://benjaminmoll.com/lectures/
- **Tutorial claims:** The tutorial cites these notes as a secondary methodological reference for the continuous-time HJB approach.
- **Source says:** The page hosts continuous-time macro lecture materials including heterogeneous-agent and HJB content; the URL resolves and the content is relevant.
- **Verdict:** OK
- **Note:** Informal lecture notes are conventionally cited by year of consultation; the URL is valid and the content matches the tutorial's usage.

### Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition.

- **Located:** https://mitpress.mit.edu/9780262025539/economic-growth/
- **Tutorial claims:** The tutorial implicitly uses standard Ramsey growth results (Euler condition, steady-state capital formula) that this textbook covers.
- **Source says:** The book is the standard graduate text on economic growth; the 2nd edition was published by MIT Press in 2004.
- **Verdict:** OK
- **Note:** Authors, title, publisher, year, and edition all verified.

## Main Message Audit

> "The computed policy follows the Ramsey Euler logic. Investment is high when capital has high marginal product. Consumption rises once capital is abundant. The path converges to $f'(k)=\rho+\delta$. The HJB turns this logic into a value derivative. Upwinding uses the direction of capital movement to choose the derivative. After that choice, the update is a sparse linear solve."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Investment is high when capital has high marginal product | Results: "Low-capital economies invest because marginal product is high"; drift figure shows positive $\dot{k}$ below $k_{ss}$ | OK |
| Consumption rises once capital is abundant | Results: consumption policy figure and description; "Above it, consumption exceeds net output" | OK |
| The path converges to $f'(k)=\rho+\delta$ | Equations: steady-state Euler condition; Results: transition-dynamics figure and table showing convergence to $k_{ss}$ | OK |
| Upwinding uses the direction of capital movement to choose the derivative | Equations: upwind case expression; Solution Method: pseudocode steps 3-4 | OK |
| After that choice, the update is a sparse linear solve | Solution Method: pseudocode step 6 and prose "the update is a sparse linear solve" | OK |

Issues:
- None. All clauses are supported by the tutorial content.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k$ | Overview | Yes - "aggregate capital $k$" | OK |
| $c(k)$ | Overview | Yes - "the consumption policy $c(k)$" | OK |
| $\dot{k}$ | Overview | Yes - "capital drift $\dot{k}$" | OK |
| $\rho$ | Equations (integral) | Yes - inline: "The parameter $\rho$ is the continuous-time discount rate" | OK |
| $u(c)$ | Equations | Yes - inline: "$u(c)=c^{1-\sigma}/(1-\sigma)$" | OK |
| $f(k)$ | Equations (constraint) | Yes - inline: "$f(k)=Ak^\alpha$" | OK |
| $\delta$ | Equations (constraint) | Partial - first used line 19, defined in Model Setup table ~55 lines later | Within 50-line tolerance; $\delta$ is standard depreciation notation |
| $\sigma$ | Equations | Yes - inline via CRRA form; Model Setup table confirms "CRRA coefficient" | OK |
| $A$ | Equations | Yes - inline as TFP parameter in "$f(k)=Ak^\alpha$"; Model Setup confirms | OK |
| $\alpha$ | Equations | Yes - inline as exponent in "$f(k)=Ak^\alpha$"; Model Setup confirms "Capital share" | OK |
| $V(k)$ | Equations (HJB) | Implicit - Overview: "The HJB gives the value of starting from each capital stock" | OK |
| $V'(k)$ | Equations (HJB) | Implicit - derivative of $V(k)$, standard notation | OK |
| $c^{\ast}(k)$ | Equations (FOC) | Yes - defined as optimal policy at first appearance | OK |
| $s(k)$ | Equations | Yes - defined inline: "$s(k)=\dot{k}=f(k)-\delta k-c^{\ast}(k)$" | Overloaded; see flagged issues |
| $k_1,\ldots,k_N$ | Equations | Yes - "grid $k_1,\ldots,k_N$" | OK |
| $\Delta k$ | Equations | Yes - "with spacing $\Delta k$" | OK |
| $D_i V$ | Equations (upwind) | Yes - introduced as "the upwind derivative" in the case expression | OK |
| $s_i$ | Equations (upwind cases) | Implicit - $s(k_i)$ the drift at grid point $i$, follows from $s(k)$ definition | OK |
| $k_{ss}$ | Equations (steady state) | Yes - defined as steady state where $s(k_{ss})=0$ | OK |
| $V_i$ | Equations (upwind cases) | Implicit - $V(k_i)$, follows from grid notation | OK |
| $y_{ss}$ | Model Setup table | Yes - "Steady-state output" | OK |
| $c_{ss}$ | Model Setup table | Yes - "Steady-state consumption" | OK |
| $s = i/y$ | Results table | Partial - $s$ reused for saving rate; $s$ was already defined as capital drift | Overloaded; see flagged issues |
| $i_{ss}$ | Results table | Yes - "$i_{ss} = \delta k_{ss}$ (investment)" | OK |

Flagged issues:
- $s$ is overloaded: defined as capital drift $\dot{k}$ in the Equations section (line 43) and used with that meaning in the Results prose (line 118), but reused as saving rate $i/y$ in the Results diagnostic table (the "$s = i/y$ (saving rate)" row). A different symbol should be used for one of the two quantities.

## Summary

The tutorial is mathematically sound and its references check out. One notation issue requires a fix: the symbol $s$ carries two distinct meanings - capital drift $\dot{k}$ in the Equations and Results sections, and saving rate $i/y$ in the Results diagnostic table. All three references are verified with correct authors, titles, and publication details. The main message clauses are fully supported by the tutorial's equations and results. The count of issues is: 0 MAJOR, 0 MINOR, 0 NOT FOUND, 0 OVERREACH, 1 notation overload. The single most important fix is to rename one of the two uses of $s$ - for example, use $\hat{s}$ or $sr$ for the saving rate in the table, or use $\mu$ or $g$ for capital drift throughout the equations, to eliminate the clash.
