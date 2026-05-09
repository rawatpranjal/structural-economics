# Proofread: heterogeneous-agents/huggett-incomplete-markets/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T05:35:00Z._

## Paper / Source Verification

### Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control* 17(5-6), 953-969.

- **Located:** https://ideas.repec.org/a/eee/dyncon/v17y1993i5-6p953-969.html
- **Tutorial claims:** The Huggett (1993) paper is cited as the source for the incomplete-markets heterogeneous-agent economy with a zero-net-supply bond market and the equilibrium risk-free rate determination.
- **Source says:** Author Mark Huggett, title "The risk-free rate in heterogeneous-agent incomplete-insurance economies," Journal of Economic Dynamics and Control, volume 17, issue 5-6, pages 953-969, year 1993.
- **Verdict:** OK
- **Note:** All bibliographic fields match the publisher record exactly.

### Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1), 45-86.

- **Located:** https://academic.oup.com/restud/article/89/1/45/6149490
- **Tutorial claims:** The Achdou et al. (2022) paper is cited as the source for the HJB-KFE continuous-time approach used to solve the household problem and recover the stationary density.
- **Source says:** Authors Yves Achdou, Jiequn Han, Jean-Michel Lasry, Pierre-Louis Lions, Benjamin Moll; Review of Economic Studies volume 89, issue 1, pages 45-86, year 2022.
- **Verdict:** OK
- **Note:** Author order, journal, volume, issue, page range, and year all match the Oxford Academic publisher record.

### Moll, B. "Lecture notes on continuous-time heterogeneous-agent models." https://benjaminmoll.com/lectures/

- **Located:** https://benjaminmoll.com/lectures/
- **Tutorial claims:** The reference is described as lecture notes specifically on continuous-time heterogeneous-agent models.
- **Source says:** The page (titled "Lectures - Benjamin Moll") covers continuous-time heterogeneous-agent material (HJB, KFE, mean field games, distributional macroeconomics) alongside a broader range of topics including HANK models, intermediate macro, financial crises, and energy policy.
- **Verdict:** OK
- **Note:** The URL resolves and the description is a fair characterization of a central strand of the page's content; the narrowness of the description is within acceptable limits for a lecture-note reference.

## Main Message Audit

> "The Huggett price is a market-clearing return. Income risk and the borrowing limit make households want buffer wealth at $r = \rho$. The bond market clears only at a lower return. In this run the wedge is $\rho - r^{\ast} = 0.0150$."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| The Huggett price is a market-clearing return | Equations: $S(r^{\ast}) = 0$; bond-market figure | OK |
| Income risk and the borrowing limit make households want buffer wealth at $r = \rho$ | Stated as motivation in Overview; not formally derived in Equations or Results | OVERREACH |
| The bond market clears only at a lower return | Results table: $r^{\ast} = 0.03499 < \rho = 0.05$; bond-market figure | OK |
| In this run the wedge is $\rho - r^{\ast} = 0.0150$ | Results table: Precautionary wedge = 0.01501 | OK |
| The HJB/KFE loop ties the household policy to the stationary cross section | Solution Method: HJB gives policies; KFE maps drift to density | OK |
| The upwind HJB respects the borrowing limit | Solution Method: state constraint and upwind scheme explained | OK |
| The KFE measures aggregate asset demand | Equations: $S(r) = \int a [g_L + g_H] da$; Solution Method: $A^{\top} g = 0$ | OK |
| Bisection on $r$ closes the zero-net-supply bond market | Solution Method pseudocode; Results: bond-market residual $5.43\times 10^{-6}$ | OK |

Issues:
- "Income risk and the borrowing limit make households want buffer wealth at $r = \rho$" is the economic intuition motivating the precautionary wedge. The tutorial demonstrates $r^{\ast} < \rho$ numerically but does not derive or formally show that incomplete insurance generates a desire for buffer wealth specifically at $r = \rho$. The claim is standard Huggett theory but is asserted rather than demonstrated within the README.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $a$ | Overview, "borrow only down to $a \geq \underline a$" | Implicit - asset holdings | Clear from context |
| $\underline a$ | Overview | Model Setup table, line ~70 | Within 50 lines of first use; Partial |
| $\bar a$ | Equations, bond-market integral | Model Setup table, line ~71 | Within 50 lines; Partial |
| $r^{\ast}$ | Overview | Defined in Overview ("the equilibrium return $r^{\ast}$") | OK |
| $S(r)$ | Overview ("stationary bond demand $S(r^{\ast})$") | Equations section: $S(r) \equiv \int a [g_L + g_H] da$ | OK - but see flag below |
| $\rho$ | Overview, "at $r = \rho$" | Equations section (~13 lines later): "discount rate $\rho$" | Late by 13 lines; Partial |
| $i$ | Equations ($i \in \{L, H\}$) | Defined at first use | OK |
| $j$ | Equations (HJB: $V_j(a)$) | Defined in same sentence: "the other state $j$" | OK |
| $z_i$ | Equations | Defined in Equations: "receives endowment $z_i$" | OK |
| $\lambda_i$ | Equations | Defined in Equations: "Poisson intensity $\lambda_i$" | OK |
| $\dot a$ | Equations | Defined in Equations: "Assets move between jumps by" | OK |
| $s_i(a)$ | Equations | Defined in same display: $\dot a = s_i(a)$ | OK |
| $c_i(a)$ | Equations | Defined in FOC: "$c_i(a) = [V_i'(a)]^{-1/\sigma}$" | OK |
| $V_i(a)$ | Equations | Defined by context: "The value function" | OK |
| $u(c)$ | Equations (HJB) | Implicit via "CRRA utility"; form not written out | CRRA is standard - not flagged |
| $\sigma$ | Equations, FOC ($-1/\sigma$) | Prose says "CRRA utility"; Model Setup table defines "$\sigma$" ~34 lines later | Within 50 lines; Partial |
| $g_i(a)$ | Equations | Defined in Equations: "The stationary density $g_i(a)$" | OK |
| $\mathbf{I}$ | Solution Method, implicit step | Standard identity matrix; not defined explicitly | Standard notation - not flagged |
| $A^{n}$ | Solution Method, implicit step | Defined in same paragraph: "the upwind generator" | OK |
| $\Delta$ | Solution Method, implicit step | Model Setup table: "Implicit step $\Delta$" | OK |
| $V^{n+1}$, $V^{n}$ | Solution Method, implicit step | Iteration index $n$ clear from pseudocode context | OK |
| $c^{n}$ | Solution Method, implicit step | Introduced here; clear from iteration context | OK |
| $I_{\rm ref}$ | Solution Method, Reference solve | Defined inline: "$I_{\rm ref} = 6000$ points" | OK |
| $p_L, p_H$ | Model Setup | Defined inline: "The symmetric income chain implies $p_L = p_H = 0.5$" | OK |
| $\bar z$ | Model Setup | Defined inline in table and prose | OK |

Flagged issues:
- $S(r)$ terminology drift: defined as "stationary bond demand" (Overview) and "aggregate assets" (Equations), but the Results section calls the figure a "supply curve" in the same sentence that says it "plots aggregate asset demand." The phrase "The supply curve plots aggregate asset demand against $r$" is internally contradictory within the Results section.

## Summary

The tutorial is well-structured and the three cited references all check out with correct bibliographic details. There are no MAJOR issues, no NOT FOUND references, and one MINOR notation drift: the Results section describes the bond-market figure as a "supply curve" while the symbol $S(r)$ is defined and used consistently as "aggregate asset demand" everywhere else (1 MINOR, 0 MAJOR, 0 NOT FOUND, 1 OVERREACH). The single most important fix is resolving the internal contradiction in the Results section figure description, where "The supply curve plots aggregate asset demand against $r$" should use one term consistently for $S(r)$.
