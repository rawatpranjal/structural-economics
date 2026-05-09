# Proofread: structural-econometrics/keane-wolpin-career-choice/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T09:15:00Z._

## Paper / Source Verification

### Keane, M. P. and Wolpin, K. I. (1997). The Career Decisions of Young Men. *Journal of Political Economy*, 105(3), 473-522.

- **Located:** https://www.journals.uchicago.edu/doi/10.1086/262080
- **Tutorial claims:** This is the source paper for the career-choice model with schooling, blue-collar work, white-collar work, and home choices; the README frames the tutorial as a "Keane-Wolpin career choice" model.
- **Source says:** Journal of Political Economy, vol. 105, no. 3, pp. 473-522, 1997, by Michael P. Keane and Kenneth I. Wolpin. The paper presents a structural dynamic discrete-choice model of schooling, work, and occupational choice for young men using NLSY79 data.
- **Verdict:** OK
- **Note:** All bibliographic details (journal, volume, issue, pages, year, authors) match.

### Keane, M. P. and Wolpin, K. I. (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. *Review of Economics and Statistics*, 76(4), 648-672.

- **Located:** https://ideas.repec.org/p/fip/fedmsr/181.html (working paper; journal: Review of Economics and Statistics)
- **Tutorial claims:** This is the source of the sampled Emax approximation method: compute exact Emax on a sampled subset of states and use regression interpolation for the rest.
- **Source says:** Review of Economics and Statistics, vol. 76, no. 4, pp. 648-672, November 1994, by Michael P. Keane and Kenneth I. Wolpin. The paper proposes solving discrete choice dynamic programming models by Monte Carlo simulation at sampled state-space points and regression-based interpolation for un-evaluated points - precisely the method the README describes.
- **Verdict:** OK
- **Note:** All bibliographic details match. The tutorial's description of the method is an accurate summary of the paper's core contribution.

## Main Message Audit

> "Finite-horizon structural labor models are conceptually simple but expensive because schooling and experience create many continuation states. The Keane-Wolpin approximation keeps the dynamic logic intact while replacing most exact Emax evaluations with a fitted continuation-value surface. The tradeoff is visible: the method saves computation, but approximation error is largest where early human-capital choices have the most future consequences."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| "structurally simple but expensive because schooling and experience create many continuation states" | Model Setup reports 2,310 pre-terminal reachable states; state counts in the diagnostics table grow from 1 at age 16 to 525 at age 29 | OK |
| "approximation keeps the dynamic logic intact" | Solution Method shows the approximate recursion uses the same Bellman structure and transition $g(s,d)$ as the exact benchmark | OK |
| "replacing most exact Emax evaluations with a fitted continuation-value surface" | In this tutorial's calibration, ages 16-25 (10 of 14 periods) sample every reachable state exactly; only ages 26-29 skip any states. The diagnostics table shows 1,754 of 2,310 pre-terminal states are evaluated exactly - roughly 76%. The approximation replaces a minority (24%) of evaluations in this run. | OVERREACH |
| "the method saves computation" | Results table: approximation runtime 0.0811 s vs exact 0.085 s - the approximation is faster | OK |
| "approximation error is largest where early human-capital choices have the most future consequences" | Diagnostics table: normalized RMSE (RMSE / exact Emax sd) peaks at age 17 (0.123) and declines at older ages | OK |

Issues:
- "replacing most exact Emax evaluations": In this tutorial's small calibration (260 sample cap, 2,310 states), 76% of states are evaluated exactly and only ages 26-29 use any interpolation. The method is designed to replace most evaluations in large-scale estimation - but the tutorial's own results show the opposite proportion. The Overview's note that "exact backward induction is straightforward in this small version" partially flags this, but the Takeaway states the shortcut as an accomplished fact rather than a design goal.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $s_t = (E_t, X^b_t, X^w_t)$ | Equations, state definition | Yes | Defined immediately on introduction |
| $E_t$ | Equations, state definition | Yes | "completed schooling" |
| $X^b_t$, $X^w_t$ | Equations, state definition | Yes | "occupation-specific experience stocks" |
| $d_t$ | Equations, action set | Yes | Defined via the action-set equation |
| $D(s,t)$ | Equations, action set | Yes | Listed as subset of school/blue/white/home |
| $g(s,d)$ | Equations, transition | Yes | Defined via the four transition cases |
| $\alpha_t$ | Equations, payoff block | Yes | "age be $\alpha_t = 16 + t$" defined inline |
| $C(E)$ | Equations, payoff block | Yes | "college years be $C(E) = \max\lbrace E-12,0\rbrace$" defined inline |
| $\bar E$ | Equations, transition ($g(s,\mathrm{school})$) | Partial - late | First used in transition equation; defined in Model Setup table, more than 50 lines later |
| $w_b(s)$, $w_w(s)$ | Equations, wage equations | Yes | Defined via the log-wage expressions |
| $u_{\mathrm{school}}, u_{\mathrm{blue}}, u_{\mathrm{white}}, u_{\mathrm{home}}$ | Equations, payoff equations | Yes | Defined via explicit formulas |
| $\mathbb{E}_T(s)$ | Equations, terminal value | Yes | Defined as $4\max\lbrace u_\mathrm{blue}(s,T-1),\ldots\rbrace$ |
| $v_t(d,s)$ | Equations, choice-specific value | Yes | Defined as $u_d(s,t)+\beta\mathbb{E}_{t+1}(g(s,d))$ |
| $\beta$ | Equations, $v_t$ definition | Partial | Used in Equations; numeric value given in Model Setup table. Standard economics notation; audience would recognize it. |
| $\sigma_\epsilon$ | Equations, Emax formula | Yes | "taste shocks of scale $\sigma_\epsilon$" |
| $\gamma_E$ | Equations, Emax formula | Yes | "Here $\gamma_E$ is Euler's constant" |
| $\mathbb{E}_t(s)$ | Equations, Emax formula | Yes | Defined as the logit log-sum-exp expression |
| $P_t(d \mid s)$ | Equations, CCP formula | Yes | "logit conditional choice probabilities" |
| $S_t^{sample}$ | Equations, approximation | Yes | Defined as sampled state set |
| $M_t$ | Equations, $S_t^{sample}$ definition | Partial | Subscript in sample-set notation; numeric value in Model Setup table |
| $Y_{t,i}$ | Equations, regression | Yes | Defined as $\mathbb{E}_t(s_{t,i})$ |
| $\phi(s,t)$ | Equations, regression | Yes | Listed as 12-term polynomial |
| $b_t$ | Equations, regression | Yes | Regression coefficient (implicit from regression model) |
| $\eta_{t,i}$ | Equations, regression | Implicit | Regression error term; unnamed but clear from context |
| $\widehat{\mathbb{E}}_t(s)$ | Equations, fitted surface | Yes | Defined as $\phi(s,t)'\widehat b_t$ |
| $\widehat b_t$ | Equations, fitted surface | Yes | Defined fully in Solution Method via ridge formula |
| $Q_t(d,s)$ | Solution Method, exact recursion | Yes | Defined as $u_d(s,t)+\beta\mathbb{E}_{t+1}(g(s,d))$ - same formula as $v_t(d,s)$ in Equations |
| $Y_t$, $\Phi_t$ | Solution Method, regression | Yes | Defined contextually as stacked targets and feature matrix |
| $\lambda$ | Solution Method, ridge formula | Yes | Defined in Model Setup table as $10^{-6}$ |
| $I$ | Solution Method, ridge formula | Implicit | Identity matrix; universally standard |
| $E_0$ | Solution Method, forward pass | Partial | Implicit from initial-state tuple; Model Setup gives the value (10) |
| $N_s$ | Model Setup table | Yes | "Reachable state count in the exact benchmark" |

Flagged issues:
- **$v_t(d,s)$ vs $Q_t(d,s)$ - notation drift**: The Equations section defines $v_t(d,s) = u_d(s,t)+\beta\mathbb{E}_{t+1}(g(s,d))$ as the deterministic choice-specific value. The Solution Method introduces $Q_t(d,s)$ with the identical formula. These are the same object under two different names in consecutive sections; a reader looking for where $Q_t$ was defined will not find it in Equations.
- **$\bar E$ - late definition**: First appears in the transition equation $g(s,\mathrm{school}) = (\min\lbrace E+1,\bar E\rbrace, X^b, X^w)$ in Equations. Formally defined in the Model Setup table more than 50 lines later. Context makes the meaning inferrable (overbar convention for a maximum), but the definition is late by the repo's threshold.

## Summary

Both cited references are bibliographically correct and the tutorial accurately describes what each paper contributes. The main structural issue is a notation drift: the choice-specific value is called $v_t(d,s)$ in the Equations section and $Q_t(d,s)$ in the Solution Method, with identical defining formulas, giving a reader two names for one object and no cross-reference. A secondary issue is that the Takeaway's claim about "replacing most exact Emax evaluations" is an OVERREACH in the context of this tutorial's calibration, where 76% of states are evaluated exactly and only ages 26-29 use any interpolation at all. There is also a minor late-definition of $\bar E$. In total: 0 MAJOR paper issues, 0 MINOR paper issues, 0 NOT FOUND, 1 OVERREACH, 1 notation drift (the single most important fix: unify $v_t$ and $Q_t$ to one name throughout), 1 late-defined symbol.
