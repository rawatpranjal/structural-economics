# Proofread: industrial-organization/dynamic-discrete-choice/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T04:15:00Z._

## Paper / Source Verification

### Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999-1033.

- **Located:** https://doi.org/10.2307/1911259
- **Tutorial claims:** Foundational reference for the bus engine replacement model, with mileage as the state variable and Harold Zurcher as the empirical subject.
- **Source says:** Exact title, journal (Econometrica), volume 55, issue 5, year 1987, and page range 999-1033 all confirmed via IDEAS/RePEC and the Econometric Society record.
- **Verdict:** OK
- **Note:** Citation is exact.

### Hotz, V. J. and Miller, R. A. (1993). Conditional Choice Probabilities and the Estimation of Dynamic Models. *Review of Economic Studies*, 60(3), 497-529.

- **Located:** https://doi.org/10.2307/2298122
- **Tutorial claims:** Source for the CCP estimator that replaces the inner Bellman solve with a first-stage policy estimate and a linear ex ante value system.
- **Source says:** Exact title, journal (Review of Economic Studies), volume 60, issue 3, year 1993, and page range 497-529 all confirmed via Oxford Academic and EconPapers.
- **Verdict:** OK
- **Note:** Citation is exact.

## Main Message Audit

> "Dynamic discrete choice turns observed hazards into statements about current payoffs and continuation values. In this replacement problem, mileage matters today and tomorrow. NFXP solves the Bellman fixed point inside the likelihood. CCP estimation uses a first-stage hazard before the structural step. MPEC estimates parameters and values while enforcing Bellman equations."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Dynamic discrete choice turns observed hazards into statements about current payoffs and continuation values | Equations: $P_\theta(1 \mid x)$ is derived from value functions that decompose into flow payoffs and continuation values | OK |
| In this replacement problem, mileage matters today and tomorrow | Equations: flow utility $u(x,0) = \theta_0 + \theta_1 x$ (today) and continuation value $\beta \sum_{x'} F_a(x' \mid x)[\ldots]$ (tomorrow) | OK |
| NFXP solves the Bellman fixed point inside the likelihood | Solution Method: nested fixed-point algorithm pseudocode iterates to convergence for every candidate $\theta$ | OK |
| CCP estimation uses a first-stage hazard before the structural step | Solution Method: Hotz-Miller CCP pseudocode fits $\hat p(x)$ in a first stage, then optimizes over $\theta$ | OK |
| MPEC estimates parameters and values while enforcing Bellman equations | Solution Method: MPEC pseudocode lists $\theta$ and $v$ as joint decision variables with Bellman residuals as equality constraints | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $x_t$ | Equations, first line | Yes - "Let $x_t \in X$ denote mileage at the start of period $t$" | |
| $X$ | Equations, first line | Partial - introduced via $x_t \in X$ without stating $X$ explicitly; Model Setup gives the grid range $[0,15]$ roughly 71 lines later | Below the 50-line window but clear from context |
| $a_t$ | Equations | Yes - "The action $a_t=1$ replaces the engine and $a_t=0$ keeps it" | |
| $u(x,a)$ | Equations | Yes - defined by the two equations that immediately follow | |
| $\theta_0, \theta_1$ | Equations | Yes - appear in $u(x,0) = \theta_0 + \theta_1 x$ | |
| $F_a(x' \mid x)$ | Equations | Yes - "The transition matrix $F_a(x' \mid x)$ gives next period's mileage" | |
| $F_1, F_0$ | Equations | Yes - defined as replacement and keep transitions in the following sentence | |
| $v_a(x)$ | Equations | Yes - "the conditional value functions satisfy [equation]" | |
| $\beta$ | Equations (value function) | Acceptable - defined in Model Setup table ("Discount factor \| 0.9") roughly 53 lines after first use; just outside the 50-line window but unambiguous | |
| $\gamma$ | Equations | Yes - "where $\gamma$ is Euler's constant" | |
| $P_\theta(1 \mid x)$ | Equations | Yes - "The replacement probability is [equation]" | |
| $d_{it}$ | Equations (likelihood) | Yes - "where $d_{it}=1$ means replacement" | |
| $\ell(\theta)$ | Equations (likelihood) | Yes - "the full-solution likelihood is" | Argument is $\theta$ here |
| $\hat p(x)$ | Equations (CCP section) | Yes - "a first-stage estimate $\hat p(x)$ of $\Pr(a=1 \mid x)$" | |
| $\hat F(x' \mid x)$ | Equations (CCP section) | Yes - "form the policy transition [equation]" | |
| $W_\theta$ | Equations (CCP section) | Yes - "the Hotz-Miller ex ante value solves the linear system" | |
| $\bar u_\theta(\hat p)$ | Equations (CCP section) | Partial - "includes the keep payoff and the logit entropy terms implied by $\hat p$"; code confirms the exact formula | Prose description sufficient |
| $P_\theta^{HM}(1 \mid x)$ | Equations (CCP section) | Yes - "The model-implied replacement probability is then" | |
| $\Lambda(z)$ | Equations (CCP section) | Yes - "with $\Lambda(z)=1/(1+\exp(-z))$" | |
| $\ell(v)$ | Equations (MPEC section) | Drift - earlier defined as $\ell(\theta)$; in MPEC the same symbol appears with $v$ as the argument | See flagged issues |
| $j$ | Equations (MPEC constraint) | Implicit - summation index $j \in \{0,1\}$ is self-describing | |

Flagged issues:
- **$\ell(\theta)$ vs $\ell(v)$ notation drift.** The full-solution likelihood is introduced as $\ell(\theta)$ in the NFXP block, but the MPEC objective is written $\max_{\theta,v} \ell(v)$ with $v$ as the argument. The prose note "The likelihood still uses the logit choice formula" explains the switch, but the argument type changes without explicit acknowledgment. A reader may not immediately see that $\ell(v)$ and $\ell(\theta)$ are the same function under different parameterizations. The consistent fix is to write the MPEC objective as $\max_{\theta,v} \ell(\theta, v)$ or add a brief inline note that the argument changes because $v$ is now a free variable.

## Summary

Both cited references are bibliographically exact (title, authors, journal, volume, issue, year, page range all confirmed). The main message is fully supported by the Equations and Solution Method sections with no overreach or unsupported claims. The only flagged notation issue is the drift of $\ell(\theta)$ (NFXP likelihood) to $\ell(v)$ (MPEC objective) without an explicit bridging statement; this is 1 MINOR notation drift, 0 MAJOR issues, 0 NOT FOUND references, and 0 OVERREACH clauses. The single most important fix is to reconcile the likelihood argument in the MPEC equation so readers see clearly that the argument type changes from $\theta$ to $v$ because the Bellman fixed point is moved into the constraints.
