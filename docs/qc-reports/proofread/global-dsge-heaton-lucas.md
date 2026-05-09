# Proofread: global-dsge/heaton-lucas/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T06:20:00Z._

## Paper / Source Verification

### Heaton, J. & Lucas, D. (1996). *JPE* 104(3), 443-487.

- **Located:** https://ideas.repec.org/a/ucp/jpolec/v104y1996i3p443-87.html
- **Tutorial claims:** Two CRRA agents trade equity and a one-period bond with short-sale and borrowing limits, generating state-dependent equity premia that depend on agent 1's wealth share.
- **Source says:** "Evaluating the Effects of Incomplete Markets on Risk Sharing and Asset Pricing," JPE 104(3), 443-487. The paper examines incomplete-markets economies where heterogeneous agents trade equity and bonds subject to portfolio constraints, generating sizable equity premia.
- **Verdict:** OK
- **Note:** Volume 104, issue 3, pages 443-487 confirmed via REPEC and journal records.

### Cao, D., Luo, W. & Nie, G. (2023). *RED* 51, 199-225.

- **Located:** https://www.sciencedirect.com/science/article/abs/pii/S1094202523000017
- **Tutorial claims:** Source of the STPFI algorithm used to solve the wealth-share economy; the HL1996.gmod example from the companion toolbox is the direct basis for the code.
- **Source says:** "Global DSGE Models," Review of Economic Dynamics 51, 199-225 (2023). Introduces Simultaneous Transition and Policy Function Iterations (STPFI) and uses the Heaton-Lucas model as a leading example application.
- **Verdict:** OK
- **Note:** Volume 51, pages 199-225 confirmed; STPFI attribution and Heaton-Lucas example confirmed via publisher page and GDSGE toolbox documentation.

## Main Message Audit

> "Limited asset trade turns the wealth distribution into an asset-pricing state. Risk premia move because constrained households cannot freely trade away high marginal-utility states. STPFI fits the model because it solves the transition and portfolio constraints inside one fixed point."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| "Limited asset trade turns the wealth distribution into an asset-pricing state" | Equations (budget constraint, market clearing, consistency equation); equity price written as $p_s(z',\omega_1')$ making the dependence explicit | OK |
| "Risk premia move because constrained households cannot freely trade away high marginal-utility states" | Equations (KT multipliers $\mu_i^s$, $\mu_i^b$); Results (equity premium ranges 0.43% to 1.42% across wealth-share states; multiplier figure shows constraint activity) | OK |
| "STPFI fits the model because it solves the transition and portfolio constraints inside one fixed point" | Solution Method (pseudocode solves consistency equations and complementary-slackness conditions jointly; fixed-point damping loop described) | OK |

Issues: None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $z_t$ | Equations, sentence 1 | Yes - "Let $z_t\in\{1,\ldots,8\}$ be the Markov shock" | OK |
| $z$ | Equations, sentence 1 | Yes - as current-period realization of $z_t$ | OK |
| $z'$ | Equations, KT conditions | Partial - prime for next period is standard; no explicit statement | Acceptable |
| $g_z$ | Equations, sentence 1 | Yes - "aggregate growth is $g_z$" | OK |
| $d_z$ | Equations, sentence 1 | Yes - "equity dividend is $d_z$" | OK |
| $\eta_{1z}$ | Equations, sentence 1 | Yes - "agent 1 receives endowment share $\eta_{1z}$" | OK |
| $\eta_{2z}$ | Equations, sentence 1 | Yes - "$\eta_{2z}=1-\eta_{1z}$" | OK |
| $\gamma$ | Equations, sentence 2 | Yes - "CRRA utility with risk aversion $\gamma$" | OK |
| $c_i$ | Equations, sentence 2 | Yes - "chooses consumption $c_i$" | OK |
| $s_i'$ | Equations, sentence 2 | Yes - "next-period equity holdings $s_i'$" | OK |
| $b_i'$ | Equations, sentence 2 | Yes - "next-period bond holdings $b_i'$" | OK |
| $p_s$ | Equations, sentence 3 | Yes - "equity... prices are $p_s$" | OK |
| $p_b$ | Equations, sentence 3 | Yes - "bond prices are... $p_b$" | OK |
| $\omega_1$ | Overview, sentence 4 | Yes - "agent 1's wealth share, $\omega_1$" | OK |
| $\omega_i$ | Equations, budget constraint | Partial - $\omega_1$ defined in Overview; $\omega_2=1-\omega_1$ stated inline | Acceptable |
| $\beta$ | Equations, KT conditions | Late - defined in Model Setup table; within 50 lines of first use | Acceptable |
| $\bar{K}^b$ | Equations, portfolio constraints | Late - defined in Model Setup table; within 50 lines of first use | Acceptable |
| $E_z$ | Equations, KT conditions | No - conditional expectation operator never defined in README | Flag |
| $\mu_i^s$ | Equations, KT conditions | Partial - context ("Kuhn-Tucker conditions") makes role clear | Acceptable |
| $\mu_i^b$ | Equations, KT conditions | Partial - same as $\mu_i^s$ | Acceptable |
| $c_i'$ | Equations, KT conditions | Partial - prime for next period is standard | Acceptable |
| $p_s(z',\omega_1')$ | Equations, KT conditions | Yes - equity price function arguments follow from earlier definitions | OK |
| $\omega_1'(z')$ | Equations, consistency equation | Partial - prime and shock-indexing follow from earlier conventions | Acceptable |

Flagged issues:
- $E_z$ - the conditional expectation operator appears in both Euler conditions without being defined. The subscript $z$ signals conditioning on the current shock, but the README never states this. The audience would likely infer the meaning, but the symbol is technically undefined within the README.

## Summary

Both references verify correctly: Heaton and Lucas (1996) is confirmed at JPE 104(3), 443-487 with content matching the tutorial's description of incomplete markets with heterogeneous agents, portfolio constraints, and state-dependent equity premia; Cao, Luo, and Nie (2023) is confirmed at RED 51, 199-225 as the source of the STPFI algorithm with the Heaton-Lucas model as an explicit example. The three takeaway clauses are each supported by the Equations, Solution Method, and Results sections without overreach. One notation issue is flagged: the conditional expectation operator $E_z$ appears in both Euler conditions without definition - a one-line definition such as "where $E_z[\cdot]$ denotes the expectation conditional on current shock $z$" placed before the first Euler equation would resolve it. Total issues: 0 MAJOR, 0 MINOR, 0 NOT FOUND, 0 OVERREACH, 1 undefined symbol.
