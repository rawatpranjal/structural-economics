# Proofread: dynamic-programming/shock-discretization/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T17:10:00Z._

## Paper / Source Verification

### Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*, 20(2), 177-181.

- **Located:** https://ideas.repec.org/a/eee/ecolet/v20y1986i2p177-181.html
- **Tutorial claims:** The paper introduces the method of approximating a continuous AR(1) with a finite-state Markov chain via conditional Gaussian mass integration on an evenly spaced grid.
- **Source says:** Develops a procedure for approximating continuous-valued univariate and vector AR processes with discrete finite-state Markov chains by matching conditional normal moments on a discrete grid. Journal, volume, issue, pages, and year all confirmed.
- **Verdict:** OK
- **Note:** DOI `10.1016/0165-1765(86)90168-0` is consistent with the Elsevier Economics Letters identifier confirmed by IDEAS/RePEC.

---

### Rouwenhorst, K. G. (1995). Asset Pricing Implications of Equilibrium Business Cycle Models. In T. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.

- **Located:** https://www.degruyterbrill.com/document/doi/10.1515/9780691218052-014/html
- **Tutorial claims:** The chapter by Rouwenhorst introduces a recursive discretization method that matches ρ and σ_z² by construction for any N ≥ 2.
- **Source says:** Chapter 10 (pp. 294-330) of Frontiers of Business Cycle Research, edited by Thomas F. Cooley, Princeton University Press, 1995. The Rouwenhorst discretization method is embedded in the chapter; the main focus is asset pricing in equilibrium business cycle models. Author is confirmed as K. Geert Rouwenhorst.
- **Verdict:** OK
- **Note:** The DOI resolves to the 2020 Princeton digital edition (ISBN 9780691218052), but the original 1995 publication details are correct as cited.

---

### Kopecky, K. A. and Suen, R. M. H. (2010). Finite State Markov-Chain Approximations to Highly Persistent Processes. *Review of Economic Dynamics*, 13(3), 701-714.

- **Located:** https://ideas.repec.org/a/red/issued/09-115.html
- **Tutorial claims:** Implicitly cited as the source establishing the superiority of the Rouwenhorst method over Tauchen for highly persistent processes; journal, volume, issue, pages, and year are given as RED 13(3), 701-714, 2010.
- **Source says:** Journal, volume, issue, pages, and year all match. The paper proves analytically that Rouwenhorst exactly matches conditional and unconditional mean and variance and the first-order autocorrelation for any stationary AR(1). Correct DOI is `10.1016/j.red.2010.02.002`.
- **Verdict:** MINOR
- **Note:** The cited DOI `10.1016/j.red.2009.07.002` is wrong - it resolves to an unrelated article ("Fiscal policy and default risk in emerging markets"). The correct DOI is `10.1016/j.red.2010.02.002`.

---

## Main Message Audit

> "Discretization is part of the economic model. With persistent shocks and small N, Rouwenhorst is the safer default. It matches σ_z and ρ by construction. Tauchen is transparent and can approximate the Gaussian shape well on finer grids. At ρ=0.95 and N=7, Tauchen overstates persistence enough to change continuation values. Choose the chain by the moments that matter in the Bellman equation."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Discretization is part of the economic model | Equations: P enters the Bellman directly as the transition probability governing continuation values | OK |
| With persistent shocks and small N, Rouwenhorst is the safer default | Results: moment-accuracy table shows Tauchen persistence error of 0.050 at N=3 vs Rouwenhorst at 0 | OK |
| It matches σ_z and ρ by construction | Solution Method: "By construction, the chain matches ρ and σ_z² for any N ≥ 2. It has no quadrature error in those moments." | OK |
| Tauchen is transparent and can approximate the Gaussian shape well on finer grids | Results: moment-accuracy table shows Tauchen errors shrinking toward zero as N grows to 15 | OK |
| At ρ=0.95 and N=7, Tauchen overstates persistence enough to **change continuation values** | Not demonstrated - the tutorial never solves a Bellman equation; only the chain moments are compared | OVERREACH |
| Choose the chain by the moments that matter in the Bellman equation | Equations: Bellman structure shown; Results: moment diagnostics framed around P entering continuation sums | OK |

Issues:
- **OVERREACH - "change continuation values":** The tutorial shows that Tauchen at N=7 overstates persistence by 0.0122 (0.9622 vs 0.95). It does not solve any Bellman equation or tabulate value functions under the two chains. The claim that this error "changes continuation values" is economically plausible but not demonstrated by the tutorial's own results. The stronger wording would be justified only if a companion figure or table compared V(a,z_i) under each chain.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| a | Equations ("A household with assets a") | Yes | Assets |
| z_i | Equations ("faces shock z_i") | Yes | Shock state i |
| a' | Equations ("It chooses next assets a'") | Yes | Next-period assets |
| 𝒜 | Equations (Bellman: "a' ∈ 𝒜") | Implicit | Feasible asset set; used in max operator but never described |
| u(·) | Equations (Bellman) | No | Utility function; only its argument is shown; no form specified |
| R | Equations (Bellman: "u(Ra+exp(z_i)−a')") | No | Appears as a multiplicative factor on a; never named or defined |
| β | Equations (Bellman: "β Σ P_{ij} V(a',z_j)") | No | Discount factor; used but never introduced or named |
| P_{ij} | Equations | Yes | "P_{ij}=Pr(z_{t+1}=z_j\|z_t=z_i)" |
| N | Equations ("grid {z_1,…,z_N}") | Yes | Number of states |
| V(a,z_i) | Equations (Bellman left side) | Implicit | Self-defined as the value function by the Bellman equation |
| ρ | Overview/Equations ("persistence ρ") | Yes | AR(1) persistence parameter |
| σ_ε | Overview/Equations ("innovation scale σ_ε") | Yes | Innovation standard deviation |
| ε_{t+1} | Equations ("ε_{t+1} ∼ 𝒩(0,1)") | Yes | Standard normal innovation |
| σ_z² | Equations ("σ_z²=σ_ε²/(1−ρ²)") | Yes | Unconditional variance |
| σ_z | Equations | Yes | Unconditional standard deviation |
| ρ_k | Equations ("ρ_k ≡ Corr(z_t,z_{t+k})=ρ^k") | Yes | Lag-k autocorrelation; defined at first use |
| π | Equations ("invariant distribution π satisfying π=πP") | Yes | Stationary distribution vector |
| x_{t+1} | Equations (conditional expectation: "E[V(x_{t+1},z_{t+1})\|z_t=z_i]") | No | Appears only in this formula; appears to generalize a' from the Bellman, but the connection is not stated; notation shifts from a/a' used elsewhere |
| m | Model Setup (parameter table) | Yes | Tauchen half-width in units of σ_z |
| T_sim | Model Setup (parameter table) | Yes | Simulation horizon |
| Φ | Algorithm 1 pseudocode ("P[i,j]=Phi(…)−Phi(…)") | No | Standard normal CDF; conventional but never introduced in the text |
| p | Solution Method ("The base uses p=(1+ρ)/2") | Yes | Rouwenhorst base probability |
| P_N, P_2, P_n | Algorithm 2 pseudocode | Contextual | Defined within the algorithm block |

Flagged issues:
- **R - undefined:** The gross return factor R in u(Ra+exp(z_i)−a') is never introduced. Readers must infer it from context.
- **β - undefined:** The discount factor β appears in the Bellman without a name, definition, or value in the parameter table.
- **u(·) - partially specified:** The utility function is referenced by name only through its argument; no functional form (CRRA, log) is given, nor is one needed for a shock-discretization tutorial, but flagging for completeness.
- **x_{t+1} - undefined and inconsistent:** The conditional expectation formula uses x_{t+1} as the first argument of V, but the Bellman equation uses a and a'. The variable x is never introduced; this is a notation shift that goes unexplained.
- **Φ - undefined in text:** The standard normal CDF is used directly in Algorithm 1 pseudocode without being named in the surrounding prose.

---

## Summary

The tutorial is clean and self-consistent. The paper verification found one substantive issue: the Kopecky-Suen (2010) DOI is wrong (`10.1016/j.red.2009.07.002` resolves to an unrelated paper; the correct DOI is `10.1016/j.red.2010.02.002`). The other two citations are fully correct. The main message audit found one OVERREACH: the Takeaway claims that Tauchen's N=7 persistence error "changes continuation values," but the tutorial never solves a Bellman equation to demonstrate this - it remains a plausible but undemonstrated consequence. The notation audit found five issues, the most important being that R (the return factor) and β (the discount factor) appear in the Bellman equation without definition anywhere in the README, and x_{t+1} appears in the conditional expectation formula as a notation shift from a/a' without explanation. Overall: **1 MINOR citation issue, 1 OVERREACH, 0 MAJOR, 0 NOT FOUND; 5 notation gaps.** The single most important fix is correcting the Kopecky-Suen DOI to `10.1016/j.red.2010.02.002`.
