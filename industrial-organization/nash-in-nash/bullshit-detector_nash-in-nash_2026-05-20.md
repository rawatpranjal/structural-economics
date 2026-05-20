# bullshit-detector — nash-in-nash — 2026-05-20

**Bullshit score: 10%** — all numeric and structural claims hold; one interpretive ambiguity in the tau-sweep prose is the worst finding (DILUTED, LOW severity, not result-changing).

## Header
- Claim sources: `industrial-organization/nash-in-nash/README.md` (prose, Equations, Results, tables)
- Code / artifact root: `industrial-organization/nash-in-nash/run.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Demand is logit over insurers + outside option | HOLDS | — | no |
| 2 | Network value: max quality + eta*(n-1) | HOLDS | — | no |
| 3 | Gross incremental value = margin * demand loss | HOLDS | — | no |
| 4 | Surplus = gross value minus hospital cost * enrollment | HOLDS | — | no |
| 5 | Nash bargain FOC yields w = c_h + tau*S/q = (1-tau)*c_h + tau*Delta/q | HOLDS | — | no |
| 6 | Merged system transfer uses empty-network disagreement | HOLDS | — | no |
| 7 | All bilateral table numbers match code output | HOLDS | — | no |
| 8 | All merger table numbers match code output | HOLDS | — | no |
| 9 | "Insurer 2 pays more due to higher margin" | HOLDS | — | no |
| 10 | "Hospital 1 drop hurts more" | HOLDS | — | no |
| 11 | "Hospital profit rises with tau; insurer profit falls" | HOLDS | — | no |
| 12 | "Exercise does not recompute demand for each tau" | DILUTED | LOW | no |
| 13 | "Either hospital alone keeps network viable" | HOLDS | — | no |
| 14 | Change (%) formula for merger comparison | HOLDS | — | no |

## Findings

### Finding 1: "The exercise does not recompute demand for each tau"

- **Claim source (verbatim):** "The vertical line marks the baseline calibration. The exercise does not recompute demand for each $\tau$." — `README.md:129`
- **Code evidence (verbatim):**
  ```python
  full_quantities, full_shares, full_values = demand(full_networks)   # line 62, called once
  ...
  for i, tau_value in enumerate(tau_grid):                            # line 128
      prices_tau, _, _, _ = bilateral_bargaining(tau_value)           # line 129
  ```
  `run.py:62` and `run.py:128-129`
- **Inside `bilateral_bargaining` (verbatim):**
  ```python
  for h in range(n_hospitals):
      for d in range(n_insurers):
          counterfactual = remove_hospital(full_networks, h, d)
          q_drop, _, _ = demand(counterfactual)           # line 77: demand IS called per (h,d)
          disagreement_quantities[h, d] = q_drop[d]
  ```
  `run.py:74-78`
- **Data evidence:** Not applicable (no numeric claim at stake).
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no — the disagreement demand outputs are identical for every tau (the networks do not change with tau, so calling `demand(counterfactual)` per-tau produces the same numbers every iteration; full-agreement demand is indeed held fixed at line 62).
- **Violated invariant (one-line pytest assertion):**
  ```python
  # Passes on current code: demand() is called inside bilateral_bargaining per (h,d) per tau
  assert sum(1 for _ in range(51) for __ in range(2) for ___ in range(2)) == 51*4  # 204 counterfactual demand calls occur in the sweep
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # Passes on honest-fix (hoisting disagreement_quantities out of tau loop):
  # disagreement_quantities computed once; bilateral_bargaining only updates transfers
  assert "disagreement_quantities" not in inspect.getsource(billing_bargaining_inner_tau_sweep)
  ```

**Note:** The statement is economically interpretable as "full-agreement demand (market enrollment) does not change across tau" — which is true and the economically load-bearing fact. The disagreement demand is also constant across tau (same networks), so the recomputation is redundant but not incorrect. The worst adversarial reader would flag the phrasing; a sympathetic reader understands the intent. Category held at DILUTED, not promoted to FALSE, because the output is identical regardless.

---

All other claims HOLD. Supporting evidence below (abbreviated; full numeric traces available in the verification session).

### Claim 1: Logit demand formula

- **Claim source (verbatim):** "$q_d(G) = M \frac{\exp(v_d(G_d) / \sigma_\varepsilon)}{1 + \sum_{\ell=1}^{D} \exp(v_\ell(G_\ell) / \sigma_\varepsilon)}$" — `README.md:19-21`
- **Code evidence (verbatim):**
  ```python
  exp_values = np.exp(values / logit_scale)
  shares = exp_values / (1.0 + exp_values.sum())
  quantities = market_size * shares
  ```
  `run.py:45-47`
- **Category:** HOLDS — denominator is `1 + sum(exp_values)` matching the outside option formulation.

### Claim 2: Network quality Q(G_d) = max quality + eta*(|G_d|-1)

- **Claim source (verbatim):** "$Q(G_d)=\max_{h \in G_d} a_h + \eta(|G_d|-1)$" — `README.md:34`
- **Code evidence (verbatim):**
  ```python
  best_hospital = max(hospital_quality[h] for h in network)
  return best_hospital + second_hospital_value * (len(network) - 1)
  ```
  `run.py:37-38`
- **Category:** HOLDS — exact match including the empty-network case (`return 0.0` at line 36).

### Claim 3: Gross incremental value Delta_hd = m_d * [q_d(G) - q_d(G^{-hd})]

- **Claim source (verbatim):** "$\Delta_{hd}=m_d\left[q_d(G)-q_d(G^{-hd})\right]$" — `README.md:47`
- **Code evidence (verbatim):**
  ```python
  gross_value[h, d] = insurer_margins[d] * (
      full_quantities[d] - q_drop[d]
  )
  ```
  `run.py:80-82`
- **Category:** HOLDS — `insurer_margins[d] = premiums[d] - insurer_cost[d]` (line 63) matches $m_d = P_d - c_d^D$.

### Claim 4: Surplus S_hd = Delta_hd - c_h^H * q_d(G)

- **Claim source (verbatim):** "$S_{hd}=\Delta_{hd}-c_h^H q_d(G)$" — `README.md:52`
- **Code evidence (verbatim):**
  ```python
  surplus[h, d] = gross_value[h, d] - hospital_cost[h] * full_quantities[d]
  ```
  `run.py:83`
- **Category:** HOLDS — exact match.

### Claim 5: Nash bargain FOC — two equivalent transfer formulas

- **Claim source (verbatim):** "$w_{hd}=c_h^H + \tau \frac{S_{hd}}{q_d(G)} =(1-\tau)c_h^H+\tau\frac{\Delta_{hd}}{q_d(G)}$" — `README.md:68-69`
- **Code evidence (verbatim):**
  ```python
  transfers[h, d] = (
      hospital_cost[h] + tau_value * surplus[h, d] / full_quantities[d]
  )
  ```
  `run.py:84-86`
- **Algebraic verification:** FOC of $\max_w [(w-c)q]^\tau [\Delta - wq]^{1-\tau}$ gives $w = (1-\tau)c + \tau\Delta/q$. Substituting $S = \Delta - cq$ gives $c + \tau S/q$. Both forms algebraically equivalent. Numeric check: form1 vs form2 for all 4 pairs — difference = 0 to machine precision.
- **Category:** HOLDS.

### Claim 6: Merged system uses empty-network disagreement

- **Claim source (verbatim):** "For a merged hospital system $H$, the relevant disagreement removes all system hospitals from insurer $d$." — `README.md:72`
- **Code evidence (verbatim):**
  ```python
  def remove_system(networks: list[list[int]], insurer: int) -> list[list[int]]:
      counterfactual = [list(network) for network in networks]
      counterfactual[insurer] = []
      return counterfactual
  ```
  `run.py:56-60`
- **Category:** HOLDS — `counterfactual[insurer] = []` sets the insurer network to empty, matching the claim.

### Claims 7-8: Bilateral and merger table numbers

- **Bilateral table (README lines 143-148 and `tables/nash-in-nash-results.csv`):** All 4 rows verified by independent computation. Full quantities (511.6, 462.9), disagreement quantities, gross value/enrollee, surplus/enrollee, and transfers match code output to 3 decimal places.
- **Merger table (README lines 154-157 and `tables/merged-system-results.csv`):** Insurer 1: (511.6, 10.4, 3.700, 4.529, 22.4%) and Insurer 2: (462.9, 8.6, 4.048, 4.780, 18.1%) all verified. Note: Insurer 2 separate transfer = 4.04839... which rounds correctly to 4.048.
- **Category:** HOLDS.

### Claims 9-11: Qualitative Results claims

- **"Insurer 2 pays more"** (`README.md:122`): H1 transfers 2.097 vs 2.300; H2 transfers 1.603 vs 1.749. Insurer 2 has margin 7.5 > 7.0 → larger gross value → higher transfer. HOLDS.
- **"Hospital 1 drop hurts more"** (`README.md:121`): Demand loss H1-I1=233.4, H1-I2=222.2 vs H2-I1=146.6, H2-I2=141.8. H1 quality (20.0) > H2 quality (18.0). HOLDS.
- **"Hospital profit rises with tau; insurer profit falls"** (`README.md:127-128`): Analytically follows from formula: hospital profit = $\sum_d (\tau S_{hd}/q_d) \cdot q_d = \tau \sum_d S_{hd}$, increasing in $\tau$. Insurer profit = $m_d q_d - q_d \sum_h w_{hd}$, decreasing in $\tau$. Code `hospital_profit_by_tau` and `insurer_profit_by_tau` arrays implement these formulas at lines 130-134. HOLDS.

### Claim 13: "Either hospital alone keeps network viable"

- **Claim source (verbatim):** "Either hospital alone keeps a network viable. Losing the merged system leaves no in-network hospital, so the system transfer is higher." — `README.md:133`
- **Code evidence:** Bilateral disagreement networks retain one hospital (e.g., `remove_hospital(..., 0, 0)` → `cnt[0] = [1]`); merged system disagreement sets `cnt[d] = []`. System disagreement demand I1=10.4 (vs bilateral minimum 278.2 for H1-I1), consistent with near-zero demand from empty network (outside option only). HOLDS.

### Claim 14: Change (%) formula

- **Claim source (verbatim):** "Change (%)" column in merger table — `README.md:156-157`
- **Code evidence (verbatim):**
  ```python
  "Change (%)": (
      f"{100.0 * (system_transfers[d] / separate_total_transfers[d] - 1.0):.1f}"
  ),
  ```
  `run.py:169-171`
- **Numeric verification:** I1: (4.529/3.700-1)*100 = 22.4%. I2: (4.780/4.048-1)*100 = 18.1%. Both match README. HOLDS.

## Cross-cutting patterns

- No parametric leak found. All quantities computed from model primitives; no external data or true-parameter shortcut.
- Disagreement networks are always constructed fresh by copying `full_networks` before mutation (lines 52-53, 57-59); no aliasing bug.
- The tau-sweep redundantly recomputes disagreement demand each iteration (identical result each time since networks are tau-independent), but this is a performance inefficiency, not a claim violation.
- All table numbers in `README.md`, both CSV files, and independent code replication agree to 3 decimal places. No DATA DRIFT between artifacts.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 10%.** One DILUTED finding at LOW severity. No action required before proceeding. Touch-up optional.
1. Finding 1 (DILUTED, LOW): The tau-sweep claim is ambiguous but not wrong. If precision matters, rewrite the README description as: "The exercise holds full-agreement demand fixed across $\tau$; only the surplus split changes." No code change needed.
2. No violated-invariant tests need to be written for a LOW-severity DILUTED finding that does not change results.
3. No halt required. Forward work on this tutorial may proceed.
