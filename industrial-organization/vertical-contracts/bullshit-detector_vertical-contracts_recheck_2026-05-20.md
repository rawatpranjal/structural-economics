# bullshit-detector — vertical-contracts — recheck — 2026-05-20

**Bullshit score: 0%** — Both prior findings resolved: F1 (margin floor ε=0.05 now documented in Equations with exact value and vacuousness note) and F2 (τ=4 now disclosed inline in Equations and in Model Setup table). All numeric claims verified against code and CSV. No new findings.

## Header
- Claim sources: `industrial-organization/vertical-contracts/README.md` (all sections)
- Code / artifact root: `industrial-organization/vertical-contracts/run.py`
- Data artifacts: `industrial-organization/vertical-contracts/tables/contract-outcomes.csv`
- Seed audit: `bullshit-detector_vertical-contracts_2026-05-20.md` (score 25%, 2 findings)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| F1 (prior) | Price floor ε=0.05 undocumented → now documented | HOLDS | — | — |
| F2 (prior) | τ value undisclosed → now τ=4 in Equations and Model Setup | HOLDS | — | — |
| 1 | Price formula p*(w) = max{(a+bw)/(2b), w+ε}, ε=0.05 | HOLDS | — | — |
| 2 | Floor never binds under committed calibration | HOLDS | — | — |
| 3 | τ=4 disclosed (Equations and Model Setup) | HOLDS | — | — |
| 4 | 792 feasible assortments | HOLDS | — | — |
| 5 | Wholesale only: 4 Mars slots, Fixed fees=0 | HOLDS | — | — |
| 6 | All-unit discount: 5 Mars slots, Fixed fees=0 | HOLDS | — | — |
| 7 | Slotting fees: 5 Mars slots, Fixed fees=6.20 | HOLDS | — | — |
| 8 | Slotting fee amounts: 1.10 Mars / 0.35 rivals | HOLDS | — | — |
| 9 | μ=0.42, d=0.18 | HOLDS | — | — |
| 10 | All 21 numeric cells in contract-outcomes table | HOLDS | — | — |
| 11 | Upstream profit subtracts fixed transfers | HOLDS | — | — |

## Findings

### Prior Finding 1 — RESOLVED: margin floor now documented in Equations

- **Prior claim (buggy):** `retail_outcome` applied `max(..., wholesale + 0.05)` but Equations section only showed the interior formula. Reader could not reproduce prices from equations alone.
- **Current code evidence (verbatim):**
  ```python
  def retail_outcome(intercept: float, wholesale: float, slope: float = 4.0) -> tuple[float, float]:
      """Interior monopoly price under linear demand, with a small margin floor."""
      price = max((intercept + slope * wholesale) / (2 * slope), wholesale + 0.05)
  ```
  `run.py:33-35`
- **Current README evidence (verbatim):** "For an interior product, the unconstrained optimum is $(a_j+bw_j)/(2b)$. The retailer never prices below a small markup over wholesale, so the price used is $p_j^{*}(w_j)=\max\lbrace (a_j+bw_j)/(2b),\ w_j+\epsilon\rbrace, \quad \epsilon=0.05.$ The margin floor $\epsilon=0.05$ is a numerical guard; it never binds under the calibration in this tutorial." — `README.md:28-38`
- **Verification:** Exhaustive check across all 12 products × 3 contracts confirms floor never binds. For all products and contracts, `(intercept + 4.0 * wholesale) / 8.0 > wholesale + 0.05`. The vacuousness claim holds. **HOLDS.**
- **Category:** HOLDS

---

### Prior Finding 2 — RESOLVED: τ=4 disclosed in Equations and Model Setup

- **Prior claim (buggy):** τ introduced as a free symbol in the equation, value τ=4 never stated in the README. Results reproducible only by inspecting the code default.
- **Current README evidence (verbatim):** "The discount applies only if the assortment contains at least $\tau$ Mars products, with $\tau=4$ in this tutorial:" — `README.md:59-60`. Also: "Mars margin falls by $d=0.18$ once the shelf holds at least $\tau=4$ Mars products" — `README.md:81`.
- **Code evidence (verbatim):**
  ```python
  def evaluate_subset(products: pd.DataFrame, subset: tuple[int, ...], contract: str, threshold: int = 4) -> dict[str, float]:
  ```
  `run.py:40` — default `threshold=4` matches disclosed τ=4.
- **Category:** HOLDS

---

### Finding 1: Price formula with floor — HOLDS

- **Claim source (verbatim):** "$p_j^{*}(w_j)=\max\lbrace (a_j+bw_j)/(2b),\ w_j+\epsilon\rbrace, \quad \epsilon=0.05.$" — `README.md:33-35`
- **Code evidence:** `run.py:35`: `price = max((intercept + slope * wholesale) / (2 * slope), wholesale + 0.05)`. Exact match. ✓
- **Category:** HOLDS

---

### Finding 2: Floor never binds — HOLDS

- **Claim source (verbatim):** "The margin floor $\epsilon=0.05$ is a numerical guard; it never binds under the calibration in this tutorial." — `README.md:37-38`
- **Verification:** Exhaustive product × contract check confirmed zero floor activations. Minimum interior margin: `(8.8 + 4.0 * (0.50 + 0.42)) / 8 = (8.8 + 3.68) / 8 = 1.56`; corresponding floor: `0.50 + 0.42 + 0.05 = 0.97`. Interior > floor for all cases. ✓
- **Category:** HOLDS

---

### Finding 3: τ=4 disclosed — HOLDS

- **Claim source (verbatim):** "with $\tau=4$ in this tutorial" — `README.md:60`; "at least $\tau=4$ Mars products" — `README.md:81`
- **Code evidence:** `run.py:40`: `threshold: int = 4`; `run.py:49`: `mars_count >= threshold`. ✓
- **Category:** HOLDS

---

### Finding 4: 792 feasible assortments — HOLDS

- **Claim source (verbatim):** "there are 792 feasible assortments" — `README.md:87-88`
- **Verification:** `C(12,7) = 792` confirmed by running `sum(1 for _ in itertools.combinations(range(12), 7))`. ✓
- **Category:** HOLDS

---

### Finding 5-7: Mars slots and fixed fees by contract — HOLDS

- **Claim source (verbatim):** Wholesale only = 4 Mars slots, Fixed fees = 0 (`README.md:122`); All-unit discount = 5 Mars slots, Fixed fees = 0 (`README.md:123`); Slotting fees = 5 Mars slots, Fixed fees = 6.20 (`README.md:124`)
- **Data evidence:** `tables/contract-outcomes.csv:2-4`: values match exactly (Mars slots 4/5/5; Fixed fees 0.00/0.00/6.20). Re-run from code confirms these values. ✓
- **Category:** HOLDS

---

### Finding 8: Slotting fee amounts — HOLDS

- **Claim source (verbatim):** "Fixed payments of 1.10 for Mars products and 0.35 for rival products" — `README.md:82`
- **Code evidence:** `run.py:51-52`: `fixed_fee = 1.10 if is_mars else 0.35`. ✓
- **Category:** HOLDS

---

### Finding 9: μ=0.42, d=0.18 — HOLDS

- **Claim source (verbatim):** "Per-unit margin $\mu=0.42$" — `README.md:81`; "Mars margin falls by $d=0.18$" — `README.md:81`
- **Code evidence:** `run.py:47`: `wholesale = row["Marginal cost"] + 0.42`; `run.py:50`: `wholesale -= 0.18`. ✓
- **Category:** HOLDS

---

### Finding 10: All numeric cells in contract-outcomes table — HOLDS

- **Claim source:** `README.md:121-124` and `tables/contract-outcomes.csv:2-4`
- **Verification:** All 21 cells (7 columns × 3 contracts) match between README and CSV to 2dp. Re-derived from code: all values match. No cell discrepancy found.
- Selected cells: Retailer variable profit Slotting fees = 24.67 (README matches CSV matches computed 24.67). Upstream profit All-unit discount = 7.94 (README matches CSV matches computed 7.94). ✓
- **Category:** HOLDS

---

### Finding 11: Upstream profit formula subtracts transfers — HOLDS

- **Claim source (verbatim):** "$\Pi_C^U(A)=\sum_{j\in A}\left[(w_j^C(A)-c_j)q_j(p_j^{*})-F_j^C(A)\right]$" — `README.md:54-56`; "Upstream profit subtracts those transfers." — `README.md:116`
- **Code evidence:** `run.py:55`: `upstream_var = (wholesale - row["Marginal cost"]) * quantity - fixed_fee`. Exact match to formula. ✓
- **Category:** HOLDS

## Cross-cutting patterns

- None. Both prior findings are fully resolved. The fix propagated the margin floor documentation into the Equations section with the exact numeric value ε=0.05 and an explicit vacuousness note. The τ=4 disclosure was added at two points: inline in the Equations equation block and in the Model Setup table row.
- All numeric claims are grounded end-to-end (code → CSV → README). No data drift between any two of the three artifact layers.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No halt trigger. No further remediation needed.
1. Violated-invariant tests for F1 and F2 (which encoded the buggy behavior) now pass per the test suite run: `test_finding1_violated_invariant_floor_present_in_code` passes (floor still in code, correctly), `test_finding2_violated_invariant_code_threshold_is_four` passes (default 4 confirmed). Both honest-fix tests pass: `test_finding1_honest_fix_floor_documented_in_readme` (0.05 and "floor" both present) and `test_finding2_honest_fix_threshold_value_disclosed_in_readme` (τ=4 present). Green state confirmed.
2. No further action required for this tutorial.
