# bullshit-detector — theory-of-the-firm — 2026-05-20

**Bullshit score: 15%** — one DILUTED/LOW finding: prose gloss "capture share b_g(s) of marginal value" is imprecise relative to what the code and FOC equation actually implement; all numeric claims hold.

## Header
- Claim sources: `industrial-organization/theory-of-the-firm/README.md` (Overview, Equations, Model Setup, Solution Method, Results)
- Code / artifact root: `industrial-organization/theory-of-the-firm/run.py`
- Data artifacts: `industrial-organization/theory-of-the-firm/tables/governance-comparison.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "capture share b_g(s) of marginal value" | DILUTED | LOW | no (FOC equation is correct; prose gloss only) |
| 2 | FOC formula b_g(s)*theta - x = 0 | HOLDS | — | — |
| 3 | x_g(s) = b_g(s)*theta | HOLDS | — | — |
| 4 | W_g(s) = theta*x_g - 0.5*x_g^2 - F_g(s) | HOLDS | — | — |
| 5 | W* = 0.5*theta^2 | HOLDS | — | — |
| 6 | Incentive schedule parameters (all six coefficients) | HOLDS | — | — |
| 7 | Governance cost parameters (all six coefficients) | HOLDS | — | — |
| 8 | Table numerics (all 12 rows) | HOLDS | — | — |
| 9 | Threshold claims (~0.21 and ~0.37) | HOLDS | — | — |

## Findings

### Finding 1: "capture share b_g(s) of marginal value" — imprecise prose gloss

- **Claim source (verbatim):** "Regime $g$ lets the investor capture share $b_g(s)$ of marginal value." — `README.md:22`
- **Code evidence (verbatim):**
  ```python
  incentive = np.clip(values["incentive"], 0.05, 1.0)
  investment = theta * incentive
  surplus = theta * investment - 0.5 * investment**2 - values["governance_cost"]
  ```
  `run.py:38-40`
- **Data evidence (if applicable):** Not applicable — this is a structural/prose claim, not numeric.
- **Analysis:** The code implements a model where the investor's private payoff is implicitly `b_g(s)*theta*x - 0.5*x^2` (investor receives fraction `b_g` of revenue `theta*x`, bears full quadratic cost). The private FOC is `b_g(s)*theta - x = 0`, giving `x = b_g(s)*theta`. This is correctly stated in the next sentence of README.md:23 as `$$b_g(s)\theta - x = 0,$$`.

  "Marginal value" in standard usage refers to `V'(x) = theta - x`. If the investor captured share `b_g` of marginal value `V'(x)`, the private FOC would be `b_g*(theta - x) = 0`, giving `x = theta` regardless of `b_g` — eliminating hold-up entirely, contradicting the model's purpose. The code and the stated FOC are consistent with the investor capturing share `b_g` of REVENUE (the linear term `theta*x`), not of marginal value `V'(x)`.

  The following sentence's equation is correct and consistent with the code. The prose gloss one sentence before is imprecise but does not change any computed result because no simulation or table draws from the prose sentence — only from the FOC equation and parameter schedules.

- **Category:** DILUTED
- **Severity:** LOW (the correct FOC equation immediately follows in the same section; no number is miscomputed)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "theta - x" not in open("run.py").read()  # PASSES on current code; FAILS if code switched to V'(x) sharing
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "share of revenue" in open("README.md").read() or "share of marginal productivity" in open("README.md").read()
  ```

---

### Finding 2: FOC formula — HOLDS

- **Claim source (verbatim):** "The private first-order condition is $$b_g(s)\theta - x = 0,$$" — `README.md:23-24`
- **Code evidence (verbatim):**
  ```python
  incentive = np.clip(values["incentive"], 0.05, 1.0)
  investment = theta * incentive
  ```
  `run.py:38-39`
- `investment = theta * incentive` ↔ `x = b_g(s)*theta` ↔ FOC `b_g*theta - x = 0`. ✓
- **Category:** HOLDS

---

### Finding 3: Investment formula x_g(s) = b_g(s)*theta — HOLDS

- **Claim source (verbatim):** "which gives $$x_g(s) = b_g(s)\theta$$" — `README.md:26`
- **Code evidence:** `investment = theta * incentive` at `run.py:39`, where `incentive = b_g(s)` from `run.py:38`. ✓
- **Category:** HOLDS

---

### Finding 4: Surplus formula — HOLDS

- **Claim source (verbatim):** "Total surplus subtracts governance cost $F_g(s)$: $$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$" — `README.md:28-29`
- **Code evidence (verbatim):**
  ```python
  surplus = theta * investment - 0.5 * investment**2 - values["governance_cost"]
  ```
  `run.py:40`
- Exact match. ✓
- **Category:** HOLDS

---

### Finding 5: First-best surplus W* = 0.5*theta^2 — HOLDS

- **Claim source (verbatim):** "The first-best surplus benchmark is $$W^{\ast}=\frac{1}{2}\theta^2$$" — `README.md:44-45`
- **Code evidence (verbatim):**
  ```python
  first_best_surplus = 0.5 * theta**2
  ```
  `run.py:36` (inside `investment_outcomes`) and `run.py:88` (in `main`).
- With `theta = 4.0`: `0.5 * 16 = 8.0`. ✓
- **Category:** HOLDS

---

### Finding 6: Incentive schedule parameters — HOLDS

- **Claim source (verbatim):**
  "`b_{\text{spot}}(s)=0.72-0.55s,\quad b_{\text{contract}}(s)=0.72-0.25s,\quad b_{\text{integration}}(s)=0.74-0.03s.`" — `README.md:32-33`
- **Code evidence (verbatim):**
  ```python
  "Spot contract": {
      "incentive": 0.72 - 0.55 * specificity,
  ...
  "Long-term contract": {
      "incentive": 0.72 - 0.25 * specificity,
  ...
  "Vertical integration": {
      "incentive": 0.74 - 0.03 * specificity,
  ```
  `run.py:22-32`
- All six coefficients match exactly. ✓
- **Note on np.clip:** `run.py:38` clips `incentive` to `[0.05, 1.0]`. On the domain `s ∈ [0,1]`, the minimum raw incentive is `b_spot(1) = 0.17 > 0.05`, so clipping never activates. This dead code does not affect any output and is not mentioned in the README.
- **Category:** HOLDS

---

### Finding 7: Governance cost parameters — HOLDS

- **Claim source (verbatim):**
  "`F_{\text{spot}}(s)=0.02+0.04s,\quad F_{\text{contract}}(s)=0.38+0.03s,\quad F_{\text{integration}}(s)=1.05-0.35s.`" — `README.md:37-38`
- **Code evidence (verbatim):**
  ```python
  "Spot contract": {
      "governance_cost": 0.02 + 0.04 * specificity,
  ...
  "Long-term contract": {
      "governance_cost": 0.38 + 0.03 * specificity,
  ...
  "Vertical integration": {
      "governance_cost": 1.05 - 0.35 * specificity,
  ```
  `run.py:23-32`
- All six coefficients match exactly. ✓
- **Category:** HOLDS

---

### Finding 8: Table numerics (all 12 rows) — HOLDS

- **Claim source:** Table in `README.md:100-115` and `tables/governance-comparison.csv:1-13`
- **Verification method:** Full re-derivation using the stated formulas and parameters for all four specificity levels (s = 0.0, 0.3, 0.5, 1.0) and all three regimes.
- **Result:** All surplus values, investment values, and efficiency ratios match within display precision. Three rows showed apparent incentive-share discrepancy of 0.005 (e.g., computed 0.6450 vs CSV 0.65 at s=0.3, Long-term contract) — these are pure display rounding artifacts in the CSV column format (`f"{x:.2f}"`). The downstream x, W, and efficiency ratio values all agree exactly. ✓
- **Selected data evidence:** `tables/governance-comparison.csv:6`: `0.3,Long-term contract,0.65,2.58,6.60,82.5%,yes` — computed W = 6.6028, rounds to 6.60. ✓
- **Category:** HOLDS

---

### Finding 9: Governance threshold claims (~0.21, ~0.37) — HOLDS

- **Claim source (verbatim):** "spot exchange wins for $s\lesssim 0.21$. Long-term contracts win for $0.21\lesssim s\lesssim 0.37$. Vertical integration wins for $s\gtrsim 0.37$." — `README.md:82`
- **Code evidence:** Thresholds are generated dynamically by `summarize_regions(fine_best)` at `run.py:86`, where `fine_best` uses a 10001-point grid. The f-string at `run.py:184-188` inserts `regions[0][1]`, `regions[1][0]`, `regions[1][1]`, `regions[2][0]` formatted to 2 decimal places.
- **Verified transitions:** Spot → Long-term contract at s ≈ 0.2059 (displays as 0.21); Long-term contract → Vertical integration at s ≈ 0.3742 (displays as 0.37). Both match the README text within 2dp rounding. ✓
- **Category:** HOLDS

## Cross-cutting patterns

- **One-finding audit.** Only a single non-HOLDS finding exists and it is LOW severity DILUTED. The math is internally consistent throughout. The prose sentence at `README.md:22` ("capture share b_g(s) of marginal value") is the only gap, immediately corrected by the correct FOC equation on `README.md:23`.
- **Dead code that cannot cause drift.** `np.clip(incentive, 0.05, 1.0)` at `run.py:38` is never triggered on the parameter domain `s ∈ [0,1]` with the stated schedules. It is not claimed in the README. If the governance parameters ever change to produce negative incentive shares, this clip could silently alter outputs without triggering an error. Not a current finding but a fragility worth noting.
- **Dynamic threshold insertion prevents data drift.** The threshold values are embedded in README via f-string at `run.py:184-188`, so they cannot drift from the code output. This is a good practice that eliminates an entire class of DATA DRIFT findings.
- **Table uses coarse grid (201 points) for chosen-regime flag.** The figure's governance regions use the fine 10001-point grid. The table uses the 201-point grid. At the four selected specificity values (0.0, 0.3, 0.5, 1.0), all are exactly representable on both grids (step = 0.005), so no discrepancy arises. Not a finding under current parameters, but the code has two separate `best_regime` calls on different grids that are not documented.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** Score is 15%. Below the 50% halt threshold. Safe to proceed to touch-up.
1. Finding 1 is LOW severity DILUTED prose imprecision. The violated-invariant test is already expressed as a one-liner above.
2. The single fix: replace "capture share $b_g(s)$ of marginal value" at `run.py:122` (inside the `add_equations` string) with "capture share $b_g(s)$ of revenue" (or "capture share $b_g(s)$ of the marginal productivity $\theta$"). Regenerate README. Confirm the honest-fix assertion passes.
3. No simulation changes needed. No table regeneration needed. No figure regeneration needed.
4. Re-run `scripts/validate_catalog.py` to confirm no math-rendering regressions.
5. Re-run this skill on the updated README. Expected new score: 0%.
