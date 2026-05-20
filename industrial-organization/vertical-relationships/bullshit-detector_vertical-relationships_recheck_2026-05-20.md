# bullshit-detector — vertical-relationships — recheck — 2026-05-20

**Bullshit score: 0%** — Prior F1 MISLABELED/MED resolved: Solution Method pseudocode now names the closed-form explicitly and the phrase "For a candidate wholesale price" is gone. All formula, algorithm, and numeric claims grounded and verified. No new findings.

## Header
- Claim sources: `industrial-organization/vertical-relationships/README.md` (all sections)
- Code / artifact root: `industrial-organization/vertical-relationships/run.py`
- Data artifacts: `industrial-organization/vertical-relationships/tables/vertical-contracts.csv`
- Seed audit: `bullshit-detector_vertical-relationships_2026-05-20.md` (score 20%, 1 finding)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| F1 (prior) | Pseudocode loop → now names closed-form w_DM | HOLDS | — | — |
| 1 | Demand q(p) = a - bp, choke price p̄ = a/b | HOLDS | — | — |
| 2 | Integrated price p^I = (p̄ + c_M + c_R)/2 | HOLDS | — | — |
| 3 | Retailer best response p_R(w) = (p̄ + w + c_R)/2 | HOLDS | — | — |
| 4 | Manufacturer FOC gives w^DM = (p̄ - c_R + c_M)/2 | HOLDS | — | — |
| 5 | Two-part tariff: w^TPT = c_M, F = (p^I - c_M - c_R)q(p^I) | HOLDS | — | — |
| 6 | Solution Method step 2: closed-form, no loop | HOLDS | — | — |
| 7 | Integrated: price=6.50, quantity=7.0 | HOLDS | — | — |
| 8 | Linear wholesale: price=8.25, quantity=3.5 | HOLDS | — | — |
| 9 | TPT restores integrated price and quantity | HOLDS | — | — |
| 10 | All 30 numeric cells in results table | HOLDS | — | — |

## Findings

### Prior Finding 1 — RESOLVED: pseudocode now names closed-form solution

- **Prior claim (buggy):** Pseudocode said "For a candidate wholesale price w: / retailer sets p_R(w)..." implying an iterative search, while the code used a single closed-form expression with no loop.
- **Current run.py evidence (verbatim):**
  ```python
  "2. Linear wholesale game\n"
  "    Retailer best response: p_R(w) = (a/b + w + c_R) / 2\n"
  "    Manufacturer FOC for max (w-c_M) q(p_R(w)) has the closed-form\n"
  "        solution w_DM = (a/b - c_R + c_M) / 2, evaluated directly\n"
  "    Evaluate p_R(w_DM), q(p_R(w_DM)), profits, and surplus\n"
  ```
  `run.py:166-170`
- **Code grounding:** `run.py:49`: `wholesale_dm = (a / b - cr + cm) / 2` — single expression. `"optimize" in src`: False; `"linspace" in src`: False; `"for w" in src`: False. No loop or optimizer present.
- **README grounding:** `"For a candidate wholesale price" in README.md`: False; `"closed-form" in README.md`: True.
- **Category:** HOLDS

---

### Finding 1: Demand and choke price — HOLDS

- **Claim source (verbatim):** "Demand follows $$q(p)=a-bp,\qquad p\leq \bar p\equiv a/b,$$" — `README.md:14`
- **Code evidence:** `run.py:16`: `return max(a - b * price, 0.0)`. ✓ Choke price a/b computed at `run.py:46`: `total_cost = cm + cr`, `monopoly_price = (a + b * total_cost) / (2 * b)` implicitly uses a/b = 10.0. ✓
- **Category:** HOLDS

---

### Finding 2: Integrated price formula — HOLDS

- **Claim source (verbatim):** "$$p^I=\frac{\bar p+c_M+c_R}{2}$$" — `README.md:21`
- **Code evidence:** `run.py:47`: `monopoly_price = (a + b * total_cost) / (2 * b)` = `(a/b + c_M + c_R)/2`. With a=20, b=2, c_M=2, c_R=1: `(10 + 3)/2 = 6.5`. ✓
- **Category:** HOLDS

---

### Finding 3: Retailer best response — HOLDS

- **Claim source (verbatim):** "Its best response is $$p_R(w)=\frac{\bar p+w+c_R}{2}.$$" — `README.md:25-26`
- **Code evidence:** `run.py:50`: `retail_dm = (a + b * (wholesale_dm + cr)) / (2 * b)` = `(a/b + w + c_R)/2`. ✓
- **Category:** HOLDS

---

### Finding 4: Manufacturer FOC w^DM — HOLDS

- **Claim source (verbatim):** "$$w^{DM}=\frac{\bar p-c_R+c_M}{2}.$$" — `README.md:31`
- **Code evidence:** `run.py:49`: `wholesale_dm = (a / b - cr + cm) / 2`. With a/b=10, c_R=1, c_M=2: `(10 - 1 + 2)/2 = 5.5`. ✓
- **Data evidence:** `tables/vertical-contracts.csv:3`: `Linear wholesale,8.25,5.50,...` — wholesale = 5.50 matches computed 5.5. ✓
- **Category:** HOLDS

---

### Finding 5: Two-part tariff setup — HOLDS

- **Claim source (verbatim):** "A two-part tariff sets $$w^{TPT}=c_M$$ and uses the fixed fee $$F=(p^I-c_M-c_R)q(p^I)$$" — `README.md:35-38`
- **Code evidence:** `run.py:52`: `two_part_w = cm`; `run.py:53-54`: fee computed as retailer profit `(p_I - c_M - c_R) * q_I` when `fixed_fee=0`, then passed to `outcome()` as fixed fee. With p_I=6.5, c_M=2, c_R=1, q_I=7: `(6.5 - 3) * 7 = 24.5`. ✓
- **Data evidence:** `tables/vertical-contracts.csv:4`: `Two-part tariff,6.50,2.00,24.50,...` — fixed fee 24.50. ✓
- **Category:** HOLDS

---

### Finding 6: Solution Method closed-form (F1 resolution) — HOLDS

- Already grounded above. No loop, no optimizer, closed-form `(a / b - cr + cm) / 2` at `run.py:49`. README pseudocode names the method: "has the closed-form solution w_DM = (a/b - c_R + c_M) / 2, evaluated directly." — `README.md:69-70`. ✓
- **Category:** HOLDS

---

### Finding 7-9: Numeric claims on prices, quantities, and contract comparisons — HOLDS

- **Claim source (verbatim):** "The integrated channel charges $6.50$ and sells 7.0 units. Linear wholesale pricing raises the retail price to $8.25$ and cuts quantity to 3.5. The two-part tariff returns price and quantity to the integrated line." — `README.md:83`
- **Computed:** Integrated: p=6.5, q=7.0. Linear: p=8.25, q=3.5. TPT: p=6.5, q=7.0. All match. ✓
- **Category:** HOLDS

---

### Finding 10: All 30 numeric cells in results table — HOLDS

- **Claim source:** `README.md:97-99` and `tables/vertical-contracts.csv`
- **Verification:** Full recomputation for all three contracts and all nine numeric columns. All 30 cells match between README and CSV and computed values within the 2dp display format. Selected checks:
  - Linear wholesale channel profit: computed 18.375, displayed 18.38. ✓ (2dp rounding)
  - Linear wholesale consumer surplus: computed 3.0625, displayed 3.06. ✓ (Python banker rounding: 3.0625 → 3.06)
  - Linear wholesale retailer profit: computed 6.125, displayed 6.12. ✓ (banker rounding: 6.125 → 6.12)
  - CSV matches README on all 30 cells. ✓
- **Category:** HOLDS

## Cross-cutting patterns

- None. The single prior finding is fully resolved. The pseudocode was updated from a loop-implying "For a candidate wholesale price w:" framing to an explicit "Manufacturer FOC... has the closed-form solution w_DM = ..., evaluated directly" statement, matching the code exactly.
- All formula derivations (p^I, p_R(w), w^DM, F) remain algebraically consistent between Equations section and run.py. No formula drift.
- The 2dp display rounding (Python's banker rounding, e.g. 6.125 → 6.12, 3.0625 → 3.06) is consistent between CSV and README in all cells. Not a finding — the rounding is systematic and a reader can verify it.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No halt trigger. No further remediation needed.
1. Both tests in the test suite are in the correct green state: `test_finding1_violated_invariant_solver_is_closed_form` passes (no loop or optimizer; closed-form formula present), and `test_finding1_honest_fix_pseudocode_names_closed_form` passes ("For a candidate wholesale price" absent; "closed-form" present in README). Green state confirmed.
2. No further action required for this tutorial.
