# bullshit-detector — vertical-contracts — 2026-05-20

**Bullshit score: 25%** — Two DILUTED findings (undisclosed τ=4 threshold; undisclosed price-floor branch); neither changes any published number, but a reader cannot reproduce the exact model from the equations alone.

## Header

- Claim sources: `industrial-organization/vertical-contracts/README.md`
- Code / artifact root: `industrial-organization/vertical-contracts/run.py`
- Data artifacts: `industrial-organization/vertical-contracts/tables/contract-outcomes.csv`
- Seed audit: none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | `p*(w) = (a+bw)/(2b)` — interior formula is complete | DILUTED | LOW | no (floor never binds in this parameterisation; needs re-run to verify under different params) |
| 2 | Discount threshold τ stated as symbol but never assigned a value | DILUTED | MED | no (τ=4 is the hardcoded default; results are correct but unreproducible from README alone) |
| 3 | 792 feasible assortments | HOLDS | — | — |
| 4 | Wholesale only: 4 Mars slots | HOLDS | — | — |
| 5 | Discount + slotting: 5 Mars slots each | HOLDS | — | — |
| 6 | All payoff numbers (obj, var profit, upstream, fees, price, qty) | HOLDS | — | — |
| 7 | μ=0.42, d=0.18 | HOLDS | — | — |
| 8 | Slotting fee 1.10 Mars / 0.35 rival | HOLDS | — | — |
| 9 | Fixed fees=0 for Wholesale only and All-unit discount | HOLDS | — | — |
| 10 | Retailer objective = variable profit + fixed fees | HOLDS | — | — |
| 11 | Upstream profit subtracts transfers | HOLDS | — | — |

## Findings

### Finding 1: Interior price formula omits margin floor

- **Claim source (verbatim):** "For an interior product, this gives $p_j^{\ast}(w_j)=\frac{a_j+bw_j}{2b}$." — `README.md:28-31`
- **Code evidence (verbatim):**
  ```python
  def retail_outcome(intercept: float, wholesale: float, slope: float = 4.0) -> tuple[float, float]:
      """Interior monopoly price under linear demand, with a small margin floor."""
      price = max((intercept + slope * wholesale) / (2 * slope), wholesale + 0.05)
      quantity = max(intercept - slope * price, 0.0)
      return price, quantity
  ```
  `run.py:33-37`
- **Data evidence:** Floor never binds for any of the 12 products under any contract with the committed parameterisation (verified by exhaustive check across all wholesale prices). All CSV numbers match the interior formula. No result impact confirmed in this parameterisation; needs re-run to verify under different params.
- **Category:** DILUTED — code implements a strictly richer formula than the one the Equations section claims; the floor branch is load-bearing logic the equations silently omit.
- **Severity:** LOW — zero result impact under the committed params; floor never fires.
- **Result-changing:** no (under committed parameterisation; needs re-run to verify under different params)
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "wholesale + 0.05" in inspect.getsource(retail_outcome)
  # PASSES on current code (floor present); FAILS on honest fix (floor removed or documented)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "wholesale + 0.05" not in readme_text or r"wholesale + 0.05" in readme_equations_section
  # PASSES once floor is either removed from code or documented in README Equations; FAILS now
  ```

### Finding 2: Discount threshold τ introduced but never assigned

- **Claim source (verbatim):** "The discount applies only if the assortment contains at least $\tau$ Mars products" — `README.md:53`; Model Setup table: "Mars margin falls by $d=0.18$ once the shelf target is met" — `README.md:75` (no numeric value given for τ or "shelf target")
- **Code evidence (verbatim):**
  ```python
  def evaluate_subset(products: pd.DataFrame, subset: tuple[int, ...], contract: str, threshold: int = 4) -> dict[str, float]:
      ...
      mars_count = int((products.loc[list(subset), "Manufacturer"] == "Mars").sum())
      ...
      if contract == "All-unit discount" and is_mars and mars_count >= threshold:
          wholesale -= 0.18
  ```
  `run.py:40-50`
  ```python
  subset, values = choose_assortment(products, capacity, contract)
  ```
  `run.py:94` — no `threshold` argument passed; default 4 is always used.
- **Data evidence:** CSV shows 5 Mars slots under All-unit discount, consistent with τ=4 (activating once 4+ Mars products are stocked, which is satisfied by the optimal assortment of 5). Number is reproducible only if a reader knows τ=4.
- **Category:** DILUTED — the symbol τ is used in the equation but its value is never disclosed in the README. A reader following the equations cannot reproduce the result without inspecting the code.
- **Severity:** MED — the tutorial teaches the mechanism of the threshold discount, but the specific threshold value that produces the reported results is invisible to the reader.
- **Result-changing:** no (τ=4 is the only value used; results are self-consistent)
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert re.search(r"\\tau\s*=\s*4|threshold.*4|4 Mars", readme_text) is None
  # PASSES on current README (τ value absent); FAILS once the value is disclosed
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert re.search(r"\\tau\s*=\s*4|\bthreshold\b.*\b4\b", readme_equations_or_setup_text) is not None
  # PASSES once τ=4 appears in README Equations or Model Setup; FAILS now
  ```

## Cross-cutting patterns

- Both DILUTED findings are omissions in the README rather than errors in the code. The code is internally consistent and all reported numbers are reproducible from the code. The gap is between the equations-as-written and the equations-as-implemented.
- The price floor (`wholesale + 0.05`) follows the pattern of a defensive numerical guard added during implementation but not propagated back to the theoretical write-up. Check other tutorials in this repo for similar silent guards that do not appear in their Equations sections.
- τ is introduced as a free symbol in the equations block but treated as a hardcoded constant in the code with no argument threading to `main()`. If a future tutorial variant changes τ, the default in `evaluate_subset` would need updating and the README would still not disclose the value. Consider either (a) exposing τ in the Model Setup table alongside μ and d, or (b) threading it through `main()` so it appears as a named constant at the top of the script.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Proceed after the two DILUTED findings are addressed.

1. **Finding 1 — price floor.**
   - Write `test_price_floor_documented`: assert `"wholesale + 0.05"` does not appear in `run.py` without a corresponding disclosure in `README.md`, OR assert that for all 12 products × all contract branches the floor never binds (confirming it is numerically vacuous and can be safely removed). Confirm test fails on current state (floor present, README silent).
   - Fix: either (a) remove the floor if it was added defensively and is never needed, OR (b) add a sentence to the Equations section documenting it. Re-run `python run.py` and confirm CSV numbers unchanged.

2. **Finding 2 — threshold τ.**
   - Write `test_threshold_disclosed`: parse `README.md` and assert `re.search(r"\\\\tau\\s*=\\s*4|shelf target.*4", readme_text) is not None`. Confirm test fails on current README.
   - Fix: add `τ = 4` to the Model Setup table row for the All-unit discount contract. Re-run `python scripts/validate_catalog.py`.

3. After fixes, re-run this skill. Expected new score: 0-10% (both findings resolve to HOLDS).
