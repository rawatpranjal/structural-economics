# bullshit-detector — optimal-growth — 2026-05-20

**Bullshit score: 20%** — pseudocode omits load-bearing 0.9999 factor (DILUTED/MED); three numeric claims cannot be grounded in committed artifacts without a re-run (DATA DRIFT/LOW).

## Header
- Claim sources: `dynamic-programming/optimal-growth/README.md` (Overview, Equations, Model Setup, Solution Method, Results)
- Code / artifact root: `dynamic-programming/optimal-growth/run.py`
- Data artifacts: `dynamic-programming/optimal-growth/tables/comparison.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|-----------------|
| 1 | Pseudocode `kp_max <- min(y_i, k_max)` (open bracket implied) | DILUTED | MED | no (code correct; pseudocode misleading) |
| 2 | Max value error outside bottom decile is 1.91e-05 | DATA DRIFT | LOW | no (cannot ground in committed CSV; needs re-run to verify) |
| 3 | Max policy gap outside bottom decile is 2.87e-02 | DATA DRIFT | LOW | no (table max is 1.98e-02; 2.87e-02 from un-sampled grid point; needs re-run to verify) |
| 4 | VFI converges in 143 steps with residual 9.32e-07 | DATA DRIFT | LOW | no (plausible but needs re-run to verify) |
| 5 | All formula/parameter claims (kss, css, B, E, saving rate, k_max, table values) | HOLDS | none | — |

## Findings

### Finding 1: Pseudocode omits 0.9999 factor — literal execution produces log(0) = -inf

- **Claim source (verbatim):** `"kp_max <- min(y_i, k_max)"` and `"kp <- N_{k'} points uniform on [k_min, kp_max)"` — `README.md:95-96` (Solution Method pseudocode block)
- **Code evidence (verbatim):**
  ```python
  kp_max = min(output * 0.9999, k_max)
  kp_grid = np.linspace(k_min, kp_max, n_kprime)
  ```
  `run.py:88-89`
- **Data evidence (if applicable):** None. The numerical results are correct; the gap is between pseudocode and implementation only.
- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** no — the code is correct and results are unaffected; the pseudocode is the artifact that misleads.
- **Analysis:** The pseudocode states `kp_max <- min(y_i, k_max)`. `np.linspace` is closed on both endpoints by default. If followed literally (no 0.9999 scaling), the last grid point would be `k' = y_i`, giving `c = y_i - y_i = 0`, and `log(0) = -inf`. The 0.9999 factor is a load-bearing implementation detail that prevents this; the pseudocode omits it entirely. The Solution Method prose one sentence earlier says the feasible range is `[k_min, A k_i^alpha)` (open bracket), which is consistent with the code's intent but does not show how openness is achieved. A reader transcribing the pseudocode verbatim would produce a broken solver.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.9999" not in open("run.py").read()  # PASSES on current buggy pseudocode state; FAILS when pseudocode is fixed
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "0.9999" in open("README.md").read()  # PASSES when pseudocode is updated; FAILS now
  ```

---

### Finding 2: Max value error 1.91e-05 unverifiable from committed artifacts

- **Claim source (verbatim):** `"Outside the bottom decile, the largest value gap is **1.91e-05**."` — `README.md:110`
- **Code evidence (verbatim):**
  ```python
  max_value_error = float(np.max(np.abs(value_error[valid_start:])))
  ```
  `run.py:154`
- **Data evidence:** `tables/comparison.csv` contains 8 sampled rows. Maximum `|V error|` across all 8 rows is `1.53e-05`. The claimed `1.91e-05` exceeds every entry in the table and comes from a grid point not present in the CSV. No committed artifact records the full 500-point error vector.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no — the order of magnitude is correct; the exact number needs re-run to verify.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert max(abs(float(r["V error"])) for r in csv.DictReader(open("tables/comparison.csv"))) >= 1.91e-05  # FAILS on current CSV (max is 1.53e-05)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(abs(float(r["V error"])) >= 1.91e-05 for r in csv.DictReader(open("tables/comparison.csv")))  # PASSES if re-run or if full error vector is committed
  ```

---

### Finding 3: Max policy gap 2.87e-02 exceeds every committed table entry

- **Claim source (verbatim):** `"The largest policy gap outside the bottom decile is **2.87e-02**."` — `README.md:114`
- **Code evidence (verbatim):**
  ```python
  max_policy_error = float(np.max(np.abs(policy_error[valid_start:])))
  ```
  `run.py:155`
- **Data evidence:** `tables/comparison.csv` max `|k' error|` across all 8 rows is `1.98e-02` (row 1: `1.98e-02`). The claimed `2.87e-02` is 45% larger than the largest entry in the committed table. The figure comes from a non-sampled grid point; no committed artifact records it.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no — the grid-discretization interpretation is qualitatively unchanged; exact value needs re-run to verify.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert max(abs(float(r["k' error"])) for r in csv.DictReader(open("tables/comparison.csv"))) >= 2.87e-02  # FAILS on current CSV
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(abs(float(r["k' error"])) >= 2.87e-02 for r in csv.DictReader(open("tables/comparison.csv")))  # PASSES after re-run or extended table
  ```

---

### Finding 4: Convergence iteration count and residual unverifiable from committed artifacts

- **Claim source (verbatim):** `"The iteration converges in **143 steps** with sup-norm residual **9.32e-07**."` — `README.md:106`
- **Code evidence (verbatim):**
  ```python
  info = {"iterations": iteration, "converged": error < tol, "error": error}
  ```
  `run.py:109`
  The README embeds these values via an f-string at report generation time (`run.py:281-283`). No committed log or stdout file records the actual values.
- **Data evidence:** No stdout file, no log file committed. Values are plausible (9.32e-07 < tol=1e-6; 143 iterations for 500-point grid with tol=1e-6 is in the expected range for this model), but cannot be verified without re-run. **needs re-run to verify**
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no — the values are self-consistent and qualitatively unremarkable.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("stdout_vfi.log").exists()  # FAILS (no log committed); PASSES after fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert int(open("stdout_vfi.log").read().split("converged in ")[1].split()[0]) == 143  # PASSES after re-run with log
  ```

---

### Finding 5: All closed-form formula, parameter, and table claims — HOLDS

- **Claim source:** Equations section (`README.md:11-59`), Model Setup table (`README.md:63-76`), Results table (`README.md:126-135`).
- **Code evidence:** All verified by re-derivation against `run.py`:
  - `kss = (alpha*beta*A)**(1/(1-alpha))` — `run.py:36` — matches README formula and value `9.9519`. HOLDS.
  - `css = A*kss**alpha - kss` — `run.py:37` — matches README value `26.9071`. HOLDS.
  - `B_const = alpha/(1-alpha*beta)` — `run.py:57` — matches README. HOLDS.
  - `E_const` formula — `run.py:58-61` — identical to README Equations. HOLDS.
  - Saving rate `alpha*beta = 0.27` — `run.py:28-30` — matches README. HOLDS.
  - `k_max = kss*2.5 = 24.88` (displayed via `.2f`) — `run.py:41,243` — matches README. HOLDS.
  - All 8 rows in `tables/comparison.csv` match `README.md` table verbatim. HOLDS.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** —

## Cross-cutting patterns

- All three DATA DRIFT findings share the same root: the `max_*_error` scalars are computed from the full 500-point grid at runtime but only 8 representative rows are committed in `tables/comparison.csv`. The committed CSV cannot reproduce any of the three "max ... outside bottom decile" claims. A committed `tables/full_errors.csv` (or extended 500-row table) would resolve all three in one step.
- The pseudocode (Finding 1) and the utility guard (`np.maximum(c, 1e-15)` in `u_np`) both hide implementation details that prevent `log(0)`. The pseudocode says `u(c) = log c`; the code does `np.log(np.maximum(c, 1e-15))`. This is a second instance of the same pattern: the 0.9999 upper-bound trick and the 1e-15 floor are complementary guards, both absent from the pseudocode. Neither change the results, but a reader implementing from the pseudocode alone would need both.
- The Equations section Bellman constraint says `0 < k'` (strictly positive); the code imposes `k' >= k_min = 0.01`. The Solution Method prose names `k_min = 0.01` explicitly, so the bound is disclosed — but the Equations section is imprecise by 0.01. This is sub-threshold (disclosed two sections later), not a separate finding.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Below the 50% halt threshold. Proceed with fixes.

1. **Finding 1 (DILUTED/MED) — pseudocode fix:**
   - Write `pytest` test: `assert "0.9999" not in open("dynamic-programming/optimal-growth/README.md").read()` — this PASSES now (proves gap).
   - Fix: update the pseudocode in `run.py`'s `add_solution_method` call to read `kp_max <- min(y_i * 0.9999, k_max)` (or add a note: `# ensures c > 0`). Regenerate README.
   - Pass condition test: `assert "0.9999" in open("dynamic-programming/optimal-growth/README.md").read()`.

2. **Findings 2-4 (DATA DRIFT/LOW) — artifact coverage fix:**
   - Add a `tables/full_errors.csv` output to `run.py` containing the full 500-point `(k, V_error, policy_error)` array, so all three `max_*_error` scalars can be independently verified without re-running.
   - Alternative: extend `tables/comparison.csv` to include the row(s) that achieve the reported maxima.
   - Pass condition for Finding 3: `assert max(abs(float(r["k' error"])) for r in csv.DictReader(open("tables/comparison.csv"))) >= 2.87e-02` must pass after the fix.

3. Re-run `python run.py` inside `dynamic-programming/optimal-growth/`. Confirm README is regenerated. Re-run this skill on the new outputs. Target score: 0-10%.
