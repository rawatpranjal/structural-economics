# bullshit-detector — envelope-equation-iteration — 2026-05-20

**Bullshit score: 30%** — title "Persistent Income" contradicts the IID income process implemented throughout the codebase; convergence comparison plots value-level and consumption-level errors on the same axis without disclosure, making the VFI iteration count comparison misleading.

## Header
- Claim sources: `heterogeneous-agents/envelope-equation-iteration/README.md`
- Code / artifact root: `heterogeneous-agents/envelope-equation-iteration/run.py`
- Data artifacts: `heterogeneous-agents/envelope-equation-iteration/tables/solution-statistics.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Title: "Persistent Income" | MISLABELED | HIGH | yes — mislabels the economic model to every reader |
| 2 | Convergence figure: VFI "needs more iterations" | DILUTED | MED | yes — iteration counts measured in different units |
| 3 | Prose "0.220" MPC vs table "0.2197" | DATA DRIFT | LOW | no — same run, rounding only |
| 4 | Prose "0.41" mean assets vs table "0.4124" | DATA DRIFT | LOW | no — same run, rounding only |
| 5 | Takeaway: "EGP is faster here" | DILUTED | LOW | no — contradicts own "not a timing claim" disclaimer |

## Findings

### Finding 1: Title says "Persistent Income"; model implements IID income

- **Claim source (verbatim):** `"# Buffer-Stock Saving with Persistent Income by Envelope-Equation Iteration"` — `README.md:1`
- **Code evidence (verbatim):**
  ```python
  """Envelope-equation iteration for an IID income-risk saving problem.
  ```
  `run.py:2`
  ```python
      # Income risk: 5-state IID approximation to N(mu_y, sd_y^2)
  ```
  `run.py:311`
  ```python
      income_grid, income_probs = build_income_grid(n_income, mean_income, sd_income)
  ```
  `run.py:336`
  No AR(1) parameter (`rho`), no `Rouwenhorst`, no autocorrelation anywhere in `run.py`. The `simulate_panel` function draws income independently each period (`income_idx = np.searchsorted(cdf, rng.random(n_agents), side="right")`, `run.py:292`), confirming IID.
- **Data evidence (if applicable):** README body uses "IID" at lines 5, 13, 70, 100, 138, 146 — the title is the only location that says "Persistent Income". The title string is hardcoded identically in `run.py:432`: `"Buffer-Stock Saving with Persistent Income by Envelope-Equation Iteration"`.
- **Category:** MISLABELED — the code implements a coherent IID buffer-stock model; the title applies a label ("Persistent Income") that names a distinct and more complex income process (typically AR(1) with rho > 0). The economics are different: IID income implies no smoothing motive from past income draws; persistent income implies an additional precautionary demand. The catalog reader cannot distinguish this tutorial from an AR(1) tutorial by title alone.
- **Severity:** HIGH — the title is the primary catalog identifier; every reader and every cross-reference uses it to classify the tutorial's subject matter.
- **Result-changing:** yes — mislabels the economic object the tutorial studies to every reader who encounters it through the catalog.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Persistent" in open("heterogeneous-agents/envelope-equation-iteration/README.md").readline()
  # PASSES on current (buggy) file; FAILS on honest fix where title says IID
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Persistent" not in open("heterogeneous-agents/envelope-equation-iteration/README.md").readline()
  # PASSES on honest fix (title corrected to IID); FAILS on current file
  ```

---

### Finding 2: Convergence comparison plots value-level and consumption-level errors without disclosing the metric difference

- **Claim source (verbatim):** `"Grid VFI updates the value level and needs more iterations. This is a fixed-point comparison, not a timing claim."` — `README.md:150`
- **Code evidence (verbatim):**
  ```python
  # EEI (run.py:153)
  err = float(np.max(np.abs(consumption - consumption_old)))
  ```
  ```python
  # EGP (run.py:208)
  err = float(np.max(np.abs(consumption - consumption_old)))
  ```
  ```python
  # VFI (run.py:256)
  err = float(np.max(np.abs(V - V_old)))
  ```
  `run.py:153`, `run.py:208`, `run.py:256`
- **Data evidence (if applicable):** Table row `"Same-grid VFI iterations, 203"` (`solution-statistics.csv:4`) is presented alongside `"EEI iterations, 149"` and `"Same-grid EGP iterations, 151"` (`csv:2-3`). The convergence figure (`figures/convergence-comparison.png`) places all three error sequences on a single log-scale axis with the same tolerance line at `1e-6`.
- **Category:** DILUTED — the code's claim that VFI "needs more iterations" is grounded on comparing a value-level sup-norm to a consumption-level sup-norm measured to the same absolute tolerance `1e-6`. With `beta=0.95`, value-function magnitudes are of order `c/(1-beta) ≈ 20c`, so the value-level error is intrinsically larger-scaled and slower to cross the same absolute threshold. The README acknowledges VFI uses the value level but does not disclose that the three error sequences in the figure are in different units. A reader who accepts the iteration counts as a fair race between methods cannot, because the race is run with different finish lines effectively calibrated in different units. The disclaimer "not a timing claim" covers only wall-clock speed; it does not cover the metric-unit mismatch.
- **Severity:** MED — the comparison supports the tutorial's main lesson (EEI/EGP update policies directly and converge in fewer iterations than value-function iteration), but the evidence offered for that lesson is contaminated by the metric difference. The qualitative conclusion may still hold; the quantitative evidence (149 vs 203) is not clean.
- **Result-changing:** yes — the iteration count comparison, which is the only quantitative evidence for the convergence claim, cannot be trusted as stated without knowing the metric difference.
- **Violated invariant (one-line pytest assertion):**
  ```python
  import ast, inspect
  src = inspect.getsource(solve_vfi)
  assert "V - V_old" in src and "consumption - consumption_old" not in src
  # PASSES on current code (VFI tracks value, not consumption); FAILS if fixed to track consumption
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # After fix: either VFI tracks consumption error, or README discloses the metric difference.
  # Testable on the prose side:
  assert "value-level error" in open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  # PASSES on honest fix (disclosure added); FAILS on current README
  ```

---

### Finding 3: Prose "0.220" MPC vs table "0.2197"

- **Claim source (verbatim):** `"The borrowing-limit mass raises the average MPC to 0.220."` — `README.md:146`
- **Code evidence (verbatim):**
  ```python
  f"The borrowing-limit mass raises the average MPC to {mean_mpc:.3f}."
  ```
  `run.py:665`
  ```python
  f"{mean_mpc:.4f}",
  ```
  `run.py:726`
- **Data evidence (if applicable):** `"Mean MPC, 0.10 transfer,0.2197"` — `tables/solution-statistics.csv:11`
- **Category:** DATA DRIFT — same quantity (`mean_mpc`) reported at `.3f` precision in the figure description (yields "0.220") and `.4f` precision in the table (yields "0.2197"). Both derive from the same variable in the same run; no fabrication. The drift is a formatting inconsistency between two sections of the same report.
- **Severity:** LOW — no incorrect number; the rounded figure is arithmetically consistent with the table value.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.220" in open("heterogeneous-agents/envelope-equation-iteration/README.md").read() and "0.2197" in open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  # PASSES on current file (both appear); FAILS after fix normalizes precision
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  readme = open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  assert "0.220" not in readme or "0.2197" not in readme
  # PASSES on honest fix (one precision used throughout); FAILS on current file
  ```

---

### Finding 4: Prose "0.41" mean assets vs table "0.4124"

- **Claim source (verbatim):** `"Mean assets are 0.41."` — `README.md:146`
- **Code evidence (verbatim):**
  ```python
  f"Mean assets are {mean_assets:.2f}. "
  ```
  `run.py:662`
  ```python
  f"{mean_assets:.4f}",
  ```
  `run.py:723`
- **Data evidence (if applicable):** `"Mean assets,0.4124"` — `tables/solution-statistics.csv:9`
- **Category:** DATA DRIFT — same variable, two format strings, two precisions within the same run. Arithmetically consistent.
- **Severity:** LOW — no incorrect number.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.41" in open("heterogeneous-agents/envelope-equation-iteration/README.md").read() and "0.4124" in open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  readme = open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  assert "Mean assets are 0.4124" in readme or "Mean assets are 0.41." not in readme
  ```

---

### Finding 5: Takeaway "EGP is faster here" contradicts "not a timing claim" disclaimer

- **Claim source (verbatim):** `"EGP is faster here because it uses an analytic inverse."` — `README.md:179`
- **Code evidence (verbatim):**
  ```python
  "This is a fixed-point comparison, not a timing claim."
  ```
  `run.py:692` (embedded in `add_solution_method` string); also `README.md:150`
- **Category:** DILUTED — the Takeaway makes a wall-clock speed claim ("EGP is faster") that the Solution Method section explicitly declines to make. EGP uses `u_prime_inv` (closed form, `run.py:197`) avoiding 80-iteration bisection per state (`run.py:139`); this does imply faster wall-clock time per iteration. But the convergence figure reports iteration counts only, not timing. The prose is self-contradictory: one section says "not a timing claim" and the next makes one. The claim may be true (EGP is almost certainly faster per iteration), but it is unverified by any artifact in the tutorial.
- **Severity:** LOW — the underlying claim (EGP avoids bisection) is grounded in code; the contradiction is a prose discipline failure rather than a factual error.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  readme = open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  assert "not a timing claim" in readme and "EGP is faster here" in readme
  # PASSES on current (contradictory) file; FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  readme = open("heterogeneous-agents/envelope-equation-iteration/README.md").read()
  assert not ("not a timing claim" in readme and "EGP is faster here" in readme)
  # PASSES on honest fix (one of the two removed or qualified); FAILS on current
  ```

---

## Cross-cutting patterns

- The only location that says "Persistent Income" is the title string, hardcoded identically in `README.md:1` and `run.py:432`. Every other prose occurrence correctly says "IID". The title was never updated to match the implemented model.
- The convergence comparison issue and the Takeaway timing contradiction are the same underlying problem: the tutorial declines to make a timing claim in the body but implicitly or explicitly makes one in the figure and the Takeaway. Both findings point to a single prose inconsistency about what the convergence experiment is actually measuring.
- The two DATA DRIFT findings (F3, F4) share a cause: the figure description strings use `.2f`/`.3f` and the table strings use `.4f` for the same variables. A single formatting convention for all in-text numbers would eliminate both.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 30%.** Below the 50% halt threshold; fix in place. No need to pause forward work.

1. **F1 (MISLABELED, HIGH) — fix first.** Turn `README.md:1` violated-invariant test into a pytest that PASSES on current file. Fix the title in `run.py:432` (and regenerate README) to remove "Persistent Income" — replace with "IID Income" or "Idiosyncratic Income". Confirm F1 honest-fix test now PASSES.

2. **F2 (DILUTED, MED).** Either (a) change `solve_vfi` to track consumption-level error (extract the policy from the value function at each iteration and measure `|c_n - c_{n-1}|` in consumption units) for a fair comparison, or (b) add a single sentence to `README.md:150` disclosing that VFI error is value-level while EEI/EGP error is consumption-level. The prose assertion ("needs more iterations") is not retracted but qualified. Confirm honest-fix test PASSES.

3. **F3+F4 (DATA DRIFT, LOW).** Standardize format strings: use `.4f` for all in-text numeric citations to match the table. Single find-replace pass in `run.py:662` and `run.py:665`. Regenerate README.

4. **F5 (DILUTED, LOW).** Remove "EGP is faster here" from `run.py:753` (Takeaway string) or replace with "EGP avoids the bisection step, saving wall-clock time per iteration" and remove the "not a timing claim" disclaimer from `run.py:692`, making the two statements consistent.

5. After fixes, regenerate `README.md` via `python run.py`, re-run `scripts/validate_catalog.py`, re-run this skill. Expected new score: 0-10% (title, metric disclosure, and prose consistency corrected; no structural code errors found).
