# bullshit-detector — aiyagari — 2026-05-20

**Bullshit score: 15%** — two DATA DRIFT findings (runtime counts not in CSV artifact; K* label implies cleared market while table shows K_d != K_s by 0.0034); all formula, algorithm, and numeric claims grounded against code and CSV.

## Header
- Claim sources: `dynamic-programming/aiyagari/README.md` (all sections)
- Code / artifact root: `dynamic-programming/aiyagari/run.py`
- Data artifacts: `dynamic-programming/aiyagari/tables/equilibrium.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "K^s(r*) = sum a_i mu(a_i,z_j) = K^d(r*)" (equilibrium clears exactly) | DATA DRIFT | LOW | no (within tolerance; table shows both sides are reported separately and differ by 0.0034) |
| 2 | "12 bisection steps" and "188 VFI iterations" | DATA DRIFT | LOW | no (runtime values injected dynamically; not stored in CSV; cannot ground without re-run) |
| 3 | CRRA utility, Bellman VFI, expected-value broadcasting | HOLDS | — | — |
| 4 | Stationary distribution forward-iteration operator | HOLDS | — | — |
| 5 | Capital demand K^d(r) = ((r+delta)/alpha)^(1/(alpha-1)) | HOLDS | — | — |
| 6 | Wage w(K) = (1-alpha)(K/L)^alpha | HOLDS | — | — |
| 7 | Bisection direction logic (K_s > K_d -> r_H = r) | HOLDS | — | — |
| 8 | Rouwenhorst grid, normalization E[z]=1 | HOLDS | — | — |
| 9 | All numeric results in diagnostics table (r*, w*, K*, Y*, K/Y, E[a], median, P90, Gini, constrained mass, gap) | HOLDS | — | — |
| 10 | "38% below complete-markets benchmark" | HOLDS | — | — |

## Findings

### Finding 1: Equilibrium equation implies K^s = K^d exactly; table shows two different values

- **Claim source (verbatim):** "The capital market clears: $K^s(r^{\ast}) = \sum_{i,j} a_i\,\mu(a_i,z_j) = K^d(r^{\ast})$." — `README.md:72`
- **Code evidence (verbatim):**
  ```python
  r_eq = r_trial
  K_eq = K_d          # line 243 — K_eq assigned from demand side
  w_eq = w_trial
  market_gap = K_s - K_d
  ```
  `run.py:242-245`
  ```python
  mean_wealth = K_s   # line 253 — mean_wealth is the supply side
  ```
  `run.py:253`
- **Data evidence:** `tables/equilibrium.csv`:
  ```
  Aggregate capital $K^{\ast}$,6.7599     <- K_d
  Mean wealth $\mathbb{E}[a]$,6.7633      <- K_s
  Relative market-clearing gap,+4.939e-04
  ```
  K_d = 6.7599, K_s = 6.7633. Absolute difference 0.0034; relative difference 0.050%.
- **Category:** DATA DRIFT — the equation asserts equality; the table presents both sides as distinct labeled values that differ.
- **Severity:** LOW — the gap is within the declared tolerance (5e-4 relative); no published number is fabricated. A careful reader comparing "K*=6.7599" and "E[a]=6.7633" in the same table will see the residual. The real issue is the equilibrium equation implying exact equality when the computation stops at a tolerance.
- **Result-changing:** no — the gap is the tolerance residual, not an error. But a reader who adds up the table rows expecting K* = E[a] will find a 0.05% discrepancy with no explanation in the prose.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(float(df.loc[df.Variable.str.contains("K.*ast"), "Value"].iloc[0]) - float(df.loc[df.Variable.str.contains("E\[a\]"), "Value"].iloc[0])) > 1e-4
  # PASSES on current code (they differ by 0.0034); FAILS if K_eq were set to K_s
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "K_eq = K_d" not in open("run.py").read() or "K^s(r^{\\ast}) \\approx K^d" in open("README.md").read()
  # PASSES on honest fix (either K_eq unified or equation changed to approximate); FAILS on current code
  ```

---

### Finding 2: "12 bisection steps" and "188 VFI iterations" not grounded in any CSV artifact

- **Claim source (verbatim):** "The run stops after **12** bisection steps, with relative gap **4.94e-04**." and "The final household VFI takes **188** iterations." — `README.md:99,116`
- **Code evidence (verbatim):**
  ```python
  f"low. The run stops after **{ge_iter}** bisection steps, with relative "
  f"gap **{abs(market_gap_rel):.2e}**.\n\n"
  ```
  `run.py:400-401`
  ```python
  f"The final household VFI takes **{sol['iterations']}** iterations. "
  ```
  `run.py:416`
- **Data evidence:** `tables/equilibrium.csv` stores 11 equilibrium quantities; no column for bisection step count or VFI iteration count. The relative gap `+4.939e-04` IS in the CSV (`Relative market-clearing gap`) and matches `4.94e-04` (different formatting, same value). The step counts 12 and 188 are injected from runtime variables `ge_iter` and `sol['iterations']` which are not persisted to any artifact.
- **Category:** DATA DRIFT — the gap value is grounded in the CSV; the two iteration counts are ungrounded runtime values that cannot be verified against committed artifacts without re-running.
- **Severity:** LOW — these are convergence diagnostics, not result-determining quantities. No other claim depends on them.
- **Result-changing:** no — no figure or economics conclusion depends on whether bisection took 12 or 15 steps.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "bisection_steps" not in pd.read_csv("tables/equilibrium.csv").columns
  # PASSES on current code (column absent); FAILS if step count were saved
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "bisection_steps" in pd.read_csv("tables/equilibrium.csv").columns or "needs re-run" in open("README.md").read()
  # PASSES if step counts are saved to CSV or README is annotated; FAILS on current code
  ```

## Cross-cutting patterns

- Both DATA DRIFT findings are benign: the gap residual is expected given the tolerance stopping rule, and the runtime counts are cosmetic diagnostics. No formula, algorithm, or economics claim is false or diluted.
- The `K_eq = K_d` assignment pattern (run.py:243) is intentional (standard to use demand-side capital for factor prices), but it silently makes "K*" in the table mean "K_d" rather than "market-clearing K." The table would be cleaner if K* = mean_wealth = K_s, or if K* were omitted and the gap row alone conveyed the residual.
- All formula claims checked: CRRA utility (run.py:30-35), Bellman VFI with feasibility masking (run.py:60-92), expected-value tensor contraction (run.py:71-73), Rouwenhorst chain construction (lib/discretize.py:24-48), forward-iteration distribution operator (run.py:105-115), capital demand (run.py:189-191), wage (run.py:193-194), Gini via trapezoidal Lorenz (run.py:135-139). All HOLD against the README equations.
- All numeric results in the diagnostics table (r*=0.025959, w*=1.2734, K*=6.7599, Y*=1.9897, K/Y=3.3975, E[a]=6.7633, median=4.4728, P90=16.3145, Gini=0.5261, constrained=0.0245, gap=+4.939e-04) match the CSV verbatim.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Below the 50% halt threshold. Forward work can proceed; surface the two findings as minor cleanup items.

1. Finding 1 (K_eq label): To make the table internally consistent, either (a) set `K_eq = K_s` after bisection and recompute `Y_eq` with K_s, or (b) change the equilibrium equation in the README to `K^s(r^{\ast}) \approx K^d(r^{\ast})` and add a prose note that the 0.05% residual is the tolerance gap. Option (a) is more faithful to the model object (equilibrium capital = household holdings = K_s); option (b) is more transparent about the numerical residual. Neither changes any published figure.

2. Finding 2 (ungrounded iteration counts): Extend the `summary` DataFrame in run.py to include bisection step count and final VFI iteration count as rows, so they are persisted in `tables/equilibrium.csv` and can be verified against committed artifacts.

3. After fixes, re-run `python run.py` inside the tutorial folder. Re-run `scripts/validate_catalog.py`. Re-run this skill on the updated code to confirm both findings now read HOLDS and the bullshit score drops to 0-10%.
