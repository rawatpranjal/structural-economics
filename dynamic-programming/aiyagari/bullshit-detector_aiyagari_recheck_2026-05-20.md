# bullshit-detector — aiyagari — recheck — 2026-05-20

**Bullshit score: 0%** — Both prior DATA DRIFT findings resolved: F1 (equilibrium equation now states approximate equality with explicit tolerance note) and F2 (bisection steps and VFI iteration counts now persisted as CSV rows and verified to match README prose). All formula, algorithm, and numeric claims HOLD. No new findings.

## Header
- Claim sources: `dynamic-programming/aiyagari/README.md` (all sections)
- Code / artifact root: `dynamic-programming/aiyagari/run.py`
- Data artifacts: `dynamic-programming/aiyagari/tables/equilibrium.csv`
- Seed audit: `bullshit-detector_aiyagari_2026-05-20.md` (score 15%, 2 DATA DRIFT findings)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| F1 (prior) | Exact equality K^s = K^d → now approximate with tolerance note | HOLDS | — | — |
| F2 (prior) | Step counts ungrounded → now persisted in CSV and verified | HOLDS | — | — |
| 1 | CRRA utility, Bellman VFI, expected-value broadcasting | HOLDS | — | — |
| 2 | Stationary distribution forward-iteration operator | HOLDS | — | — |
| 3 | Capital demand K^d(r) = ((r+delta)/alpha)^(1/(alpha-1)) | HOLDS | — | — |
| 4 | Wage w(K) = (1-alpha)(K/L)^alpha | HOLDS | — | — |
| 5 | Bisection direction logic (K_s > K_d -> r_H = r) | HOLDS | — | — |
| 6 | Rouwenhorst grid, normalization E[z]=1 | HOLDS | — | — |
| 7 | All 13 numeric cells in diagnostics table | HOLDS | — | — |
| 8 | "38% below complete-markets benchmark" | HOLDS | — | — |

## Findings

### Prior Finding 1 — RESOLVED: equilibrium equation now approximate

- **Prior claim (verbatim, buggy):** "The capital market clears: $K^s(r^{\ast}) = \sum_{i,j} a_i\,\mu(a_i,z_j) = K^d(r^{\ast})$." — original `README.md:72`
- **Current README evidence (verbatim):**
  ```
  Bisection stops once the relative gap falls below tolerance, so the run delivers
  K^s(r^{\ast}) \approx K^d(r^{\ast}) rather than exact equality. The diagnostics
  table reports both sides; the small residual between them is the tolerance gap,
  not a model object.
  ```
  `README.md:75-78`
- **Code evidence:** `run.py:242-245`: `r_eq = r_trial; K_eq = K_d; w_eq = w_trial; market_gap = K_s - K_d`. The demand-side `K_eq` assignment is unchanged (correct: factor prices use demand-side capital), but the README now explicitly states the approximation. The prose notes the residual is "the tolerance gap, not a model object." `README.md:138`: "The market-clearing gap is the bisection residual. It is numerical error, not a model object." ✓
- **Data evidence:** `tables/equilibrium.csv`: `Aggregate capital $K^{\ast}$,6.7599` (K_d) and `Mean wealth $\mathbb{E}[a]$,6.7633` (K_s). Absolute difference = 0.0034; relative = 0.050%. Both sides still reported distinctly. The diagnostics table includes `Relative market-clearing gap,+4.939e-04`. ✓
- **Test:** `test_finding1_violated_invariant_equation_states_exact_equality` FAILS (correct post-fix: `\,\mu(a_i,z_j) = K^d(r^{\ast})` no longer present). `test_finding1_honest_fix_equation_is_approximate` PASSES: `K^s(r^{\ast}) \approx K^d(r^{\ast})` and "tolerance" both present. `test_finding1_table_reports_both_sides_distinctly` PASSES: `abs(6.7599 - 6.7633) = 0.0034 > 1e-4`. ✓
- **Category:** HOLDS

---

### Prior Finding 2 — RESOLVED: step counts persisted to CSV

- **Prior claim (verbatim, buggy):** `tables/equilibrium.csv` had no column for bisection step count or VFI iteration count. README prose injected `ge_iter` and `sol['iterations']` dynamically; no committed artifact grounded them.
- **Current CSV evidence (verbatim):**
  ```
  Bisection steps,12
  VFI iterations,188
  ```
  `tables/equilibrium.csv:13-14`
- **Current README evidence (verbatim):** "The run stops after **12** bisection steps, with relative gap **4.94e-04**." — `README.md:105`. "The final household VFI takes **188** iterations." — `README.md:122`. Both match CSV rows exactly. ✓
- **Test:** `test_finding2_violated_invariant_counts_absent_from_csv` FAILS (correct post-fix: "Bisection steps" and "VFI iterations" now present in CSV). `test_finding2_honest_fix_counts_persisted_to_csv` PASSES: both variable names found in CSV. `test_finding2_readme_counts_match_csv` PASSES: `**12** bisection steps` and `**188** iterations` present in README. ✓
- **Category:** HOLDS

---

### Finding 1: CRRA utility and Bellman VFI — HOLDS

- **Claim source (verbatim):** "$u(c)=\frac{c^{1-\sigma}}{1-\sigma}$" and Bellman equation — `README.md:22-24`, `README.md:37-41`
- **Code evidence:** `run.py:30-35`: `c_safe ** (1.0 - sigma) / (1.0 - sigma)` with log special case; `run.py:57-92`: vectorized VFI with feasibility masking. ✓
- **Category:** HOLDS

---

### Finding 2: Stationary distribution operator — HOLDS

- **Claim source (verbatim):** "$\mu(a',z_k) = \sum_j P_{jk}\sum_{i:\,g_a(a_i,z_j)=a'} \mu(a_i,z_j)$" — `README.md:51-53`
- **Code evidence:** `run.py:105-115`: forward iteration over `(policy_idx, transition)`. Operator matches claim. ✓
- **Category:** HOLDS

---

### Finding 3: Capital demand formula — HOLDS

- **Claim source (verbatim):** "$K^d(r) = (\tfrac{r+\delta}{\alpha})^{1/(\alpha-1)}$" — `README.md:65`
- **Code evidence:** `run.py:189-191`: `capital_demand` function. Formula matches. ✓
- **Category:** HOLDS

---

### Finding 4: Wage formula — HOLDS

- **Claim source (verbatim):** "$w(K) = (1-\alpha)(\tfrac{K}{L})^{\alpha}$" — `README.md:62`
- **Code evidence:** `run.py:193-194`: `wage` function. Formula matches. ✓
- **Category:** HOLDS

---

### Finding 5: Bisection direction logic — HOLDS

- **Claim source (verbatim):** "if K^s(r) > K^d(r): set r_H = r / else: set r_L = r" — `README.md:118-119` pseudocode
- **Code evidence:** `run.py:228-232`: `if K_s > K_d: r_high = r_trial / else: r_low = r_trial`. Exact match. ✓
- **Category:** HOLDS

---

### Finding 6: Rouwenhorst grid normalization — HOLDS

- **Claim source (verbatim):** "normalized so $\mathbb{E}[z]=1$" — `README.md:34`
- **Code evidence:** `lib/discretize.py`: Rouwenhorst chain construction with normalization. ✓
- **Category:** HOLDS

---

### Finding 7: All 13 numeric cells in diagnostics table — HOLDS

- **Claim source:** `README.md:142-156` and `tables/equilibrium.csv:1-14`
- **Verification:** All 13 rows match between README table and CSV verbatim (r*=0.025959, w*=1.2734, K*=6.7599, Y*=1.9897, K/Y=3.3975, E[a]=6.7633, median=4.4728, P90=16.3145, Gini=0.5261, constrained=0.0245, gap=+4.939e-04, bisection=12, VFI=188). The two new rows (Bisection steps, VFI iterations) are present in both artifacts. ✓
- **Category:** HOLDS

---

### Finding 8: "38% below complete-markets benchmark" — HOLDS

- **Claim source (verbatim):** "roughly 38% below the complete-markets benchmark" — `README.md:126`
- **Verification:** Complete-markets ceiling `1/beta-1 = 1/0.96 - 1 = 0.04167`. Equilibrium `r*=0.025959`. Discount: `(0.04167 - 0.025959) / 0.04167 = 37.7%`, rounds to "roughly 38%". ✓
- **Category:** HOLDS

## Cross-cutting patterns

- Both prior DATA DRIFT findings are resolved. F1 fix: changed the chained equality in the Equations equilibrium block to an approximate relation with a prose note about the tolerance residual appearing in two places (Equations and Results). F2 fix: extended the `summary` DataFrame in `run.py` to include bisection step count and VFI iteration count as rows, persisting them to `tables/equilibrium.csv`.
- The K_eq = K_d assignment at `run.py:243` is unchanged and correct: factor prices are computed from demand-side capital. The README now explains why K* (=K_d=6.7599) and E[a] (=K_s=6.7633) differ: the tolerance residual.
- All formula derivations (CRRA, Bellman, stationary distribution, capital demand, wage) remain algebraically consistent between Equations and run.py. No formula drift.
- The 13-cell diagnostics table is consistent end-to-end: code output → CSV → README. No data drift between any two artifact layers.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No halt trigger. No further remediation needed.
1. Both violated-invariant tests fail post-fix (correct green state): `test_finding1_violated_invariant_equation_states_exact_equality` (exact equality string absent) and `test_finding2_violated_invariant_counts_absent_from_csv` (counts now present). Three honest-fix tests pass: `test_finding1_honest_fix_equation_is_approximate`, `test_finding1_table_reports_both_sides_distinctly`, `test_finding2_honest_fix_counts_persisted_to_csv`, `test_finding2_readme_counts_match_csv`. Full green state confirmed.
2. No further action required for this tutorial.
