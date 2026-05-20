# bullshit-detector — metropolis-hastings — recheck — 2026-05-20

**Bullshit score: 0%** — the original DATA DRIFT finding is resolved; the MH-draws row now reads "20,000 total | 19,000 retained after burn-in of 1,000" and is generated directly from code variables; all other claims hold against code and CSV artifacts.

## Header
- Claim sources: `computational-methods/metropolis-hastings/README.md`
- Code / artifact root: `computational-methods/metropolis-hastings/run.py`
- Data artifacts: `computational-methods/metropolis-hastings/tables/conjugate-summary.csv`, `computational-methods/metropolis-hastings/tables/proposal-comparison.csv`
- Seed audit (if any): `computational-methods/metropolis-hastings/bullshit-detector_metropolis-hastings_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | MH draws row: "20,000 total, 19,000 retained after burn-in of 1,000" | HOLDS | — | — |
| 2 | Solution Method prose matches retained count (19,000) | HOLDS | — | — |
| 3 | Figure caption matches retained count (19,000) | HOLDS | — | — |
| 4 | Beta-Binomial posterior: Beta(16, 8), mean 0.6667, variance 0.00889 | HOLDS | — | — |
| 5 | TRUE_COV[0,0] = 3.25 (marginal variance claim) | HOLDS | — | — |
| 6 | Conjugate-summary CSV values match README table | HOLDS | — | — |
| 7 | Proposal-comparison CSV values match README table | HOLDS | — | — |
| 8 | MH acceptance ratio: min(1, kernel ratio), log form | HOLDS | — | — |
| 9 | 1D bounded chain rejects proposals outside (0,1) | HOLDS | — | — |
| 10 | ESS truncated at first nonpositive lag | HOLDS | — | — |

## Findings

### Finding 1 (original): MH-draws row implied 20,000 retained — NOW HOLDS

- **Claim source (verbatim):** `"| MH draws | 20,000 total | 19,000 retained after burn-in of 1,000 |"` — `README.md:167`
- **Code evidence (verbatim):**
  ```python
  f"| MH draws | {n_conj_draws:,} total | {n_conj_draws - burn_conj:,} retained after burn-in of {burn_conj:,} |\n"
  ```
  `run.py:449`
  With `n_conj_draws = 20_000` (`run.py:225`) and `burn_conj = 1_000` (`run.py:226`), the generated string is `"| MH draws | 20,000 total | 19,000 retained after burn-in of 1,000 |"`.
- **Category:** HOLDS — the row is now generated from code variables and correctly states both the total draw count (20,000) and the retained count (19,000). The ambiguity present in the original README (where the row read `"20,000 | After burn-in of 1,000"`) is resolved. The Solution Method prose at `README.md:196` (`"After 1,000 burn-in draws and 19,000 retained draws"`) and the figure label at `run.py:531` (`"MH histogram (19,000 draws)"`) are now internally consistent with the Model Setup row.
- **Original finding resolved:** yes.

### Numeric cross-checks (new this pass)

**conjugate-summary.csv vs README.md:232-236:**

| Quantity | CSV | README | Match |
|----------|-----|--------|-------|
| Posterior mean: Analytical | 0.6667 | 0.6667 | ✓ |
| Posterior mean: MH empirical | 0.6661 | 0.6661 | ✓ |
| Posterior mean: Error | 0.0005 | 0.0005 | ✓ |
| Posterior variance: Analytical | 0.00889 | 0.00889 | ✓ |
| Posterior variance: MH empirical | 0.00901 | 0.00901 | ✓ |
| Posterior variance: Error | 0.00012 | 0.00012 | ✓ |
| P(theta > 0.5): Analytical | 0.9534 | 0.9534 | ✓ |
| P(theta > 0.5): MH empirical | 0.9517 | 0.9517 | ✓ |
| P(theta > 0.5): Error | 0.0017 | 0.0017 | ✓ |

**proposal-comparison.csv vs README.md:254-258:**

| Step | Accept | Switches | Error | ESS1 | ESS2 | Match |
|------|--------|----------|-------|------|------|-------|
| 0.15 | 0.918 | 71 | 0.374 | 24 | 23 | ✓ |
| 0.6 | 0.699 | 319 | 0.255 | 120 | 118 | ✓ |
| 2.0 | 0.304 | 689 | 0.048 | 467 | 494 | ✓ |

No drift between any artifact and the README. All values match exactly.

**Analytical cross-check:**
- `ALPHA_POST = 2 + 14 = 16`, `BETA_POST = 2 + 20 - 14 = 8` → `run.py:34-35` ✓
- `POST_MEAN = 16/24 = 0.6667` → `run.py:36` ✓
- `POST_VAR = 16*8 / (24² * 25) = 128/14400 = 0.00889` → `run.py:37-39` ✓
- `TRUE_COV = SIGMA + 0.5*0.5*outer(MU1-MU2, MU1-MU2)` = `[[1,0.5],[0.5,1]] + 0.25*[[9,9],[9,9]]` = `[[3.25,2.75],[2.75,3.25]]` → `TRUE_COV[0,0] = 3.25` → `run.py:106` ✓; README claims "marginal variance 3.25" (`README.md:250`) ✓

## Cross-cutting patterns

- The fix converted the Model Setup row string from a literal to an f-string computed from `n_conj_draws` and `burn_conj`. All three locations that state the retained count (Model Setup row, Solution Method prose, figure label) now derive from the same source variables and are guaranteed to stay consistent across future changes to `n_conj_draws` or `burn_conj`.
- No new gaps found. The algorithmic claims (acceptance ratio form, bounded rejection, ESS truncation), the parameter values, and all numeric results in both CSVs hold against code.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required. Ship.
1. The honest-fix test `test_f1_honest_fix_readme_row_disambiguates_total_vs_retained` passes: the README no longer contains `"20,000 | After burn-in of 1,000"` and does contain `"20,000 total"` and `"19,000 retained"`.
2. No further action needed for this tutorial.
