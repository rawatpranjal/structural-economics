# bullshit-detector — urn-behavioral-mixtures — recheck — 2026-05-20

**Bullshit score: 0%** — original Findings 8 and 9 (DATA DRIFT, LOW: runtime scalars unbacked by any CSV artifact) resolved by adding `tables/diagnostics.csv`; all formula, weight, accuracy, and separation-count claims verified HOLDS against code and all three CSV artifacts.

## Header
- Claim sources: `choice/urn-behavioral-mixtures/README.md`
- Code / artifact root: `choice/urn-behavioral-mixtures/run.py`
- Data artifacts: `choice/urn-behavioral-mixtures/tables/mixture-weights.csv`, `choice/urn-behavioral-mixtures/tables/type-allocation.csv`, `choice/urn-behavioral-mixtures/tables/diagnostics.csv`
- Seed audit: `bullshit-detector_urn-behavioral-mixtures_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Lambda formula matches code | HOLDS | — | no |
| 2 | Bayesian posterior formula matches code | HOLDS | — | no |
| 3 | Finite-mixture likelihood matches code | HOLDS | — | no |
| 4 | E-step responsibility formula matches code | HOLDS | — | no |
| 5 | M-step w_m = mean_i tau_im matches code | HOLDS | — | no |
| 6 | Four rule definitions match code | HOLDS | — | no |
| 7 | Tremble symmetric at 0.06 | HOLDS | — | no |
| 8 | k=3, n=5 crosses Bayes (0.5), not conservative (0.75) | HOLDS | — | no |
| 9 | L1 weight error = 0.028 matches CSV | HOLDS | — | no |
| 10 | Hard allocation accuracy = 0.998 matches CSV | HOLDS | — | no |
| 11 | Weight table in README matches mixture-weights.csv | HOLDS | — | no |
| 12 | Confusion matrix in README matches type-allocation.csv | HOLDS | — | no |
| 13 | EM converges in 6 iterations; log likelihood = -8816.71 | HOLDS (fixed) | — | no |
| 14 | Bayes-conservative separating tasks = 6 | HOLDS (fixed) | — | no |
| 15 | Bayes-share split = 4; Bayes-count split = 10 | HOLDS (fixed) | — | no |

## Findings

### Finding 1 (original F8): DATA DRIFT — EM iteration count and log-likelihood unbacked — RESOLVED

- **Original gap:** `estimates['iterations']` (6) and `estimates['log_likelihood']` (-8816.71) appeared only as README prose strings baked in by f-string at `run.py:382-384`. No CSV artifact stored them.

- **Current code evidence:**
  ```python
  # run.py:396-413
  diagnostics = pd.DataFrame(
      {
          "iterations": [int(estimates["iterations"])],
          "log_likelihood": [float(estimates["log_likelihood"])],
          "bayes_conservative_split": [bayes_conservative_split],
          "bayes_share_split": [bayes_share_split],
          "bayes_count_split": [bayes_count_split],
      }
  )
  report.add_table(
      "tables/diagnostics.csv",
      "EM and rule-separation diagnostics",
      diagnostics.round({"log_likelihood": 2}),
      ...
  )
  ```

- **Data evidence:** `tables/diagnostics.csv:1-2`:
  ```
  iterations,log_likelihood,bayes_conservative_split,bayes_share_split,bayes_count_split
  6,-8816.71,6,4,10
  ```

- **Cross-check:** `README.md:109` — "EM converges in 6 iterations; log likelihood is -8816.71." Matches `diagnostics.csv` iteration count (6) and log-likelihood (-8816.71) exactly.

- **Resolution:** `diagnostics.csv` now stores both values. The README prose and the CSV artifact agree. Finding fully resolved.

- **Category:** HOLDS (post-fix)

### Finding 2 (original F9): DATA DRIFT — rule-separation counts unbacked — RESOLVED

- **Original gap:** `bayes_conservative_split` (6), `bayes_share_split` (4), `bayes_count_split` (10) appeared in `README.md:103` via f-string injection at `run.py:357-359`. Neither `mixture-weights.csv` nor `type-allocation.csv` stored them.

- **Current code evidence:** `run.py:396-403` — same `diagnostics` DataFrame as above. All three split counts are column entries in `diagnostics.csv`.

- **Data evidence:** `tables/diagnostics.csv:2` — `bayes_conservative_split,6`, `bayes_share_split,4`, `bayes_count_split,10`. `README.md:66` — "6" (Model Setup table). `README.md:103` — "6 tasks", "4 tasks", "10 tasks". All three sources agree.

- **Resolution:** All three separation counts are now persisted in `diagnostics.csv`. Every runtime scalar cited in the README has a backing artifact. Finding fully resolved.

- **Category:** HOLDS (post-fix)

## Grounded HOLDS findings (key claims verified fresh)

**H1 — Lambda formula.** `README.md:17-21` — `Λ(k,n)=k·log(pH/pL)+(n-k)·log((1-pH)/(1-pL))`. Code `run.py:36-39` — `k_red * np.log(p_red_h / p_red_l) + (n_draws - k_red) * np.log((1.0 - p_red_h) / (1.0 - p_red_l))`. Exact match. HOLDS.

**H2 — Bayesian posterior.** `README.md:26-28` — `Pr(H|k,n)=1/(1+exp[-{log(π₀/(1-π₀))+Λ}])`. Code `run.py:50-51` — `expit(log_prior_odds + log_likelihood_ratio(...))`. `expit(x)=1/(1+exp(-x))`. Exact match. HOLDS.

**H3 — Finite-mixture likelihood and EM.** `README.md:43-51,83`. Code `run.py:131-135`:
```python
log_joint = log_like_by_rule + np.log(weights)[None, :]
log_den = logsumexp(log_joint, axis=1)
responsibilities = np.exp(log_joint - log_den[:, None])
weights = responsibilities.mean(axis=0)
log_likelihood = float(np.sum(log_den))
```
E-step computes `τᵢₘ` via numerically stable log-space; M-step is `mean(axis=0)`. Exact match. HOLDS.

**H4 — Four rule definitions.** `README.md:115-118` — Bayes≥0.5, Conservative≥0.75, share≥0.5, count≥4. Code `run.py:64-67`:
```python
choices[0] = posterior >= 0.5
choices[1] = posterior >= 0.75
choices[2] = k_red / n_draws >= 0.5
choices[3] = k_red >= 4
```
Exact match. HOLDS.

**H5 — L1 weight error = 0.028.** `README.md:97`. Code `run.py:190` — `weight_l1 = float(np.sum(np.abs(weights_hat - true_weights)))`. CSV sum: |−0.0025|+|0.0123|+|0.0019|+|−0.0117|=0.0284; `f'{0.0284:.3f}'`=`'0.028'`. HOLDS.

**H6 — Hard allocation accuracy = 0.998.** `README.md:103`. Code `run.py:189` — `type_accuracy = float(np.mean(assigned == panel["type_id"]))`. CSV diagonal: 275+151+120+53=599 of 600; 0.9983→`0.998`. HOLDS.

**H7 — Separation counts backed by diagnostics.csv.** All three counts (6, 4, 10) in `README.md:66,103` match `diagnostics.csv:2` exactly. `run.py:192-194` computes them from `rule_choices` with the correct indices. HOLDS.

**H8 — EM iteration and LL backed by diagnostics.csv.** `README.md:109` — "6 iterations", "-8816.71". `diagnostics.csv:2` — `6,-8816.71`. Match. HOLDS.

## Cross-cutting patterns

- All three original DATA DRIFT findings share the same root cause and the same fix: a single new `tables/diagnostics.csv` written by `run.py:396-413` now backs every runtime scalar cited in README prose.
- The fix does not change any formula, weight, or accuracy claim. Published numbers are identical to the original audit.
- No new faithfulness gaps identified in this recheck pass. The tutorial passes the Iron Law for all mathematical, algorithmic, and numeric content.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings read HOLDS. No further fixes required.
1. Test suite status: `test_f8_violated_invariant_weights_csv_lacks_iteration_columns` PASSES (correct — `mixture-weights.csv` still lacks `iterations` column; the fix added a separate file); `test_f9_violated_invariant_csvs_lack_split_columns` PASSES (correct — split columns absent from old CSVs); `test_f8_honest_fix_diagnostics_csv_has_em_scalars` PASSES (correct — `diagnostics.csv` exists with matching iteration count and log-likelihood); `test_f9_honest_fix_diagnostics_csv_has_split_counts` PASSES (correct — all three split counts present and match README).
2. No mathematical or data-artifact changes required beyond the already-committed `diagnostics.csv`.
</content>
