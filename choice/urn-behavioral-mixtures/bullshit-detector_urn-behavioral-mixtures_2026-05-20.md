# bullshit-detector — urn-behavioral-mixtures — 2026-05-20

**Bullshit score: 15%** — all formula, weight, and accuracy claims HOLD against code and CSV; three runtime-only scalar values (iteration count, log-likelihood, three separation-task counts) are embedded in the committed README but stored in no CSV artifact, so they cannot be re-grounded without a re-run (DATA DRIFT, LOW severity each).

## Header
- Claim sources: `choice/urn-behavioral-mixtures/README.md`
- Code / artifact root: `choice/urn-behavioral-mixtures/run.py`
- Data artifacts: `choice/urn-behavioral-mixtures/tables/mixture-weights.csv`, `choice/urn-behavioral-mixtures/tables/type-allocation.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, six-pass interrogation)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Lambda formula matches code | HOLDS | none | no |
| 2 | Bayesian posterior formula matches code | HOLDS | none | no |
| 3 | Finite-mixture likelihood matches code | HOLDS | none | no |
| 4 | E-step responsibility formula matches code | HOLDS | none | no |
| 5 | M-step w_m = mean_i tau_im matches code | HOLDS | none | no |
| 6 | Four rule definitions match code | HOLDS | none | no |
| 7 | Tremble is symmetric at 0.06 | HOLDS | none | no |
| 8 | k=3, n=5 crosses Bayes (0.5), not conservative (0.75) | HOLDS | none | no |
| 9 | L1 weight error = 0.028 matches CSV | HOLDS | none | no |
| 10 | Hard allocation accuracy = 0.998 matches CSV | HOLDS | none | no |
| 11 | Weight table in README matches tables/mixture-weights.csv | HOLDS | none | no |
| 12 | Confusion matrix in README matches tables/type-allocation.csv | HOLDS | none | no |
| 13 | EM converges in 6 iterations; log likelihood = -8816.71 | DATA DRIFT | LOW | no |
| 14 | Bayes-conservative separating tasks = 6 (Model Setup table) | DATA DRIFT | LOW | no |
| 15 | Bayes-share split = 4; Bayes-count split = 10 | DATA DRIFT | LOW | no |

## Findings

### Finding 1: Lambda formula (HOLDS)

- **Claim source (verbatim):** "$\Lambda(k,n) = \log \frac{\Pr(k\mid H,n)}{\Pr(k\mid L,n)} = k\log\frac{p_H}{p_L} + (n-k)\log\frac{1-p_H}{1-p_L}$" — `README.md:17-21`
- **Code evidence (verbatim):**
  ```python
  return (
      k_red * np.log(p_red_h / p_red_l)
      + (n_draws - k_red) * np.log((1.0 - p_red_h) / (1.0 - p_red_l))
  )
  ```
  `run.py:36-39`
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2: Bayesian posterior formula (HOLDS)

- **Claim source (verbatim):** "$\Pr(H\mid k,n) = \frac{1}{1+\exp[-\{\log(\pi_0/(1-\pi_0))+\Lambda(k,n)\}]}$" — `README.md:26-28`
- **Code evidence (verbatim):**
  ```python
  log_prior_odds = np.log(prior_h / (1.0 - prior_h))
  return expit(log_prior_odds + log_likelihood_ratio(k_red, n_draws, p_red_h, p_red_l))
  ```
  `run.py:50-51`; `expit(x) = 1/(1+exp(-x))` — exact match.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 3: Finite-mixture likelihood and E/M steps (HOLDS)

- **Claim source (verbatim):** "$\ell(w)=\sum_i \log\left[\sum_m w_m L_{im}\right]$", "$\tau_{im} = \frac{w_m L_{im}}{\sum_h w_h L_{ih}}$", M step: "w_m = mean_i tau_im" — `README.md:43-51, README.md:83`
- **Code evidence (verbatim):**
  ```python
  log_joint = log_like_by_rule + np.log(weights)[None, :]
  log_den = logsumexp(log_joint, axis=1)
  responsibilities = np.exp(log_joint - log_den[:, None])
  weights = responsibilities.mean(axis=0)
  log_likelihood = float(np.sum(log_den))
  ```
  `run.py:131-135`
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 4: Four rule definitions (HOLDS)

- **Claim source (verbatim):**
  - "Choose high if the posterior probability of the high urn is at least one half." — `README.md:115`
  - "Choose high only when the posterior probability of the high urn is at least 0.75." — `README.md:116`
  - "Choose high when at least half of sampled balls are red." — `README.md:117`
  - "Choose high when at least four sampled balls are red, ignoring sample size." — `README.md:118`
- **Code evidence (verbatim):**
  ```python
  choices[0] = posterior >= 0.5
  choices[1] = posterior >= 0.75
  choices[2] = k_red / n_draws >= 0.5
  choices[3] = k_red >= 4
  ```
  `run.py:64-67`
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 5: L1 weight error = 0.028 (HOLDS)

- **Claim source (verbatim):** "The L1 distance between estimated and true weights is **0.028**." — `README.md:97`
- **Code evidence (verbatim):**
  ```python
  weight_l1 = float(np.sum(np.abs(weights_hat - true_weights)))
  ```
  `run.py:190`; formatted via `f'{weight_l1:.3f}'` at `run.py:338`.
- **Data evidence (verbatim):** `tables/mixture-weights.csv:2-5`
  ```
  Bayes,0.46,0.4575,-0.0025
  Conservative,0.24,0.2523,0.0123
  Share cutoff,0.2,0.2019,0.0019
  Count cutoff,0.1,0.0883,-0.0117
  ```
  Sum of absolute errors: 0.0025 + 0.0123 + 0.0019 + 0.0117 = 0.0284; `f'{0.0284:.3f}'` = `'0.028'`.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 6: Hard allocation accuracy = 0.998 (HOLDS)

- **Claim source (verbatim):** "Hard allocation accuracy is **0.998**." — `README.md:103`
- **Code evidence (verbatim):**
  ```python
  type_accuracy = float(np.mean(assigned == panel["type_id"]))
  ```
  `run.py:189`; formatted via `f'{type_accuracy:.3f}'` at `run.py:357`.
- **Data evidence (verbatim):** `tables/type-allocation.csv:2-5`
  ```
  Bayes,275,0,0,0
  Conservative,0,151,0,0
  Share cutoff,1,0,120,0
  Count cutoff,0,0,0,53
  ```
  Diagonal = 275+151+120+53 = 599; total = 600 (one misclassification in Share cutoff row); 599/600 = 0.9983; `f'{0.9983:.3f}'` = `'0.998'`.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 7: k=3, n=5 crosses Bayes but not conservative threshold (HOLDS)

- **Claim source (verbatim):** "With 5 draws, a count of three red balls crosses the Bayes threshold. It does not cross the conservative cutoff." — `README.md:91-92`
- **Code evidence:** Computed analytically from `run.py:50-51` with `prior_h=0.45`, `p_red_h=0.72`, `p_red_l=0.32`, `k=3`, `n=5`: posterior = 0.6124 > 0.5 (Bayes threshold crossed); 0.6124 < 0.75 (conservative threshold not crossed).
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 8: EM iteration count and log-likelihood are runtime-only (DATA DRIFT)

- **Claim source (verbatim):** "EM converges in 6 iterations; log likelihood is -8816.71." — `README.md:109`
- **Code evidence (verbatim):**
  ```python
  description=(
      f"EM converges in {estimates['iterations']} iterations; log likelihood is "
      f"{float(estimates['log_likelihood']):.2f}."
  ),
  ```
  `run.py:382-384`
- **Data evidence:** Neither `tables/mixture-weights.csv` nor `tables/type-allocation.csv` stores iteration count or log-likelihood. — `tables/mixture-weights.csv:1` (columns: Rule, Definition, True weight, Estimated weight, Error; no iterations/log-likelihood column).
- **Category:** DATA DRIFT — runtime-only values baked into README but not committed to any CSV artifact; cannot be re-grounded without re-run. Values are deterministic under `np.random.default_rng(1234)` (`run.py:161`) so drift only occurs if seed or parameters change; qualitative lesson (EM converges quickly) is unaffected.
- **Severity:** LOW
- **Result-changing:** no — needs re-run to verify exact values.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "iterations" not in open("choice/urn-behavioral-mixtures/tables/mixture-weights.csv").readline()  # PASSES on current code (column absent), FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "iterations" in pd.read_csv("choice/urn-behavioral-mixtures/tables/diagnostics.csv").columns  # PASSES on honest fix, FAILS currently
  ```

### Finding 9: Separation-task counts (6, 4, 10) are runtime-only (DATA DRIFT)

- **Claim source (verbatim):** "Bayes-conservative separating tasks | 6" — `README.md:66`; "Bayes differs from the conservative rule on 6 tasks. It differs from the red-share rule on 4 tasks and the raw-count rule on 10 tasks." — `README.md:103`
- **Code evidence (verbatim):**
  ```python
  bayes_conservative_split = int(np.sum(rule_choices[0] != rule_choices[1]))
  bayes_share_split = int(np.sum(rule_choices[0] != rule_choices[2]))
  bayes_count_split = int(np.sum(rule_choices[0] != rule_choices[3]))
  ```
  `run.py:192-194`; injected into report at `run.py:358-359`.
- **Data evidence:** These three integers appear nowhere in the committed CSVs. — `tables/mixture-weights.csv:1`, `tables/type-allocation.csv:1` (neither stores separation counts).
- **Category:** DATA DRIFT — same structural gap as Finding 8. Deterministic under fixed seed; unverifiable from committed artifacts.
- **Severity:** LOW
- **Result-changing:** no — needs re-run to verify exact counts.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "bayes_conservative_split" not in open("choice/urn-behavioral-mixtures/tables/mixture-weights.csv").read()  # PASSES on current code, FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert pd.read_csv("choice/urn-behavioral-mixtures/tables/diagnostics.csv")["bayes_conservative_split"].iloc[0] == 6  # PASSES on honest fix, FAILS currently
  ```

## Cross-cutting patterns

- All formula-level claims (Lambda, posterior, mixture likelihood, E-step, M-step) are faithfully implemented with no parametric leakage. The tutorial passes the Iron Law for its mathematical content.
- The only gap is structural: five scalar diagnostic values (EM iteration count, log-likelihood, and three rule-separation counts) are computed at runtime and embedded in the README via f-strings (`run.py:357-359`, `run.py:382-384`) but never written to any CSV artifact. If parameters or seed change, the README numbers will silently drift. This is a documentation-hygiene gap, not a math error.
- The EM implementation correctly uses TRUE rule probabilities (computed from true parameters `p_red_h`, `p_red_l`, `prior_h`) as fixed inputs, consistent with the El-Gamal/Grether design where rule shapes are known and only mixture weights are estimated. The README makes no false claim of estimating rule parameters.
- Both DATA DRIFT findings share the same root cause: the `report.add_table()` description string (`run.py:382-384`) and the `report.add_results()` call (`run.py:357-359`) embed runtime values that are never persisted to CSV.

## TDD execution sequence (for the next agent)

0. **Score is 15% — below the 50% halt threshold.** Forward work can continue. DATA DRIFT findings are a documentation-hygiene task, not a correctness fix.

1. For Findings 8 and 9: the violated-invariant tests confirm that `tables/mixture-weights.csv` and `tables/type-allocation.csv` do not store iteration count, log-likelihood, or separation counts. Run these tests to confirm they PASS on current code (proving the gap is real).

2. Honest-fix: add a `tables/diagnostics.csv` written by `run.py` storing `iterations`, `log_likelihood`, `bayes_conservative_split`, `bayes_share_split`, `bayes_count_split`. Pass-condition tests will then PASS.

3. No mathematical fix required. All formula and weight claims HOLD.

4. After adding the diagnostics artifact, re-run `python scripts/validate_catalog.py` to confirm no regressions.

5. Re-run this skill on the updated code to confirm all findings read HOLDS and the score drops to 0-10%.
