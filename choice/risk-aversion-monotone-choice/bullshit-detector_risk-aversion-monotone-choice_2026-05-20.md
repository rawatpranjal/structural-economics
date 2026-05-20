# bullshit-detector — risk-aversion-monotone-choice — 2026-05-20

**Bullshit score: 10%** — one DATA DRIFT finding (inconsistent violation-counting threshold across estimators in the same summary column); all numeric claims ground against CSV and recomputation; no FALSE, DILUTED, MISLABELED, or UNIMPLEMENTED findings.

## Header
- Claim sources: `choice/risk-aversion-monotone-choice/README.md` (prose, Equations, Results, tables)
- Code / artifact root: `choice/risk-aversion-monotone-choice/run.py`
- Data artifacts: `choice/risk-aversion-monotone-choice/tables/model-comparison.csv`, `choice/risk-aversion-monotone-choice/tables/row-fits.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Payoffs A=(2.00,1.60), B=(3.85,0.10) | HOLDS | — | no |
| 2 | CRRA utility formula exact | HOLDS | — | no |
| 3 | Fixed-scale logit Pr=lambda+(1-2lambda)*sigma(s*DeltaEU) | HOLDS | — | no |
| 4 | Monotone constraint alpha_{j+1}>=alpha_j via SLSQP ineq | HOLDS | — | no |
| 5 | DGP: rho=0.45, scale=5.00, lapse=0.02, 10 rows, 80 trials | HOLDS | — | no |
| 6 | Estimated rho=0.451 (README); 0.45074 (CSV) | HOLDS | — | no |
| 7 | True probs column matches structural_probabilities recomputation | HOLDS | — | no |
| 8 | Simulated counts reproducible at seed=1 | HOLDS | — | no |
| 9 | Unconstrained fit = observed shares | HOLDS | — | no |
| 10 | LL: unconstrained=-215.051, CRRA=-221.83, monotone=-216.044 | HOLDS | — | no |
| 11 | LL loss monotone vs saturated = 0.99 | HOLDS | — | no |
| 12 | Monotone fit pools rows 2-3 (both 0.0313) | HOLDS | — | no |
| 13 | "Monotonicity violations" column uses different thresholds per estimator | DATA DRIFT | LOW | no |

## Findings

### Finding 1: DATA DRIFT — violations threshold inconsistency

- **Claim source (verbatim):** Column header "Monotonicity violations" appears for all three estimators in the Estimator comparison table, implying a uniform measurement. — `README.md:131-135`

- **Code evidence (verbatim):**
  ```python
  # estimate_unconstrained_logits
  "violations": float(np.sum(np.diff(shares) < -1e-12)),
  ```
  `run.py:58`
  ```python
  # estimate_monotone_logits
  "violations": float(np.sum(np.diff(probabilities) < -1e-10)),
  ```
  `run.py:109`

- **Data evidence:** Both rows contribute to the same "Monotonicity violations" column. `tables/model-comparison.csv:2-4` — values reported as 1, 0, 0.

- **Category:** DATA DRIFT — two helper functions feeding the same summary column apply different sign thresholds (-1e-12 vs -1e-10). The reported numbers are unaffected: true violations are of magnitude ~0.04, far above either threshold; the constrained optimizer enforces monotonicity to within ftol=1e-10. But the measurement definition is internally inconsistent.

- **Severity:** LOW

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  import inspect; src = open("run.py").read(); assert src.count("< -1e-12") == 1 and src.count("< -1e-10") == 1
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  import inspect; src = open("run.py").read(); assert src.count("< -1e-12") + src.count("< -1e-10") <= 1
  ```

## Grounded HOLDS findings

**H1 — Payoff parameters.**
Claim: `README.md:18` — `A(p)=(2.00 with p, 1.60 otherwise), B(p)=(3.85 with p, 0.10 otherwise)`.
Code: `run.py:26-27` — `a_high, a_low = 2.00, 1.60` / `b_high, b_low = 3.85, 0.10`. Exact match.

**H2 — CRRA utility.**
Claim: `README.md:22-24` — `u(c;rho)=(c^{1-rho}-1)/(1-rho)`.
Code: `run.py:21` — `(np.asarray(x) ** (1.0 - rho) - 1.0) / (1.0 - rho)`. Exact match.

**H3 — Fixed-scale logit.**
Claim: `README.md:35-39` — `Pr(d=1|p;rho)=lambda+(1-2lambda)*1/(1+exp[-s*DeltaEU])`.
Code: `run.py:40-41` — `latent = scale * lottery_eu_difference(prob_high, rho)` / `return lapse + (1.0 - 2.0 * lapse) * expit(latent)`. `expit(x)=1/(1+exp(-x))`. Exact match.

**H4 — Monotone constraint.**
Claim: `README.md:51-53` — `alpha_{j+1} >= alpha_j for all adjacent rows j`.
Code: `run.py:93` — `{"type": "ineq", "fun": lambda alpha, j=j: alpha[j + 1] - alpha[j]}`. SLSQP "ineq" requires fun >= 0, i.e. alpha[j+1] - alpha[j] >= 0. Exact match.

**H5 — DGP parameters.**
Claim: `README.md:67-71` — rho=0.45, scale=5.00, lapse=0.02, 10 rows, 80 trials.
Code: `run.py:163-169` — `rho_true = 0.45`, `scale_true = 5.00`, `lapse = 0.02`, `prob_high = np.arange(0.10, 1.01, 0.10)` (10 elements), `trials_per_task = 80`. Exact match.

**H6 — Estimated rho.**
Claim: `README.md:98` — "estimate is **0.451**".
Code: `run.py:325` — `f"**{float(fixed['rho']):.3f}**"`. Optimizer returns 0.4507353890579118; rounds to 0.451 at 3dp. CSV: `0.45074` (5dp rounded). Exact match.

**H7 — True probabilities.**
Claim: `README.md:116-125` / `tables/row-fits.csv:2-11`, "True probability" column.
Recomputed: `structural_probabilities(rho=0.45, scale=5.0, lapse=0.02, prob_high=arange(0.10,1.01,0.10))` yields `[0.0204, 0.0219, 0.0285, 0.0570, 0.1659, 0.4471, 0.7707, 0.9237, 0.9668, 0.9770]`. Matches CSV to 4dp exactly.

**H8 — Simulated counts reproducible.**
Claim: counts in `tables/row-fits.csv:2-11` — `[1,4,1,8,12,29,69,72,77,79]`.
Code: `run.py:163` — `rng = np.random.default_rng(1)`. Recomputed with same seed and DGP: identical counts. Exact match.

**H9 — Unconstrained fit = observed shares.**
Claim implied by README table where "Unconstrained fit" = "Observed share" in all rows.
Code: `run.py:52-53` — `shares = np.clip(counts / trials, 1e-5, 1.0 - 1e-5)` / `"probabilities": shares`. All shares lie well within (1e-5, 1-1e-5), so no clipping. Unconstrained fit equals observed shares. CSV confirms exact equality across all 10 rows.

**H10 — Log likelihoods.**
Claim: `README.md:133-135` — unconstrained=-215.051, CRRA=-221.83, monotone=-216.044.
CSV: `-215.05113`, `-221.82988`, `-216.04389`. Round to 3dp: `-215.051`, `-221.830`, `-216.044`. Exact match.

**H11 — LL loss monotone = 0.99.**
Claim: `README.md:127` — "gives up 0.99 likelihood points".
Code: `run.py:379` — `f"{...:.2f}"`. CSV: `0.99276`. `f"{0.99276:.2f}"` = `"0.99"`. Exact match.

**H12 — Monotone pools rows 2-3.**
Claim: `README.md:92-94` — "The observed share falls from p=0.20 to p=0.30. The monotone fit pools the two rows."
CSV: monotone fit row2 = `0.0313`, row3 = `0.0313` (equal, pooled). Observed share row2 = `0.05`, row3 = `0.0125` (falls). Exact match.

## Cross-cutting patterns

- All three estimators (unconstrained saturated logit, fixed-scale CRRA logit, monotone row logit) are implemented exactly as named and described. No mislabeling.
- The only issue (Finding 1) spans two helper functions and is cosmetic: different floating-point thresholds for the same conceptual quantity. Published numbers unaffected.
- `scale_true` (`run.py:166`) and `fixed_scale` (`run.py:168`) are separate variables that happen to share the value 5.00. The README correctly describes the estimator as "fixing" the scale, not as knowing the DGP parameter name. No conflation.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below 50% halt threshold. Forward work may proceed.
1. Finding 1 is LOW severity and non-result-changing. Optional fix: unify violation threshold to a single value (e.g. `-1e-10` throughout) in both `estimate_unconstrained_logits` (`run.py:58`) and `estimate_monotone_logits` (`run.py:109`).
2. After any fix, re-run `python run.py` and confirm reported counts remain 1 and 0.
3. Re-run this skill to confirm score stays <= 10%.
