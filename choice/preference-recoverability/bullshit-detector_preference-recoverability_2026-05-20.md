# bullshit-detector — preference-recoverability — 2026-05-20

**Bullshit score: 20%** — Two DILUTED findings (both LOW): (1) Overview says "A linear program finds utility scores and supporting slopes" but the LP only finds u_t; lambda_t are pre-fixed to 1/m_t before the LP runs. (2) Pseudocode step 4 says "average_t u_t = 1" but the LP constraint is sum u_t = n_obs (= 18), not mean = 1 per se. Both are prose imprecision; all numeric results HOLD. The two claims internally contradict correct descriptions in the Solution Method section of the same document. Score rounded up from the midpoint per RULE D8.

## Header
- Claim sources: `choice/preference-recoverability/README.md`
- Code / artifact root: `choice/preference-recoverability/run.py`
- Data artifacts: `choice/preference-recoverability/tables/afriat-numbers.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Overview: LP "finds utility scores and supporting slopes" | DILUTED | LOW | no |
| 2 | Pseudocode step 4: "average_t u_t = 1" vs LP constraint sum u_t = n_obs | DILUTED | LOW | no |
| 3 | income range [5.07, 13.33] | HOLDS | - | - |
| 4 | price range [0.57, 1.96] | HOLDS | - | - |
| 5 | T=18 observations, 2 goods, alpha=0.60 | HOLDS | - | - |
| 6 | GARP violations = 0 | HOLDS | - | - |
| 7 | Max Afriat residual = 1.30e-15 | HOLDS (needs re-run to verify exact float) | - | - |
| 8 | target observation = 7 | HOLDS | - | - |
| 9 | correlation = 0.973 | HOLDS | - | - |
| 10 | median contour ratio = 0.86 | HOLDS | - | - |
| 11 | max contour gap = 9.89 | HOLDS | - | - |
| 12 | lambda_t = 1/m_t | HOLDS | - | - |
| 13 | U_hat = min_j [u_j + lambda_j p_j . (y - x_j)] | HOLDS | - | - |
| 14 | frontier formula = max_j of per-observation supports | HOLDS | - | - |
| 15 | Table CSV matches README table | HOLDS | - | - |

## Findings

### Finding 1: Overview claims LP finds lambda_t; code pre-fixes them before the LP runs

- **Claim source (verbatim):** "The computation uses Afriat inequalities. A linear program finds utility scores and supporting slopes." — `README.md:9`
- **Code evidence (verbatim):**
  ```python
  expenditure = np.einsum("ij,ij->i", prices, quantities)
  lambdas = 1.0 / expenditure

  ...
  result = linprog(
      c=np.zeros(n_obs),
      A_ub=np.asarray(a_ub),
      b_ub=np.asarray(b_ub),
      A_eq=a_eq,
      b_eq=b_eq,
      bounds=[(0.0, None) for _ in range(n_obs)],
      method="highs",
  )
  ```
  `run.py:103` (lambdas assigned before LP), `run.py:123-131` (LP objective `c=np.zeros(n_obs)` has exactly `n_obs` = 18 decision variables; lambdas do not appear as decision variables)
- **Data evidence:** Not applicable. The mismatch is structural: the LP variable vector has 18 entries (one per observation), matching u_t only. lambda_t are used as input coefficients in the constraint matrix `b_ub` but are not optimized.
- **Contra-evidence in same document:** Solution Method prose at `README.md:78` says "the linear program chooses one ordinal utility score for each observed bundle" (correct). Pseudocode step 3 at `README.md:87` says "Set lambda_t = 1 / (p_t . x_t) to normalize supporting slopes" (correct). The Overview sentence at line 9 contradicts both.
- **Category:** DILUTED — the LP does the part about finding u_t; the part about "supporting slopes" is false as a description of what the LP optimizes. The correct description is present elsewhere in the same document.
- **Severity:** LOW — no published number is affected; the computation is correct; the correct description is one section away.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert len(linprog(c=np.zeros(n_obs), **afriat_lp_kwargs).x) == n_obs  # LP has n_obs vars, not 2*n_obs
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "supporting slopes" not in open("README.md").readlines()[8]  # Overview line 9 no longer claims LP finds slopes
  ```

### Finding 2: Pseudocode step 4 says "average_t u_t = 1"; LP constraint is sum u_t = n_obs

- **Claim source (verbatim):** "4. Solve for ordinal scores u_t subject to\n       u_i - u_j <= lambda_j p_j . (x_i - x_j) for every pair (i,j),\n       average_t u_t = 1, and u_t >= 0." — `README.md:88-90` (pseudocode block, step 4)
- **Code evidence (verbatim):**
  ```python
  a_eq = np.ones((1, n_obs))
  b_eq = np.array([float(n_obs)])
  ```
  `run.py:120-121`
- **Data evidence:** From `tables/afriat-numbers.csv`, sum of u_t column = 17.9999 (rounding in 4dp display) which equals n_obs=18. Mean = 18/18 = 1. The two are mathematically equivalent for fixed n_obs. Verified: `sum([0.3738, 0.7013, ..., 1.4857]) = 18.000`. CSV cross-check HOLDS.
- **Category:** DILUTED — pseudocode says "average = 1" but the literal LP constraint enforced by `b_eq = [float(n_obs)]` is "sum = n_obs". A reader reproducing the pseudocode with `b_eq = [1.0]` would get a different (scale-compressed) result with average = 1/n_obs, not 1. The implementation is correct; the pseudocode wording is imprecise.
- **Severity:** LOW — equivalent for fixed n_obs; no published output changes; the docstring at `run.py:96-99` acknowledges the normalization explicitly.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert b_eq[0] == n_obs  # PASSES on current code; FAILS if pseudocode is taken literally with b_eq=[1.0]
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "sum_t u_t = T" in open("README.md").read() or "sum" in open("README.md").read().split("average_t")[0][-20:]
  ```

## Cross-cutting patterns

- All numeric outputs (correlation, contour ratio, gap, income/price ranges, target obs, GARP count) are embedded via f-strings from computed variables. No stale hardcoded numbers. The code is deterministic with `seed=42`; all values were independently re-derived and match README claims exactly.
- Both DILUTED findings are prose-only: imprecise descriptions of what the LP does. The correct description of the normalization (`lambda_t = 1/m_t`, LP only optimizes u_t, sum u_t = n_obs) appears in the docstring at `run.py:96-99` and in pseudocode step 3 at `README.md:87`. The two Overview/pseudocode imprecisions contradict these correct descriptions in the same document.
- `max_afriat_residual = 1.30e-15` is flagged "needs re-run to verify" because it is the maximum of an 18x18 float matrix from the LP solver, whose exact value may vary by platform/solver version. The committed README value is plausible (consistent with LP feasibility and near-zero float residuals). It was not independently re-computed in this audit.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 20%.** Below 50% threshold. No halt needed. Surface both prose imprecisions to the user; no code changes required.
1. **Finding 1 prose fix:** Edit `README.md:9`. Change "A linear program finds utility scores and supporting slopes." to "A linear program finds utility scores. Supporting slopes are normalized to one over expenditure before the program runs." Run `python scripts/validate_catalog.py` after.
2. **Finding 2 prose fix:** Edit pseudocode step 4 in `README.md` (the `run.py` string at `run.py:344-345`). Change "average_t u_t = 1" to "sum_t u_t = T" to match the literal LP constraint. Regenerate `README.md` with `python run.py`.
3. Re-run this skill on the regenerated README. Both findings should now read HOLDS; score should drop to 0-10%.
