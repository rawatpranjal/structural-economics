# bullshit-detector -- logit-discrete-choice -- 2026-05-20

**Bullshit score: 5%** -- All structural, numeric, and formula claims hold against code and committed data artifacts. Score reflects the non-zero probability that BFGS hess_inv SE values drift on a future re-run with a different seed or optimizer state (needs re-run to verify absolute SE values), but no current finding crosses into a non-HOLDS category.

## Header
- Claim sources: `choice/logit-discrete-choice/README.md` (Overview, Equations, Model Setup, Results prose, tables)
- Code / artifact root: `choice/logit-discrete-choice/run.py`
- Data artifacts: `choice/logit-discrete-choice/tables/estimation-results.csv`, `choice/logit-discrete-choice/tables/elasticity-matrix.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|-----------------|
| 1 | Utility U_ij = V_j + eps_ij, V_j = beta_p*p_j + beta_q*q_j | HOLDS | -- | no |
| 2 | eps_ij i.i.d. Type I extreme value | HOLDS | -- | no |
| 3 | Choice prob P_j = exp(V_j)/sum_k exp(V_k) | HOLDS | -- | no |
| 4 | Log-lik ell = sum_i sum_j d_ij log P_j | HOLDS | -- | no |
| 5 | Own-price elas eta_jj = beta_p * p_j * (1-s_j) | HOLDS | -- | no |
| 6 | Cross-price elas eta_jk = -beta_p * p_k * s_k | HOLDS | -- | no |
| 7 | True beta_p = -0.5, True beta_q = 1.2 | HOLDS | -- | no |
| 8 | N=5000 consumers, J=5 products | HOLDS | -- | no |
| 9 | Prices [2.0,3.5,5.0,7.0,10.0], Quality [1.0,2.0,3.5,4.0,5.0] | HOLDS | -- | no |
| 10 | Estimate beta_p = -0.4913, SE=0.0174, t=-28.21 | HOLDS | -- | no |
| 11 | Estimate beta_q = 1.1559, SE=0.0362, t=31.94 | HOLDS | -- | no |
| 12 | Elasticity table values (all 25 cells) match formula | HOLDS | -- | no |
| 13 | "Higher prices make demand more elastic in absolute value" | HOLDS | -- | no |
| 14 | IIA: pairwise odds ratios stay fixed after removing Product 3 | HOLDS | -- | no |
| 15 | MLE via BFGS maximization of ell(beta) | HOLDS | -- | no |

## Findings

### Finding 1 -- All claims hold

All 15 extracted claims were grounded against verbatim code or data artifacts. No non-HOLDS finding was produced. Full grounding follows.

**Claim 1 -- Utility spec.** Claim: "V_j = beta_p*p_j + beta_q*q_j" -- `README.md:19`. Code `run.py:77`: `V_j = beta_price * X_price + beta_quality * X_quality`. HOLDS.

**Claim 2 -- Error distribution.** Claim: "eps_ij i.i.d. Type I extreme value" -- `README.md:21`. Code `run.py:123`: `epsilon = np.random.gumbel(loc=0, scale=1, size=(N, J))`. Gumbel(0,1) is the standard Type I extreme value distribution. HOLDS.

**Claim 3 -- Choice probability.** Claim: `P_j = exp(V_j)/sum_k exp(V_k)` -- `README.md:25-27`. Code `run.py:39-41`: max-shifted softmax. Algebraically identical (max subtraction is a numerical stability trick, does not change the result). HOLDS.

**Claim 4 -- Log-likelihood.** Claim: `ell = sum_i sum_j d_ij log P_j` -- `README.md:30`. Code `run.py:83-84`: `chosen_probs = probs[np.arange(N), choices]`, `ll = np.sum(np.log(...))`. Indexing by `choices` is the equivalent of the indicator sum. HOLDS.

**Claim 5 -- Own-price elasticity formula.** Claim: `eta_jj = beta_p * p_j * (1 - s_j)` -- `README.md:34`. Derived from first principles: `dP_j/dp_j = beta_p * P_j * (1-P_j)`, so `eta_jj = dP_j/dp_j * p_j/P_j = beta_p * p_j * (1-P_j)`. Code `run.py:186`: `own_elasticities = beta_price_hat * X_price * (1 - predicted_shares)`. HOLDS.

**Claim 6 -- Cross-price elasticity formula.** Claim: `eta_jk = -beta_p * p_k * s_k` -- `README.md:35`. Derived: `dP_j/dp_k = -beta_p * P_j * P_k` for j!=k, so `eta_jk = -beta_p * p_k * P_k`. Code `run.py:193`: `-beta_price_hat * X_price[k] * predicted_shares[k]`. HOLDS.

**Claim 7 -- True parameters.** README table `README.md:51-52`: True beta_p = -0.5, True beta_q = 1.2. Code `run.py:105-106`: `beta_price_true = -0.5`, `beta_quality_true = 1.2`. CSV `estimation-results.csv:2-3`: `-0.5000`, `1.2000`. HOLDS.

**Claim 8 -- Sample size.** README table: "Consumers 5000" -- `README.md:47`. Code `run.py:101`: `N = 5000`. HOLDS.

**Claim 9 -- Product characteristics.** README table prices/quality -- `README.md:49-50`. Code `run.py:112-113` exact match. HOLDS.

**Claim 10 -- MLE estimate beta_p.** README table: Estimate=-0.4913, SE=0.0174, t=-28.21 -- `README.md:94`. CSV `estimation-results.csv:2`: `-0.4913,0.0174,-28.21`. Arithmetic check: -0.4913/0.0174 = -28.24 (rounds to -28.21 within rounding of stored values at 4 decimal places). SE sourced from `result.hess_inv` `run.py:160`; no claim about SE method in README prose. HOLDS.

**Claim 11 -- MLE estimate beta_q.** README table: Estimate=1.1559, SE=0.0362, t=31.94 -- `README.md:95`. CSV `estimation-results.csv:3`: `1.1559,0.0362,31.94`. Arithmetic: 1.1559/0.0362 = 31.93 (rounds to 31.94). HOLDS.

**Claim 12 -- Elasticity table.** All 25 cells verified internally consistent with formula using implied shares. Own-elas diagonal monotonically more negative with price, consistent with `README.md:80-81`. HOLDS.

**Claim 13 -- "Higher prices make demand more elastic."** Own-elas values: -0.896, -1.490, -1.568, -2.609, -4.051 for prices 2.0, 3.5, 5.0, 7.0, 10.0 -- strictly decreasing (more negative). HOLDS.

**Claim 14 -- IIA odds ratios.** Claim: "pairwise odds ratios stay fixed" -- `README.md:84`. Code `run.py:200-215` computes logit softmax over restricted set; for logit, P_j/P_k = exp(V_j-V_k) holds regardless of which other alternatives are present. Algebraically exact, not approximate. HOLDS.

**Claim 15 -- BFGS optimization.** README Solution Method: "maximize ell(beta)" -- `README.md:58`. Code `run.py:149-154`: `minimize(log_likelihood, ...)` where `log_likelihood` returns negative LL. Minimizing negative LL = maximizing LL. HOLDS.

## Cross-cutting patterns

- Every numeric claim in the README is generated directly from code variables (f-string interpolation in `add_model_setup`, `add_table`), not hardcoded. There is no source of doc-vs-code numeric drift except stochastic variation across seeds. `np.random.seed(42)` at `run.py:99` fixes the seed, so committed CSV values will be reproducible on re-run.
- The SE values come from BFGS `result.hess_inv` (`run.py:160`), which is an approximation. README never claims these are exact Fisher information SEs or sandwich SEs. No false claim, but a future reader may interpret them as exact -- minor pedagogical gap, not a faithfulness violation.
- No hardcoded numbers appear in README prose that are not also derivable from code. All result numbers are injected by the `ModelReport` machinery.

## TDD execution sequence (for the next agent)

0. Bullshit score is 5% -- below the 50% halt threshold. No surface-to-user pause required. Proceed.
1. No non-HOLDS findings. No pytest tests required to prove bugs.
2. No honest-fix pass conditions required.
3. No fixes needed.
4. If a re-run is desired to confirm SE values (needs re-run to verify), run `python run.py` inside `choice/logit-discrete-choice/` and diff the resulting CSV against the committed artifact. With `np.random.seed(42)` the values should be bit-for-bit identical.
5. Re-run this skill after any future change to `run.py` that modifies the SE formula, optimizer, or elasticity computation.
