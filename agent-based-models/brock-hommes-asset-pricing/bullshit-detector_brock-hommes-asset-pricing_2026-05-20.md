# bullshit-detector — brock-hommes-asset-pricing — 2026-05-20

**Bullshit score: 15%** — all core equations hold; two LOW-severity omissions (undisclosed initial conditions and implicit c_F=0) are the worst findings; no result-changing bugs

## Header
- Claim sources: `agent-based-models/brock-hommes-asset-pricing/README.md`
- Code / artifact root: `agent-based-models/brock-hommes-asset-pricing/run.py`, `lib/brock_hommes.py`
- Data artifacts: `agent-based-models/brock-hommes-asset-pricing/tables/smm-fit.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Market clearing eq x_t = (n_F f_F + n_T f_T)/R + eps | HOLDS | - | no |
| 2 | Profit score pi_h,t = e_t(f_h,t - R x_{t-1})/(as^2) - c_h | HOLDS | - | no |
| 3 | Score smoothing U_h,t = lam U_{h,t-1} + (1-lam) pi_h,t | HOLDS | - | no |
| 4 | Logit shares n_h,t = exp(beta U_h,t) / sum_j exp(beta U_j,t) | HOLDS | - | no |
| 5 | Trend forecast bounded by xbar * tanh(.../ xbar) | HOLDS | - | no |
| 6 | p* = d/(R-1) = 20.00 | HOLDS | - | no |
| 7 | All calibration table values match code | HOLDS | - | no |
| 8 | SMM grid "even candidates from 2 to 60" | HOLDS | - | no |
| 9 | SMM table numbers match tables/smm-fit.csv | HOLDS | - | no |
| 10 | Initial conditions x_0, x_1 not given values in Model Setup | DILUTED | LOW | no |
| 11 | c_F = 0 not disclosed; only c_T = 0.001 listed | DILUTED | LOW | no |

## Findings

### Finding 1: Initial conditions not disclosed in Model Setup table

- **Claim source (verbatim):** "1. Set p* = d / (R - 1), x_t = p_t - p*, and initialize x_0, x_1." — `README.md:103`
- **Code evidence (verbatim):**
  ```python
  x[0] = params.x_lag
  x[1] = params.x0
  ```
  `lib/brock_hommes.py:82-83`
  ```python
  x_lag: float = 0.10
  x0: float = 0.12
  ```
  `lib/brock_hommes.py:33-34`
- **Data evidence (if applicable):** None.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no — burn-in of 100 periods is designed to absorb transient dynamics from any reasonable initialization; changing x_lag/x0 within a small neighborhood of 0.10/0.12 does not materially shift the SMM moments.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "x_lag" not in open("agent-based-models/brock-hommes-asset-pricing/README.md").read() and "0.10" not in open("agent-based-models/brock-hommes-asset-pricing/README.md").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "0.10" in open("agent-based-models/brock-hommes-asset-pricing/README.md").read() or "x_lag" in open("agent-based-models/brock-hommes-asset-pricing/README.md").read()
  ```

### Finding 2: Fundamentalist cost c_F = 0 implicit but undisclosed

- **Claim source (verbatim):** "| Trend cost | $c_T$ | 0.001 | Small information or trading cost |" — `README.md:88`; and equation "cost $c_h$" — `README.md:18`
- **Code evidence (verbatim):**
  ```python
  costs = np.array([0.0, params.trend_cost])
  ```
  `lib/brock_hommes.py:88`
- **Data evidence (if applicable):** None.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no — c_F=0 is the standard Brock-Hommes calibration; the asymmetry (trend-follower pays a cost, fundamentalist does not) is the economically meaningful assumption; omitting it from the table means a reader cannot fully reproduce from the README alone, but the result does not change.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "c_F" not in open("agent-based-models/brock-hommes-asset-pricing/README.md").read() and "0.0" not in open("agent-based-models/brock-hommes-asset-pricing/README.md").read().split("Model Setup")[1].split("Solution")[0]
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "c_F" in open("agent-based-models/brock-hommes-asset-pricing/README.md").read() or "Fundamentalist cost" in open("agent-based-models/brock-hommes-asset-pricing/README.md").read()
  ```

## Cross-cutting patterns

- Both findings are omissions from the Model Setup table, not equation errors. No core algorithm, equation, or result-table value is wrong. The pattern is underdisclosed initialization/cost structure rather than any parametric leak or label mismatch.
- The SMM objective notation `||W * (m(beta) - m_data)||^2` (README.md:121) uses W to denote element-wise diagonal scaling by `1/scale_k`; the code implements exactly this. Notation is ambiguous but mathematically consistent - not flagged as a finding.
- All six core algorithmic equations (market clearing, profit scoring, score smoothing, logit shares, trend forecast, fundamental price) have verbatim code counterparts with no gaps.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Both findings are LOW severity and not result-changing. Safe to ship after cosmetic table fix; no need to halt or re-audit siblings.
1. For Finding 1: add rows `| Initial deviation lag | $x_{t-2}$ | 0.10 | Starting price deviation |` and `| Initial deviation | $x_{t-1}$ | 0.12 | Starting price deviation |` to the Model Setup table in `run.py`.
2. For Finding 2: add row `| Fundamentalist cost | $c_F$ | 0.000 | Zero cost for the fundamental rule |` to the Model Setup table in `run.py`, immediately before or after the Trend cost row.
3. Regenerate README.md with `python run.py` and confirm `validate_catalog.py` passes.
4. Re-run this skill on the new README to confirm both findings now read HOLDS and score drops to 0-10%.
