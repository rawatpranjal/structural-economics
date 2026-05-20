# bullshit-detector — dcegm-retirement-saving — 2026-05-20

**Bullshit score: 20%** — one DILUTED claim: "centered at 2.8" for a lognormal whose mean is 3.06; median is 2.8 but the prose word "centered" conventionally reads as mean, creating an internal contradiction with the reported "Mean assets at age 55 = 3.0556" in the same document.

## Header
- Claim sources: `structural-econometrics/dcegm-retirement-saving/README.md` (prose, Equations, Model Setup, Results tables)
- Code / artifact root: `structural-econometrics/dcegm-retirement-saving/run.py`
- Data artifacts: `tables/bruteforce-comparison.csv`, `tables/lifecycle-moments.csv`
- Seed audit (if any): None
- Run by: bullshit-detector skill (Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "initial assets centered at 2.8" | DILUTED | LOW | no |
| 2 | EGM Euler inversion formula | HOLDS | - | - |
| 3 | Endogenous asset formula | HOLDS | - | - |
| 4 | Branch value at endogenous point | HOLDS | - | - |
| 5 | CRRA utility and terminal bequest | HOLDS | - | - |
| 6 | Income formula y_t(work) | HOLDS | - | - |
| 7 | Work disutility psi_t(work) sign | HOLDS | - | - |
| 8 | Upper envelope: choose retire when V_retire >= V_work | HOLDS | - | - |
| 9 | Next-period status indexing (m'(work)=0, m'(retire)=1) | HOLDS | - | - |
| 10 | Absorbing retirement in simulation | HOLDS | - | - |
| 11 | Pseudocode constraint region V formula | HOLDS | - | - |
| 12 | All table numbers match CSV artifacts | HOLDS | - | - |
| 13 | Budget constraint c + a+ = R*a + y | HOLDS | - | - |
| 14 | Retire_gap sign convention consistent across DC-EGM and brute-force | HOLDS | - | - |

## Findings

### Finding 1: "Synthetic panel simulated with initial assets centered at 2.8"

- **Claim source (verbatim):** "Synthetic panel | 8,000 households | Simulated with initial assets centered at 2.8" — `README.md:202`

  Also in `run.py:683`:
  ```python
  f"| Synthetic panel | {p.simulation_agents:,} households | Simulated with initial assets centered at {p.initial_asset_mean:.1f} |"
  ```

- **Code evidence (verbatim):**
  ```python
  assets = np.clip(
      rng.lognormal(
          mean=np.log(p.initial_asset_mean),
          sigma=p.initial_asset_sigma,
          size=p.simulation_agents,
      ),
      0.2,
      13.5,
  )
  ```
  `run.py:324-331`

  `p.initial_asset_mean = 2.8` (`run.py:44`) and `p.initial_asset_sigma = 0.42` (`run.py:45`).

- **Data evidence:** `tables/lifecycle-moments.csv:5` reports `Mean assets at age 55,3.0556`. This is the actual simulated mean of the initial asset draw (before any period-1 saving), confirming the mean of the distribution is 3.0556, not 2.8.

- **Analysis:** For a lognormal with log-scale parameters `mu = log(2.8)` and `sigma = 0.42`, the distribution mean is `exp(mu + sigma^2/2) = 2.8 * exp(0.42^2/2) = 3.058`. The median is `exp(mu) = 2.8`. The code uses `mean=np.log(p.initial_asset_mean)` as the log-scale location parameter, so 2.8 is the *median* (and the log-scale mean), not the arithmetic mean. The word "centered" in the prose and table is ambiguous but conventionally reads as the arithmetic mean. The table in the same document already reports the correct arithmetic mean (3.0556), creating an internal contradiction: the prose says "centered at 2.8" while the table reports the mean as 3.0556.

- **Category:** DILUTED — the claim contains a kernel of truth (median = 2.8) but the chosen word "centered" suppresses the distinction between mean and median that matters for a right-skewed lognormal; it will mislead a reader who interprets "centered" as mean.

- **Severity:** LOW — the model and all reported outputs are correct; this is a prose precision failure, not a computational one. No number in the results table is wrong.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(np.exp(np.log(2.8)) - 2.8) < 1e-10  # median IS 2.8, PASSES on current code
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "centered at 2.8" not in open("README.md").read() and "median" in open("README.md").read()
  # PASSES on honest fix (replaces "centered at 2.8" with "median initial assets 2.8"),
  # FAILS on current prose
  ```

---

All other audited claims HOLD. Key confirmations:

**EGM Euler inversion** (`run.py:135`): `c_endog = inverse_marginal_utility(p.beta * p.R * vprime_next, p.gamma)` implements `c = (beta*R*mu)^{-1/gamma}` exactly as stated at `README.md:131`.

**Endogenous asset** (`run.py:137`): `current_asset_endog = (c_endog + a_next - y) / p.R` matches `README.md:137-138`.

**Branch value** (`run.py:138`): `branch_value_endog = utility(c_endog, p.gamma) + shift + p.beta * next_value` matches `README.md:146-147`.

**Upper envelope** (`run.py:207`): `choose_retire = retire_branch["value"] >= work_branch["value"]` matches `README.md:97-103`.

**Next-period status** (`run.py:200,205-206`): work branch uses `value[t+1, 0]`, retire branch uses `value[t+1, 1]`, matching `m'(work)=0, m'(retire)=1` (`README.md:26-29`).

**Absorbing retirement in simulation** (`run.py:340`): `new_retire = (~retired) & (...)` prevents already-retired households from transitioning; `retired = retired | new_retire` persists the state.

**Pseudocode constraint region** (`run.py:157-164`): code sets `branch_value[constrained] = utility(...) + shift + p.beta * next_value[0]`; `next_value[0]` equals `V_{t+1}^{next_status}(asset_min)` because `asset_grid[0] = 0.0 = p.asset_min` (confirmed by `linspace(0.0, 22.0, 420)`). Matches pseudocode at `README.md:265-267`.

**All table numbers**: `tables/bruteforce-comparison.csv` and `tables/lifecycle-moments.csv` match `README.md:336-357` exactly, cell by cell.

**Income formula** (`run.py:82`): `1.42 - 0.012*(age-55) - 0.006*max(age-62,0)**2` matches `README.md:44-46`.

**Work disutility sign** (`run.py:87,102`): `work_disutility` returns the positive magnitude; `branch_shift` returns `-work_disutility(age)`, implementing the negative sign in `README.md:48-50`.

## Cross-cutting patterns

- The single DILUTED finding is isolated to one prose string in the Model Setup table; it does not propagate to any equation, figure, or numeric result. The table itself reports the correct arithmetic mean (3.0556).
- The code is unusually faithful to the stated equations: every EGM formula, sign convention, next-status index, and boundary condition maps to a verbatim code line without indirection. The Iskhakov et al. (2017) logistic-taste-shock interpretation is consistent with the simulation's smooth retirement probability (`1/(1+exp(-gap/taste_scale))`).
- No claim in the Equations section fails; the DILUTED finding is confined to a single table cell in Model Setup.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% (< 50%).** No halt required. The single finding is a prose fix, not a code fix.

1. Violated-invariant test (confirms bug is real — PASSES on current prose):
   ```python
   assert "centered at 2.8" in open("structural-econometrics/dcegm-retirement-saving/README.md").read()
   ```

2. Honest-fix pass condition (FAILS on current prose, PASSES after fix):
   ```python
   assert "centered at 2.8" not in open("structural-econometrics/dcegm-retirement-saving/README.md").read()
   ```

3. Fix: in `run.py:683`, replace `"centered at {p.initial_asset_mean:.1f}"` with `"median initial assets {p.initial_asset_mean:.1f}"` (or `"log-scale mean {p.initial_asset_mean:.1f}, arithmetic mean {p.initial_asset_mean * np.exp(p.initial_asset_sigma**2/2):.2f}"`). Regenerate `README.md` with `python run.py`.

4. No re-run of the numeric solver is needed; the distribution parameters and all outputs are unchanged.

5. Re-run this skill after the prose fix to confirm the finding now reads HOLDS and the bullshit score drops to 0-10%.
