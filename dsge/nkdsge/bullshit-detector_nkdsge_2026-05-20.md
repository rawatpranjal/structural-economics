# bullshit-detector — nkdsge — 2026-05-20

**Bullshit score: 5%** — all six claim categories return HOLDS; the sole residual is the QZ-diff precision string (1.4e-15) which is runtime-generated and was independently recomputed to 1.44e-15, rounding correctly. No load-bearing gap found.

## Header
- Claim sources: `dsge/nkdsge/README.md` (Overview, Equations, Model Setup, Solution Method, Results, Takeaway)
- Code / artifact root: `dsge/nkdsge/run.py`, `lib/perturbation.py`
- Data artifacts: `dsge/nkdsge/tables/impact-responses.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Three-equation NK system (IS, NKPC, Taylor) as stated | HOLDS | none | no |
| 2 | Parameters match table (sigma=1, beta=0.99, kappa=0.3, phi_pi=1.5, phi_y=0.125, rho_v=0.5, rho_d=0.8) | HOLDS | none | no |
| 3 | Undetermined-coefficients scalar equation psi_y*[...]=b_s | HOLDS | none | no |
| 4 | b_s = -1/sigma for monetary, b_s = 1 for demand | HOLDS | none | no |
| 5 | QZ diff <= 1.4e-15 | HOLDS | none | no |
| 6 | Impact table: monetary y=-0.820, pi=-0.487, i=0.166 | HOLDS | none | no |
| 7 | Impact table: demand y=0.749, pi=1.081, i=1.715 | HOLDS | none | no |
| 8 | "systematic part of the rule partly offsets the wedge" | HOLDS | none | no |
| 9 | "output and inflation fall" for monetary shock | HOLDS | none | no |
| 10 | "output and inflation rise" for demand shock | HOLDS | none | no |
| 11 | "two shocks move output and inflation in opposite directions" | HOLDS | none | no |
| 12 | IS-curve v_in_is sign in QZ matrices | HOLDS | none | no |
| 13 | psi_iv formula includes +1 (monetary); psi_id excludes +1 (demand) | HOLDS | none | no |
| 14 | Klein QZ BK satisfied for both shocks | HOLDS | none | no |
| 15 | CSV matches README table numbers | HOLDS | none | no |

## Findings

### Finding 1 (consolidated HOLDS): All claims verified

Every assertion in `README.md` was grounded in `run.py` and `lib/perturbation.py`. Full evidence follows.

**Claim: three-equation NK system as stated**
- `README.md:18-29` writes IS curve, NKPC, Taylor rule.
- `run.py:67-70` (docstring), `run.py:79-95` implement exactly this system.
- `HOLDS`

**Claim: parameter table values**
- `README.md:50-58`: sigma=1, beta=0.99, kappa=0.3, phi_pi=1.5, phi_y=0.125, rho_v=0.5, rho_d=0.8, sigma_e=0.010.
- `run.py:169-176`:
  ```python
  sigma = 1.0
  beta = 0.99
  phi_pi = 1.5
  phi_y = 0.125
  kappa = 0.3
  rho_v = 0.5
  rho_d = 0.8
  sigma_e = 0.01
  ```
  `run.py:169-176`
- `HOLDS`

**Claim: scalar coefficient equation psi_y*[(1-rho_s) + phi_y/sigma + (phi_pi-rho_s)*kappa/(sigma*(1-beta*rho_s))] = b_s**
- `README.md:74`
- `run.py:93` (monetary): `coeff = (1 - rho_v) + phi_y / sigma + (phi_pi - rho_v) * kappa / (sigma * denom_pc)`; `psi_yv = -1.0 / (sigma * coeff)`.
- `run.py:121` (demand): same `coeff` with `rho_d`; `psi_yd = 1.0 / coeff`.
- Both match `README.md:74-76` exactly (b_s = -1/sigma for monetary via `-1.0/(sigma*coeff)`, b_s = 1 for demand via `1.0/coeff`).
- `HOLDS`

**Claim: QZ diff "at most 1.4e-15"**
- `README.md:90`; generated at `run.py:319`: `f"...differ by at most {max(mp_qz_diff, d_qz_diff):.1e}"`.
- Independently computed: monetary diff = 2.22e-16, demand diff = 1.44e-15, max = 1.44e-15 => rounds to `1.4e-15` at `.1e` format. ✓
- `HOLDS`

**Claim: impact table numbers (monetary: y=-0.820, pi=-0.487, i=0.166; demand: y=0.749, pi=1.081, i=1.715)**
- `README.md:106-110`; `tables/impact-responses.csv` verbatim:
  ```
  Variable,Monetary shock impact,Demand shock impact
  Output gap,-0.820,0.749
  Inflation,-0.487,1.081
  Nominal rate,0.166,1.715
  ```
- Independently computed from `run.py` formulas: MP y=-0.8203, pi=-0.4873, i=0.1665; D y=0.7493, pi=1.0807, i=1.7147. All match to stated decimal places.
- `HOLDS`

**Claim: IS-curve sign convention in QZ matrices**
- `README.md:94`: monetary shock raises real rate; demand shock raises demand.
- `run.py:29-34`:
  ```python
  if shock_kind == "monetary":
      v_in_is = 1.0 / sigma
  elif shock_kind == "demand":
      v_in_is = -1.0
  ```
  `run.py:29-34`
- Algebraically verified: monetary v_t enters IS via i_t with net coefficient +1/sigma on RHS after rearrangement; demand d_t enters IS LHS, moves to RHS as -1.0. Both signs correct.
- `HOLDS`

**Claim: psi_iv includes +1 (monetary), psi_id excludes it (demand)**
- `run.py:98`: `psi_iv = phi_pi * psi_piv + phi_y * psi_yv + 1.0`
- `run.py:124`: `psi_id = phi_pi * psi_pid + phi_y * psi_yd` (no +1)
- Taylor rule: `i_t = phi_pi*pi + phi_y*y + v_t`. For monetary shock, v_t=shock so coefficient carries +1. For demand shock, v_t=0, so no +1. ✓
- `HOLDS`

**Claim: Blanchard-Kahn satisfied**
- `README.md:90`, `README.md:116`
- Independently verified: both `klein_qz_nk` calls return `bk_message = "Blanchard-Kahn satisfied"`.
- `HOLDS`

## Cross-cutting patterns

- Zero fabrication or drift detected. All numeric claims are runtime-generated from the same formula the README describes, embedded in the report at generation time (`run.py:319`).
- The only potential vulnerability is that `README.md` is a committed artifact: if parameters change but `run.py` regeneration is not run, the README could drift. The QZ diff string (1.4e-15) would be the canary — any parameter change would alter it and expose stale docs. Structural safeguard already present.
- The `model.mod` file is acknowledged as non-executed (`run.py:157-158`, `CLAUDE.md` for dsge/) and README explicitly flags the different parameter values (`README.md:60`). No claim of Dynare equivalence is made.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 5%.** Below the 25% halt threshold. No blocking issues. Forward work may proceed.
1. No non-HOLDS findings. No failing tests to write.
2. If a parameter change is made to `run.py`, the regression contract is: re-run `python run.py` and confirm `tables/impact-responses.csv` and the embedded QZ diff string in `README.md` update consistently.
3. Optional regression invariants (would pass on current code and must continue to pass):
   ```python
   assert abs(float(df.loc[df.Variable=="Output gap","Monetary shock impact"]) - (-0.820)) < 0.001
   assert abs(float(df.loc[df.Variable=="Inflation","Demand shock impact"]) - 1.081) < 0.001
   ```
4. Re-run this skill only if `run.py` or `lib/perturbation.py` is modified.
