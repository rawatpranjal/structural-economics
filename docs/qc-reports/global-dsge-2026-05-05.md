# QC Report - `global-dsge/` (2026-05-05)

Quality-control sweep over the 3 tutorials in `global-dsge/`. Same four
dimensions (crux, pseudocode, coherence, reproducibility) as the prior
subjects.

Headline: **2 clean (A), 1 flagged (B)**. One major finding on
`rbc-capital-tax`'s pedagogical motivation. No blockers.

## Tutorials in scope

`heaton-lucas` (STPFI), `rbc-capital-tax` (VFI + Euler refinement), and
`rbc-irreversible-investment` (VFI with binding boundary).

Per `global-dsge/CLAUDE.md`, tutorials in this section should foreground "the
economic friction or payoff from global solution: taxation, irreversibility,
incomplete markets, constraints, or risk premia." That contract is the
rubric the reviewer applied below.

## Stage 1 - Static intake

`scripts/qc_subject.py global-dsge`:

| tutorial | sections | pseudocode | broken refs | notes |
|---|---|---|---|---|
| heaton-lucas | 7/7 | present | 0 | - |
| rbc-capital-tax | 7/7 | present | 0 | - |
| rbc-irreversible-investment | 7/7 | present | 0 | - |

## Stage 2 - Reproducibility sweep

`scripts/qc_repro.py global-dsge --timeout 600`:

| tutorial | exit | wall (s) | files changed | classification |
|---|---:|---:|---:|---|
| heaton-lucas | 0 | 391.1 | 0 | idempotent (STPFI) |
| rbc-capital-tax | 0 | 4.1 | 0 | idempotent |
| rbc-irreversible-investment | 0 | 2.0 | 0 | idempotent |

heaton-lucas is the long pole at 6.5 min wall time. STPFI is fundamentally
iterative - `lib/stpfi.py` solves a nonlinear system per collocation point
per iteration. No regression, no change to action; just a note for future
sweep budgeting.

## Stage 3 - Subjective grading

| tutorial | crux | pseudocode | coherence claim | verdict |
|---|---|---|---|---|
| heaton-lucas | A | present | "computed equity premium ranges from 0.43% to 1.42% across the wealth-share grid" | match (run.py 329-331) |
| rbc-capital-tax | B | present | "at τ_k=30%, capital is 42.7% below the no-tax value, output is 18.2% lower, consumption 9.7% lower" | match (CSV: 21.7584/37.9893, 3.0307/3.7041, 2.4868/2.7543) |
| rbc-irreversible-investment | A | present | "binding frequency: 9.5% of state space; 0.42% in the stationary distribution; 13.3% in the overhang experiment" | match (numerical-checks.csv) |

## Findings

### Blockers

None.

### Majors

**`rbc-capital-tax` does not articulate why a global solution is needed.**
The capital tax is a permanent, linear wedge on the Euler equation - a setting
where log-linear approximation around the *new* steady state typically performs
well. The tutorial computes a global VFI policy but never explains what that
buys over a local approximation. Per `global-dsge/CLAUDE.md`, this section's
pedagogical contract is to "foreground the economic friction or payoff from
global solution"; here the friction (a permanent wedge) does not strictly
require global methods. The Overview, Solution Method, and Takeaway should
either:

- name a feature of the global solution that a linear approximation would
  miss (large deviations from steady state? off-steady-state nonlinearity?
  precautionary savings effects?), or
- demote the tutorial's claim to "global solution as implementation choice,
  not necessity," and let the reader see why local would also work.

**Not auto-fixed** because the right framing depends on the user's
pedagogical intent. Suggested edits in `global-dsge/rbc-capital-tax/run.py`:

- Overview (around line 270-290): add one sentence justifying global over
  linear.
- Solution Method: optionally add a one-line comparison against a log-linear
  approximation around the no-tax SS, showing where they agree and disagree.
- Takeaway (around line 575): name what the global solution shows that
  matters.

### Minors

- **`heaton-lucas`** (`run.py` Results section): the fine-grid 10th and 90th
  percentiles for `omega_all` are reported as -0.050 and 1.050 - i.e., the
  grid bounds. The simulated wealth distribution is hitting the boundary of
  the state space. Worth a sentence noting that this is a real modeling
  feature (extreme realizations of relative wealth), not a numerical
  artifact, and what it implies for interpretation.
- **`rbc-capital-tax`** Overview/Takeaway: see above; minor sub-issues of
  the same major flag.
- **`rbc-irreversible-investment`** Results: the fine-grid value-function
  gap of 0.2941 capital units is reported without a brief note on whether it
  signals grid-resolution issues or is a deliberate accuracy disclaimer. A
  sentence on grid quality would strengthen confidence in the coarse-grid
  policy.

## Fixes applied

None. All three tutorials are idempotent under repro; the major and minors
are substantive prose calls routed to the user.

## Suggested follow-up

1. Rewrite the `rbc-capital-tax` Overview to make explicit why this is a
   global-DSGE tutorial and not a local-perturbation tutorial. ~3 sentences.
2. Add a one-line note in `heaton-lucas` Results explaining the boundary-
   hitting wealth-share distribution.
3. Add a one-line grid-quality note in `rbc-irreversible-investment` Results.

## Reproducibility of this report

```bash
python3 scripts/qc_subject.py global-dsge --json --out /tmp/qc-static-global-dsge.json
python3 scripts/qc_repro.py global-dsge --timeout 600 --out /tmp/qc-repro-global-dsge.json
```
