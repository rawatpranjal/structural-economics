# QC Report - `heterogeneous-agents/` (2026-05-05)

Quality-control sweep over the 3 tutorials in `heterogeneous-agents/`. Same
four dimensions (crux, pseudocode, coherence, reproducibility) as the
`dynamic-programming/` pass.

Headline: **all 3 tutorials pass cleanly.** No blockers, no majors, no minors.

## Tutorials in scope

`endogenous-grid-points`, `envelope-equation-iteration`,
`huggett-incomplete-markets`.

## Stage 1 - Static intake

`scripts/qc_subject.py heterogeneous-agents`:

| tutorial | sections | pseudocode | broken refs | notes |
|---|---|---|---|---|
| endogenous-grid-points | 7/7 | present | 0 | - |
| envelope-equation-iteration | 7/7 | present | 0 | - |
| huggett-incomplete-markets | 7/7 | present | 0 | - |

## Stage 2 - Reproducibility sweep

`scripts/qc_repro.py heterogeneous-agents --timeout 600`:

| tutorial | exit | wall (s) | files changed | classification |
|---|---:|---:|---:|---|
| endogenous-grid-points | 0 | 7.1 | 0 | idempotent |
| envelope-equation-iteration | 0 | 13.4 | 0 | idempotent |
| huggett-incomplete-markets | 0 | 1.4 | 0 | idempotent |

Notable: these tutorials are markedly faster than `dynamic-programming/aiyagari`
(213s) despite tackling overlapping incomplete-markets territory. The EGP/EEI
fixed-point loop avoids the inner maximization that dominates Aiyagari's grid
VFI; Huggett uses a continuous-time HJB/KFE on a small grid.

## Stage 3 - Subjective grading

One `Explore` agent graded all 3 tutorials. The rubric was tightened for this
subject: tutorials here are method comparisons, so reviewer was asked to
verify each explains *why* the alternate method exists, *what* it computes
that grid VFI doesn't, and *when* to reach for it.

| tutorial | crux | pseudocode | coherence claim | verdict |
|---|---|---|---|---|
| endogenous-grid-points | A | present | "main grid converged in 103 EGP iterations with consumption sup-norm error 9.77e-07" | match (CSV row 2; run.py 272, 468-469) |
| envelope-equation-iteration | A | present | "coarse-grid EEI converged in 149 iterations; fine-grid EGP shows max consumption gap 1.09e-02 over a≤20" | match (CSV rows 2, 5, 7; run.py 575, 582-583) |
| huggett-incomplete-markets | A | present | "HJB converged in 8 iterations with sup-norm 1.56e-07; bisection gives r\*=0.03192, residual 2.86e-06" | match (CSV rows 2, 9, 10, 11; run.py 305, 420-421) |

Reviewer notes (paraphrased):

- **endogenous-grid-points**: clearly explains why EGP exists (reverses the
  Bellman grid question to avoid the costly inner maximization) and ties the
  method to economics.
- **envelope-equation-iteration**: motivates EEI as a fixed point on
  marginal continuation values $W_a(a)$ rather than value levels - frames the
  envelope condition as an updating equation, not just a theorem.
- **huggett-incomplete-markets**: continuous-time HJB/KFE makes dual
  equilibrium objects (drift and density) transparent; upwind selection step
  in the pseudocode preserves the key economic insight.

Pseudocode in all three is symbolic, names inputs/outputs, and aligns with
the Equations sections.

## Findings

### Blockers

None.

### Majors

None.

### Minors

None.

## Fixes applied

None needed. Working tree unchanged in `heterogeneous-agents/`.

## Suggested follow-up

- These three tutorials are the cleanest QC pass yet. They are good
  templates for new method-comparison tutorials in adjacent subjects (e.g.,
  alternative DSGE solvers in `global-dsge/`).

## Reproducibility of this report

```bash
python3 scripts/qc_subject.py heterogeneous-agents --json --out /tmp/qc-static-ha.json
python3 scripts/qc_repro.py heterogeneous-agents --timeout 600 --out /tmp/qc-repro-ha.json
```
