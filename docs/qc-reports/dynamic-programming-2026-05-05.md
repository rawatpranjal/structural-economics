# QC Report — `dynamic-programming/` (2026-05-05)

Quality-control sweep over the 10 tutorials in `dynamic-programming/`. Four
dimensions, per CLAUDE.md and STYLE_GUIDE.md:

1. **Crux** — does the writeup explain the critical concept and intuition?
2. **Pseudocode** — present, aligned with Equations, not a Python copy?
3. **Coherence** — do prose claims match the figures and tabulated numbers?
4. **Reproducibility** — does `python run.py` regenerate `README.md`,
   `figures/`, and `tables/` cleanly?

Headline: **all 10 tutorials pass**. No blockers, no majors, one minor.

## Tutorials in scope

`aiyagari`, `asset-pricing`, `cake-eating`, `consumption-savings`,
`diamond-mortensen-pissarides`, `job-search-mccall`, `optimal-growth`, `rbc`,
`shock-discretization`, `solow-growth`.

## Stage 1 — Static intake

`scripts/qc_subject.py dynamic-programming` checks section presence,
pseudocode detection, and asset references against committed `README.md`.

| tutorial | sections | pseudocode | broken refs | notes |
|---|---|---|---|---|
| aiyagari | 7/7 | present | 0 | — |
| asset-pricing | 7/7 | present | 0 | — |
| cake-eating | 7/7 | present | 0 | — |
| consumption-savings | 7/7 | present | 0 | — |
| diamond-mortensen-pissarides | 7/7 | present | 0 | — |
| job-search-mccall | 7/7 | present | 0 | — |
| optimal-growth | 7/7 | present | 0 | — |
| rbc | 7/7 | present | 0 | — |
| shock-discretization | 7/7 | present | 0 | — |
| solow-growth | 7/7 | present | 0 | — |

Section coverage = `Overview, Equations, Model Setup, Solution Method, Results,
Takeaway, References`. Pseudocode is detected inside Solution Method as a
labeled `Algorithm:` block or numbered step list.

## Stage 2 — Reproducibility sweep

`scripts/qc_repro.py dynamic-programming --timeout 300` snapshots each tutorial,
runs `python3 run.py` with `MPLBACKEND=Agg` and `JAX_PLATFORMS=cpu`, then diffs.

| tutorial | exit | wall (s) | files changed | classification |
|---|---:|---:|---:|---|
| aiyagari | 0 | 213.0 | 0 | idempotent |
| asset-pricing | 0 | 1.4 | 0 | idempotent |
| cake-eating | 0 | 2.0 | 1 | cosmetic PNG byte delta (reverted) |
| consumption-savings | 0 | 12.7 | 1 | cosmetic PNG byte delta (reverted) |
| diamond-mortensen-pissarides | 0 | 7.5 | 0 | idempotent |
| job-search-mccall | 0 | 1.0 | 0 | idempotent |
| optimal-growth | 0 | 1.7 | 0 | idempotent |
| rbc | 0 | 7.6 | 0 | idempotent |
| shock-discretization | 0 | 10.1 | 0 | idempotent |
| solow-growth | 0 | 0.8 | 0 | idempotent |

Cosmetic PNG diffs (file-size delta < 1%, no structural change — matplotlib
metadata or PIL rounding) reverted via `git checkout --` so the working tree
is not polluted. The two reverted files were:

- `dynamic-programming/cake-eating/figures/policy-function.png` (48912 → 48906 bytes, 0.01%)
- `dynamic-programming/consumption-savings/figures/consumption-policy.png` (99087 → 99137 bytes, 0.05%)

Aiyagari is the lone slow runner (3.5 min, JAX-jit dominated). Within the
budget, no action needed.

## Stage 3 — Subjective grading (parallel reviewer agents)

Two `Explore` agents in parallel, 5 tutorials each. Each read `run.py +
README.md + tables/*.csv` and graded crux, pseudocode, and one coherence
spot-check per tutorial.

| tutorial | crux | pseudocode | coherence claim | verdict |
|---|---|---|---|---|
| aiyagari | A | present | "wealth Gini is 0.527, with 2.4% at the borrowing constraint" | match (CSV row matches; run.py 286–310) |
| asset-pricing | A | present | "stays within 0.011% of that benchmark" | match (run.py 185–192) |
| cake-eating | A | present | "with β=0.9, that share is 10.0%" | match (analytical c\* = (1−β)W = 0.1W) |
| consumption-savings | A | present | "MPC about 0.52 near the constraint and 0.04 near the top" | match (run.py 222–224) |
| diamond-mortensen-pissarides | A | present | "the local rule implies C=1.554" | match (run.py 204–212) |
| job-search-mccall | A | present | "finite grid accepts about 6.0%; continuous benchmark 6.1%" | match (table CSV row 6) |
| optimal-growth | A | present | "largest policy deviation outside bottom decile is 2.87e-02" | match (run.py 166–167) |
| rbc | A | present | "investment relative SD 4.12; consumption 0.34" | match (run.py 281–282; CSV row 4) |
| shock-discretization | A | present | "Tauchen 7-state persistence 0.9622; Rouwenhorst 0.9500" | match (CSV rows 7–8) |
| solow-growth | B | present | "terminal \|k\_{T−1}−k\*\| is 2.73e-04" | match (CSV row 2) |

Reviewers found no mismatches and no missing pseudocode. Pseudocode in every
tutorial is symbolic, names inputs/outputs, and aligns with the Equations
section per `STYLE_GUIDE.md` §Pseudocode.

## Findings

### Blockers

None.

### Majors

None.

### Minors

- **`solow-growth/README.md` Overview could position the model more
  explicitly within the dynamic-programming progression.** Reviewer note: "The
  Overview could more explicitly frame Solow as sitting between cake-eating
  (no production) and optimal-growth (endogenous saving). Currently mentions
  cake-eating reference but could strengthen the pedagogical progression."
  Suggested follow-up — not auto-applied because Overview prose is the
  tutorial's economic framing, which the user owns.

## Fixes applied

Mechanical only:

1. Reverted `dynamic-programming/cake-eating/figures/policy-function.png` —
   cosmetic regeneration delta (matplotlib metadata noise).
2. Reverted `dynamic-programming/consumption-savings/figures/consumption-policy.png`
   — same.

No `README.md` files edited. No code edited.

## Suggested follow-up

1. Consider strengthening the Solow Overview to explicitly contrast exogenous
   saving (Solow) with the endogenous saving introduced in `optimal-growth/`.
   Two or three sentences, no figure changes needed.
2. The Aiyagari run.py wall time (213s) is the long pole for the next subject
   sweep too. If the next subjects (`heterogeneous-agents/`,
   `global-dsge/`) have similar JAX-jit kernels, batch them in a parallel
   second pass after a sequential first pass establishes per-tutorial baselines.

## Reproducibility of this report

```bash
python3 scripts/qc_subject.py dynamic-programming --json --out /tmp/qc-static-dp.json
python3 scripts/qc_repro.py dynamic-programming --timeout 300 --out /tmp/qc-repro-dp.json
```

Raw scorecards from the parallel reviewer agents are not persisted on disk;
they are summarized in the Stage 3 table above.
