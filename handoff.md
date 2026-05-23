# Handoff - 2026-05-23 - main

## Where we left off

Audit follow-through session. Closed top-5 Overview-math + 3 notation renames + added validator guardrail. Then pruned stale docs: Wave 2 now marked DONE in spec.md (was "Open" / W2.4 "deferred"), missing-prelims doc tagged historical. Spec.md has zero forward work; roadmap currently empty.

## Active streams

**Audit residue (PARALLEL, low-priority).**
- 44 Overview-math warnings remain in 30 tutorials. Run `python scripts/validate_catalog.py --strict` to surface. No validator gate.
- Em-dash scrub: pre-existing dense tutorials likely still have em-dashes (only `spec.md` was scrubbed 2026-05-22). Repo-wide grep needed.
- NIT items from yesterday's audit: Zinkevich year drift (2007 vs 2008 across `cfr-asymmetric-auction/` and `regret-matching/`); dead `Lambda` symbol in KFE prelim.
- W2.4 multi-seed averaging (optional, ~30 LOC, tightens regret-comparison figure).

**Wave 3 design (BLOCKED on user brainstorm).**
- Not specced. Deferred items live in `spec.md > Out of scope (Wave 2)`: Gallego-van Ryzin sawtooth, network RM, markdown DP, behavioral runners (habit formation, reference dependence).
- Needs roadmap/spec session before any authoring.

## Decisions made this session

- `regret-matching/` literal rename `\bar\pi^T → \bar\pi_i^T` would have broken the Hart-Mas-Colell joint-CE statement. Resolved by adding per-player marginal row to Model Setup table, keeping joint for convergence statement.
- `hjb-growth/` used bold `\mathbf{A}^n` not plain `A` to avoid collision with scalar TFP `A` already in Model Setup. Matches Huggett convention.
- Overview-math: hybrid path (top-5 rewrite + validator warn) over mass-rewrite or skip. Long tail of 30 tutorials gets cleaned when next touched.
- 4 of 5 originally specced renames were false positives (sim-based-est folder absent; merger-sim divergence intentional+labeled; mixed-logit already documented; rbc prelims already populated). Only 3 real renames remained.

## Open questions

- Wave 3 brainstorm: behavioral runners vs IO runners vs new prelim batch. Needs user direction.
- Mass em-dash scrub: opportunistic per-PR or one-shot sweep?
- 30-tutorial Overview-math tail: incremental or batch?

## Landmines

- `optimal-control/upwind-finite-differences/README.md:90` still flags `hjb-growth/`: `G^n`, rename to `A` pending - the `pending` annotation is now stale (done today). Future cleanup.
- `docs/qc-reports/` reports dated 2026-05-05 to 2026-05-21 predate prelim cuts; do not trust them for post-Wave-1 state.
- `_legacy/` is 776MB, untouched. No runtime deps.

## Suggested next move

User-facing decision: open Wave 3 brainstorm (roadmap/spec session) OR opportunistic audit-residue cleanup. Spec.md is empty of forward work; nothing to grind without a fresh plan.
