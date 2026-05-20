# Roadmap: Audit, Fix, and Split the Tutorial Catalog

A->B arc for verifying and restructuring the 97-tutorial catalog. Full plan
and subagent prompt templates live in
`~/.claude/plans/need-to-run-bullshit-quirky-aho.md`.

**A (start):** 97 tutorials, each a single `run.py` that both computes and
generates `README.md` via `lib.output.ModelReport`. Faithfulness of code vs
writeup is unverified.

**B (goal):** every tutorial audited and fixed to a bullshit score <=25%,
and restructured into a hand-written standalone `README.md` plus a
pure-computation `run.py`.

## Increments

- [x] **Stage 0 - Scaffold.** Create `ROADMAP.md` and `audit.md`.
- [x] **Stage 1 - Audit.** Run `bullshit-detector` on all 97 tutorials via
      batched sonnet subagents. One `bullshit-detector_<stem>_<date>.md` per
      folder; consolidate scores into root `audit.md`. 97/97 done; 8 HOLDS,
      18 FALSE, top score 75%.
- [x] **Stage 2 - Highest-risk fixes.** Opus subagents fixed the 19-tutorial
      shortlist; sonnet checks re-audited all 19 to <=25% (top was 75%->10%).
- [x] **Stage 3 - Mop up.** Opus subagents cleared findings in the other
      70 tutorials; sonnet checks re-audited all to <=25% (most to 0%).
- [x] **Stage 4 - Split writeup from code.** `CLAUDE.md` Tutorial
      Contract rewritten; all 97 tutorials converted to hand-written
      `README.md` + pure-computation `run.py` by sonnet subagents;
      `lib/output.py` moved to `_legacy/`; `scripts/validate_catalog.py`
      passes for all 97 tutorials.

## Status

All stages complete. 97 tutorials audited, fixed, re-checked <=25%, and
split into hand-maintained `README.md` + pure-computation `run.py`.
`validate_catalog.py` passes.
