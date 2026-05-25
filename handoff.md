# Handoff - 2026-05-25 - main

## Where we left off

Catalog-wide rollout of `## Worked Numerical Example` complete. 50 tutorials now carry a hand-computable section between `## Equations` and `## Model Setup`, ending in `\boxed{...}`. Last commit `5aa5cc2`. Project back to close-out.

## Active streams

(none in flight)

Deferred:
- **Phase 2 Worked Numerical Example PR**: ~21 borderline tutorials triaged but not authored. Candidates listed in `/Users/pranjal/.claude/plans/this-is-really-good-mighty-fountain.md` (assetNews, behavioral-nk, blanchard-kahn-determinacy, reduced-form-var, minnesota-svar, regret-matching, etc.). Permanently skipped: ~38 (BLP, HANK, neural-net, stochastic-algorithm tutorials).
- Wave 3: no spec, user has not signaled intent. Out-of-scope items live in `spec.md > Out of scope (Wave 2)`.
- 25-item Wave 2 sub-tutorial queue at `~/.claude/plans/wave-2-integration-axial-falcon.md` spec-locked but not started.

## Decisions made this session

- Worked Numerical Example format: `## ` top-level, between Equations and Model Setup; one motivating sentence then line-by-line math; no `**Step.**` / `**Setup.**` / `**Pre-merger.**` bold paragraph labels; final `\boxed{...}`. Rule lives in CLAUDE.md `## Learned Rules`.
- Mixed Opus/Sonnet dispatch worked once anti-pattern lists were named. Wave 2B (sonnet, abstract rules) had 7 of 11 bold-label violations; waves 2C-2E (named ban list) had 0 of 27. Memory: `feedback_parallel_dispatch_format_drift.md`.
- Three-layer audit: format validator + per-wave shallow audit (2 random tutorials per wave) + end-of-rollout deep audit (8-of-47 random sample). Deep audit caught the only real formula bug (nash-in-nash insurer-vs-hospital enrollment), fixed at `5aa5cc2`.
- Weitzman README sign error in the Gaussian closed form (`(1 - Phi)` should be `Phi`) found during initial math verification, fixed standalone at `1e0ef12` before the pilot.

## Open questions

(none)

## Landmines

- Phase 2 borderline list in plan file: do NOT redo eligibility from scratch. Triage already done by 3 Explore agents; eligibility verdicts are in the plan file's per-cluster tables.
- Nash-in-Nash worked example: the README's `q_d(G)` denotes INSURER d's total enrollment, not hospital-conditional. The original draft used hospital-level 50 in the transfer denominator; corrected to insurer-level 100. Future edits to that tutorial must preserve this distinction. See commit `5aa5cc2`.

## Suggested next move

If user signals Phase 2: open `/Users/pranjal/.claude/plans/this-is-really-good-mighty-fountain.md`, read the Phase 2 borderline list, triage to ~5-6 tutorials with eyes-on, dispatch a single wave. Otherwise project remains closed-out.
