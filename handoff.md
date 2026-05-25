# Handoff - 2026-05-25 - main

## Where we left off

Phase 2 of `## Worked Numerical Example` rollout complete. 15 borderline tutorials now carry the section, bringing the total to 65 across the catalog (50 Phase 1 + 15 Phase 2). Three wave commits: `99fb963` (wave A, 7 DSGE/macro), `707a26a` (wave B, 5 search/choice/estimation), `a174a4c` (wave C, 3 games + ABM).

## Active streams

(none in flight)

Deferred:
- **Wave 2 sub-tutorial integration**: 25 new tutorials spec-locked at `/Users/pranjal/.claude/plans/wave-2-integration-axial-falcon.md` but not started. This is the largest unshipped chunk.
- **Wave 3**: no spec; user has not signaled intent.
- **Worked Numerical Example, remaining unnamed candidates**: plan file mentioned "9 borderline from agent 2's report" without enumerating; the 15 explicitly named in `~/.claude/plans/this-is-really-good-mighty-fountain.md > Out of scope (Phase 2)` have all been shipped. If user wants the unnamed remainder, would need to recover agent 2's triage list first.

## Decisions made this session

- Phase 2 dispatch: 3 waves of 7/5/3 tutorials, all Opus authors (judgment-heavy because borderline). Same dispatch template as Phase 1 with explicit anti-pattern ban list. Zero format violations across 15 tutorials.
- Per-tutorial substitutions where README's headline calibration was too messy for hand arithmetic: documented in one opening sentence per section. Examples: auction-valuation-recovery used U[0,1]/n=3 instead of README's Beta(2,5)/n=4; cobweb-arifovic-ga used n=4 firms instead of n=30; blanchard-kahn-determinacy dropped Taylor wedge to reduce 3x3 to 2x2.
- Several worked examples explicitly reproduced numbers from Results sections to provide a cross-check: assetNews `q_0=-0.918%` matches Results table `-0.917`; behavioral-nk both attention regimes match Results; rbc-irreversible-investment reproduces `K_ss=37.989` exactly; diamond-mortensen-pissarides recovers `k=0.2106` and `u_ss=0.0649` to match Model Setup table.
- Audit: 8-of-15 deep audit dispatched to two Sonnet auditors (4 each, focused on multi-step arithmetic). Both reports CLEAN. No bugs found. Higher sample rate (53%) than Phase 1's 8-of-47 (17%); zero bugs vs Phase 1's one nash-in-nash bug.

## Open questions

(none)

## Landmines

- Phase 2 substitution disclosures: each adapted README documents its parameter substitution in one sentence. If future edits to those Model Setup tables happen, verify the worked example's substitution disclosure still reads correctly. Tutorials affected: `structural-econometrics/auction-valuation-recovery/`, `agent-based-models/cobweb-arifovic-ga-learning/`, `dsge/blanchard-kahn-determinacy/` (drops Taylor wedge), `time-series/reduced-form-var/` (uses VAR(1) where headline is VAR(2)).
- `dsge/blanchard-kahn-determinacy/` worked example: the boxed verdict is for the *reduced* 2x2 system (n_x=0). The full system in Results has n_x=1 (Taylor wedge) so the active-vs-passive determinacy thresholds in Results don't directly equal the worked example's. Internally consistent; future edits should preserve the "drop Taylor wedge" disclosure.
- `dynamic-programming/diamond-mortensen-pissarides/` worked example uses inverse calibration (solves for k given θ_ss=1) rather than the standard forward direction (solve for θ given k). This is unusual but matches the README's calibration convention. Future edits should preserve.

## Suggested next move

If user signals Wave 2 sub-tutorial integration: open `~/.claude/plans/wave-2-integration-axial-falcon.md` and dispatch the spec'd 25-tutorial queue. Otherwise the catalog is in close-out mode: 65 worked examples across 112 tutorials, validator clean, no format violations.
