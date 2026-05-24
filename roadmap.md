# Roadmap

_Last updated: 2026-05-23. Mode: **close-out**. No new tutorials planned this period._

Roadmap pinned to "what is done" vs "what is left" for project close-out. Detailed authoring contracts live in `spec.md`; per-session implementation plans live in `~/.claude/plans/`.

## North-star

Pedagogical tutorial catalog of computational and structural economics models. Self-contained folders, hand-maintained `README.md`, `run.py` regenerates `figures/` and `tables/`. Public name: **Computational Economics**. Catalog organized by economic subject first, numerical method second.

## DONE

### Catalog (active)

- **97 tutorials shipped** across 14 subject blocks (138 catalog rows in root `README.md`).
- Tutorial contract enforced repo-wide: hand-maintained `README.md`, `run.py`, `figures/thumb.png`, `figures/`, optional `tables/`.
- Shared infrastructure in `lib/` (grids, discretization, VFI, STPFI, plotting).

### Wave 1 - Preliminary-Readings Curriculum

- **DONE 2026-05-22.** 14 prelims shipped (P0 x 5, P1 x 5, P2 x 4).
- Commits: `51e25ae`, `b36454f`, `f602b3d`, `4bac3ed`, with audit follow-through in `0565639`.
- See `spec.md > Wave 1` for the full curriculum, notational policy, and per-prelim contracts.

### Wave 2 - Behavioral and Dynamic Learning Tutorials

- **DONE 2026-05-22.** All four sub-tutorials shipped (W2.1 convex time budget, W2.2 consideration set, W2.3 probability weighting, W2.4 online pricing).
- W2.1-W2.3 shipped 2026-05-20/21 (predated formal spec); W2.4 shipped 2026-05-22 (commit `48podzero` -> see `48bce6c` for the actual ship).
- Original `plan.md` retired to stub. Spec absorbed into `spec.md > Wave 2`.

### Audits + hygiene

- **Adversarial audit (P2 prelims) DONE 2026-05-22.** Score 22% (78% claim confidence). NIT-level findings only; no blockers. Report at `docs/audits/wave1-p2-audit.md`.
- **Audit follow-through DONE 2026-05-23** (commit `0565639`): 3 notation renames, 5 Overview rewrites (gold-standard pattern for the remaining sweep), validator strictness bump, spec DONE markers.
- **Math-syntax migration DONE.** Repo-wide code-fence math syntax (`` $`...`$ ``, `` ```math ``). Validator hard-rejects bare-dollar math. `scripts/grep_math_bugs.sh` reports clean.
- **Em-dash and en-dash scrub DONE.** Zero hits across active `.md` (excluding `_legacy/`).

## LEFT (mop-up only - close-out scope)

### Mechanical residue

| Item | Status | Where | Sweep cost |
|------|--------|-------|------------|
| 44 Overview-math warns across 30 tutorials | testing Haiku sweep 2026-05-23 | `python scripts/validate_catalog.py --strict` | 1 batch if Haiku quality holds, else manual |
| Stale "rename to $`A`$ pending" annotation | open | `optimal-control/upwind-finite-differences/README.md:90` | 1-line edit |
| Zinkevich year drift (2007 vs 2008) | open | `game-theory/cfr-asymmetric-auction/README.md:223` vs `game-theory/regret-matching/README.md:119` | 1-line edit (canonical year: 2008) |
| Undefined `\Lambda` symbol | open | `industrial-organization/dynamic-games-estimation/README.md` | define at first use or rephrase |
| Stale remote branch `migrate-code-fence-math` | open | `origin/migrate-code-fence-math` (unreachable, superseded) | destructive; user decision |

The Overview-math sweep is the only sizable item. Everything else is sub-minute.

### Active session plan

The Haiku-pilot test for the Overview sweep lives at:
`~/.claude/plans/test-if-a-haiku-tingly-kahan.md`

If the pilot passes, three parallel Haiku batches sweep the remaining 27 tutorials and the three landmines above. Validator must drop to **zero** Overview-math warns post-sweep.

## DEFERRED (out of close-out scope)

Listed only so future agents do not treat these as "missing." User has explicitly excluded them from the current close-out.

- **Wave 3 (new tutorials).** Items live in `spec.md > Out of scope (Wave 2)`: Gallego-van Ryzin sawtooth pricing, network revenue management, markdown dynamic-program, behavioral runners (habit formation, reference dependence). No spec contract. Brainstorm required if re-opened.
- **Wave 2 sub-tutorial integration queue.** 25 sub-tutorials specced in `~/.claude/plans/wave-2-integration-axial-falcon.md` (2026-05-22, 52KB plan, predicted-numbers contracts, lib-promotion roadmap, TDD-loop execution). Spec locked, execution not started.
- **`_legacy/` archive (776MB).** No runtime deps; left alone.

## How to use this file

- A new session reads this first, then `spec.md` for authoring contracts, then any active plan in `~/.claude/plans/`.
- Update DONE rows by date and commit hash.
- Move LEFT items to DONE when they ship; do not delete them.
- Move DEFERRED items into LEFT only when the user explicitly re-opens that scope.

## Cross-links

- `CLAUDE.md` - tutorial contract, style rules, learned rules.
- `STYLE_GUIDE.md` - prose tone, pseudocode, figure and table style.
- `spec.md` - per-wave authoring contracts.
- `docs/audits/` - adversarial audit reports.
- `docs/qc-reports/` - per-topic QC sweeps (note: any report dated before 2026-05-22 predates the prelim cuts; do not trust for current state).
- `handoff.md` - most-recent session forward brief (written by `/end`).
