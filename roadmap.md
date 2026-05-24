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

### Mop-up sweep (2026-05-23)

| Item | Status | Where | Outcome |
|------|--------|-------|---------|
| 44 Overview-math warns across 30 tutorials | DONE 2026-05-23 (commits `42ce1e1`, `70055bd`) | repo-wide | validator now 0 warns; 30 files touched, 54+/54- total (pure prose-for-symbol substitution) |
| Stale "rename to $`A`$ pending" annotation | DONE 2026-05-23 (`70055bd`) | `optimal-control/upwind-finite-differences/README.md:90` | parenthetical updated to current state |
| Zinkevich year drift (2007 vs 2008) | DONE 2026-05-23 (`70055bd`) | `game-theory/cfr-asymmetric-auction/README.md:223` | year aligned to 2008 (NIPS proceedings year, matching `regret-matching/`) |
| `\Lambda` undefined in dynamic-games-estimation | NOT NEEDED | `industrial-organization/dynamic-games-estimation/README.md:53` | verified defined at first use ("`\Lambda(\cdot)` is the logistic CDF"); audit was a false positive |

### Still open

| Item | Where | Cost |
|------|-------|------|
| Stale remote branch `migrate-code-fence-math` | `origin/migrate-code-fence-math` (unreachable, superseded) | destructive; user decision |

### Sweep tooling

The Haiku-pilot test plan lives at `~/.claude/plans/test-if-a-haiku-tingly-kahan.md`. Pilot passed on 3 tutorials; full sweep ran as 3 parallel batches over 27 tutorials with a tight "minimal surgical edit" prompt. End-state: zero Overview-math warns, math-syntax clean, em-dash scrub clean.

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
