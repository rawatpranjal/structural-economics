# Handoff — 2026-05-21 — main

## Where we left off

Three rounds of GitHub-KaTeX render hygiene shipped to `main` as commits
`d513709`, `61ba589`, `3310fb3`. Validator now enforces 15 rules covering
the failure modes the user observed live; auto-fixer handles the
mechanical conversions. Stopped after pushing round 3.

## Active streams

### Render hygiene (PARALLEL, awaiting user verification)
- State: validator passes for all 97 tutorials; 198 inline-subscript
  warnings remain non-fatal.
- Next: user spot-checks four GitHub pages (cake-eating, revealed-price-
  preference, huggett-incomplete-markets, logit-supply-side) to confirm
  the fixes render. If any pattern still breaks, paste URL — extend
  rule and tighten warnings to errors.

### Inline subscript warnings (BLOCKED on user audit)
- 198 sites flagged by `inline_subscript_braces_errors` are warnings,
  not errors. Each is `_{...}` with non-alphanumeric body inside inline
  `$...$`. Hand-fixing all 198 would massacre prose flow; many may
  render fine on GitHub.
- Next: once user confirms a category breaks live, promote that
  category from warning to hard error and run a targeted sweep.

### Doc bloat (DEFERRED)
- Compression diff for `CLAUDE.md` Learned Rules section prepared and
  shown; user opted to skip. ~9 lines could go; not worth the cost.

## Decisions made this session

- `\,` thin-space banned outright (not just after close-delimiters). The
  CLAUDE.md prose had previously allowed `\,` between letters; we tightened
  because GitHub KaTeX is unreliable enough that the safer rule is total.
- Inline-subscript rule landed as warning, not error. Reason: 198
  candidate sites with no empirical confirmation that all 198 break;
  fail-stopping the validator would block legitimate commits.
- `qc-reports/` excluded from validator (historical reports quote bad
  math by design).
- Auto-fixer also repairs round-1 glued-Greek artifacts (`\alphas` ->
  `\alpha s`) caused by deleting `\,` between Greek macro and letter.

## Open questions

- Do all 198 inline-subscript warnings actually render broken on GitHub,
  or only a subset? Needs visual audit before tightening the rule.
- Should the bloat-compression be applied later (next session) or
  permanently skipped?

## Landmines / gotchas

- `scripts/fix_render_issues.py` is destructive (rewrites files in place);
  always run it on a clean working tree so the diff is reviewable.
- The auto-fixer's `$$`-reformatter can corrupt content if `$$` appears
  inside backtick-quoted prose. Round 1 broke CLAUDE.md this way; the
  fix (skipping inline code spans) is in `reformat_dollar_blocks` at
  `scripts/fix_render_issues.py:55`.
- `CLAUDE.md` is gitignored — edits to it never ship to GitHub. Don't
  expect changes there to influence the public repo.
- `.serena/` is now gitignored; agent caches there are local-only.

## Suggested next move

User opens the four GitHub URLs from the session summary and reports
back which render correctly. That decides whether to tighten the
inline-subscript rule to an error.
