# bullshit-detector — houtman-maks-rational-subsets — recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. Both original findings (pseudocode omitted tie-breakers and the more-than-one-observation component filter) are resolved: the updated pseudocode explicitly names both, and the code at run.py:154 and run.py:165-169 matches exactly.

## Header
- Claim sources: `choice/houtman-maks-rational-subsets/README.md`
- Code / artifact root: `choice/houtman-maks-rational-subsets/run.py`
- Data artifacts: `choice/houtman-maks-rational-subsets/tables/houtman-maks-diagnostics.csv`
- Seed audit: `bullshit-detector_houtman-maks-rational-subsets_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | GARP violations = 9 | HOLDS | — | — |
| 2 | Exact HM index = 11 | HOLDS | — | — |
| 3 | Swapped rows are 3 and 4 (1-indexed) | HOLDS | — | — |
| 4 | Greedy removes observation 4 | HOLDS | — | — |
| 5 | Cobb-Douglas shares (0.45, 0.35, 0.20) | HOLDS | — | — |
| 6 | GARP definition via direct revealed preference | HOLDS | — | — |
| 7 | Transitive closure of R^D | HOLDS | — | — |
| 8 | Violation participation table (all 12 rows) | HOLDS | — | — |
| 9 | Greedy pseudocode: more-than-one-observation component filter | HOLDS | — | — |
| 10 | Greedy pseudocode: tie-breaker by strict-arc degree, then lower obs id | HOLDS | — | — |
| 11 | Retained set keeps 11 of 12 observations | HOLDS | — | — |

## Findings

### Original Finding 1 (from seed audit): Greedy pseudocode omitted tie-breakers — RESOLVED

- **Original claim:** pseudocode said only "remove the observation with the most violation participation" with no mention of the strict-degree or obs-id tie-breakers.
- **Current state:** README Solution Method pseudocode now reads: "remove the observation with the most violation participation, breaking ties by strict-arc degree, then by lower observation id" — matching `score()` at `run.py:165-167` which returns `(participation[local_node], strict_degree, -remaining[local_node])`.
- **Resolution:** RESOLVED. DILUTED → HOLDS.

### Original Finding 2 (from seed audit): Greedy pseudocode omitted `len(component) > 1` filter — RESOLVED

- **Original claim:** pseudocode said "containing a strict internal arc" without making the more-than-one-observation condition explicit.
- **Current state:** README Solution Method pseudocode now reads: "restrict attention to components of more than one observation with a strict internal arc" — matching `if len(component) > 1 and has_strict_arc:` at `run.py:154`.
- **Resolution:** RESOLVED. DILUTED → HOLDS.

## Cross-cutting patterns

None. All claims verified end-to-end. Numeric claims injected dynamically via f-strings; CSV cross-check confirms all 12 participation rows and all action labels match exactly.

## TDD execution sequence (for the next agent)

None required. Score is 0%. Both honest-fix tests now pass:
- `assert "strict-arc degree" in text and "breaking ties" in text` — PASSES.
- `assert "more than one observation" in README.read_text()` — PASSES.
Both violated-invariant tests now correctly fail, confirming fixes are in place.
