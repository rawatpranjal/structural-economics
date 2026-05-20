# bullshit-detector — houtman-maks-rational-subsets — 2026-05-20

**Bullshit score: 10%** — all numeric claims hold; two pseudocode simplifications omit tie-breakers present in code (DILUTED, LOW severity, non-result-changing)

## Header
- Claim sources: `choice/houtman-maks-rational-subsets/README.md` (Overview, Equations, Model Setup, Solution Method, Results, Takeaway)
- Code / artifact root: `choice/houtman-maks-rational-subsets/run.py`
- Data artifacts: `choice/houtman-maks-rational-subsets/tables/houtman-maks-diagnostics.csv`
- Seed audit (if any): none
- Run by: claude-sonnet-4-6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | GARP violations = 9 | HOLDS | none | no |
| 2 | Exact HM index = 11 | HOLDS | none | no |
| 3 | Swapped rows are 3 and 4 (1-indexed) | HOLDS | none | no |
| 4 | Greedy removes observation 4 | HOLDS | none | no |
| 5 | Cobb-Douglas shares (0.45, 0.35, 0.20) | HOLDS | none | no |
| 6 | GARP definition in Equations | HOLDS | none | no |
| 7 | Transitive closure of R^D | HOLDS | none | no |
| 8 | Violation participation table (all 12 rows) | HOLDS | none | no |
| 9 | Greedy pseudocode omits strict-degree tie-breaker | DILUTED | LOW | no |
| 10 | Greedy pseudocode omits `len(component) > 1` filter | DILUTED | LOW | no |

## Findings

### Finding 1: Greedy pseudocode omits strict-degree and obs-id tie-breakers

- **Claim source (verbatim):** "remove the observation with the most violation participation" — `README.md:72`
- **Code evidence (verbatim):**
  ```python
  def score(local_node: int) -> tuple[int, int, int]:
      strict_degree = int(strict[local_node].sum() + strict[:, local_node].sum())
      return (int(participation[local_node]), strict_degree, -remaining[local_node])

  to_remove = max(bad_nodes, key=score)
  ```
  `run.py:165-169`
- **Data evidence:** Not applicable — no numeric result depends on tie-breaker resolution. In this run only one node maximizes participation (obs 4 = 6, obs 3 = 5), so tie-breakers are never activated.
- **Category:** DILUTED — pseudocode accurately names the primary key but silently drops the secondary (strict_degree) and tertiary (-obs_id) keys that the code uses when participation ties.
- **Severity:** LOW — the omitted tie-breakers are never reached in this run's single-removal scenario; no published number is affected.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "strict_degree" not in inspect.getsource(type("_Pseudocode", (), {"text": open("README.md").read()}))  # pseudocode text lacks tie-breaker; always passes
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "strict-degree" in open("README.md").read() or "tie" in open("README.md").read()
  ```

### Finding 2: Greedy pseudocode omits `len(component) > 1` guard

- **Claim source (verbatim):** "restrict attention to components containing a strict internal arc" — `README.md:70`
- **Code evidence (verbatim):**
  ```python
  if len(component) > 1 and has_strict_arc:
      bad_nodes.update(component)
  ```
  `run.py:154`
- **Data evidence:** Not applicable — a singleton component cannot have an internal strict arc (the `any(... if i != j)` guard at `run.py:148-153` rules it out). The `len(component) > 1` check is therefore redundant but not wrong.
- **Category:** DILUTED — pseudocode says "containing a strict internal arc" which is correct but does not make explicit the `len > 1` redundant filter. The omission creates no behavioral gap because the arc check is already sufficient.
- **Severity:** LOW — redundant guard; does not affect any output.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "len(component) > 1" not in open("README.md").read()  # pseudocode omits the guard; always passes
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "len(component) > 1" in open("README.md").read() or "singleton" in open("README.md").read()
  ```

## Cross-cutting patterns

- All numeric claims in Model Setup and Results are generated dynamically from code at report-write time (`run.py:375-382`). The values printed in `README.md` match the recomputed values exactly. This removes the entire class of DATA DRIFT finding that affects tutorials with hardcoded numbers.
- The preference-matrix construction (`run.py:41-44`) and GARP violation detection (`run.py:62-67`) both implement the exact definitions in the Equations section. Direction of indexing (`costs[t,s] = p_t . x_s`) was verified.
- The transitive closure (`run.py:48-53`) uses Warshall's algorithm on a reflexive base (weak already has a True diagonal), which correctly computes the reflexive transitive closure R. The `i != j` guard in `garp_violations` prevents spurious self-loop violations. Both match the mathematical definition.
- The two DILUTED findings share the same root: the pseudocode blocks in Solution Method are simplified descriptions of the code, not complete specifications. Both gaps are non-result-changing because the tie-breakers are never triggered in the single-run scenario.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10% (<50%).** No halt required. Findings are documentation-quality gaps only.
1. Finding 1: Write a test that generates a dataset where two observations tie on `participation`. Confirm `score()` uses `strict_degree` as the secondary key. This test currently passes on the code; the README claim is merely incomplete.
2. Finding 2: Confirm by test that a singleton component with `has_strict_arc = False` (always true for singletons since `i != j` in the arc check) is never added to `bad_nodes`, making the `len > 1` guard redundant. No fix needed in code.
3. Optional documentation fix: add a parenthetical to `README.md` Solution Method pseudocode noting the strict-degree tie-breaker. One sentence is enough.
4. Re-run this skill after any documentation edit to confirm score stays at 0-10%.
