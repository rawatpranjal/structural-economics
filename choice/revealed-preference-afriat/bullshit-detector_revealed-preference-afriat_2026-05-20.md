# bullshit-detector -- revealed-preference-afriat -- 2026-05-20

**Bullshit score: 20%** -- Worst finding: tutorial title and all "Afriat's test" labels
(README lines 1, 47, 88) promise the constructive Afriat-inequalities method but the
code only implements GARP checking via Warshall transitive closure (MISLABELED, MED).
All numeric claims hold for seed=42. No FALSE or UNIMPLEMENTED findings. Diagram-only
cap does NOT apply (algorithm results present).

## Header

- Claim sources: `choice/revealed-preference-afriat/README.md`
- Code / artifact root: `choice/revealed-preference-afriat/run.py`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, six-pass method)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Afriat's test" title/label; code does GARP checking only | MISLABELED | MED | no (results correct; label misleads readers expecting Afriat inequalities) |
| 2 | Pseudocode omits TOL in both RP and violation conditions | DILUTED | LOW | no (TOL=1e-10, functionally negligible for continuous data) |
| 3 | Static "2 violations" in README; seed-dependent at runtime | DATA DRIFT | LOW | no (matches seed=42 run; latent drift risk if seed changes) |
| 4 | "Two bundles swapped" narrative omits undisclosed fallback path | DILUTED | LOW | no (fallback not triggered at seed=42; would change narrative if it fired) |
| 5 | Cobb-Douglas weights 0.337, 0.328, 0.335 | HOLDS | -- | -- |
| 6 | Warshall triple-loop transitive closure | HOLDS | -- | -- |
| 7 | GARP violation condition iR*j and m_j > p_j.x_i | HOLDS | -- | -- |
| 8 | Rational sample: 0 violations | HOLDS | -- | -- |
| 9 | Red arrows mark GARP contradiction pairs | HOLDS | -- | -- |

## Findings

### Finding 1: "Afriat's Test" label without Afriat inequalities

- **Claim source (verbatim):** "Consumer Rationalizability with Afriat's Test" -- `README.md:1`; "The code uses the graph version of Afriat's test." -- `README.md:47`; "Afriat's test asks whether finite household choice data can still be read as utility maximization" -- `README.md:88`
- **Code evidence (verbatim):**
  ```python
  def warshall_transitive_closure(R):
      """Warshall's algorithm: compute the transitive closure of relation R.

      Returns the indirect revealed preference relation. If R*[i,j] = 1,
      then i is (directly or indirectly) revealed preferred to j.
      """
      T = R.shape[0]
      R_star = R.copy()
      for k in range(T):
          for i in range(T):
              for j in range(T):
                  if R_star[i, k] and R_star[k, j]:
                      R_star[i, j] = 1
      return R_star
  ```
  `run.py:47-60`
- **Data evidence (if applicable):** No Afriat inequalities, no CCEI computation, no utility function construction anywhere in `run.py`. Confirmed by: `grep -n "Afriat\|CCEI\|efficiency\|inequality\|inequalities" run.py` returns only title/docstring strings and the reference citation -- zero algorithmic hits.
- **Category:** MISLABELED
- **Severity:** MED
- **Result-changing:** no -- the GARP algorithm is correct; per Afriat's theorem GARP is equivalent to rationalizability. But "Afriat's test" in the literature (Varian 1982, the cited reference) typically denotes the constructive Afriat inequalities approach that produces a utility function, not just a pass/fail decision. A reader following the Varian (1982) reference expecting to see the Afriat efficiency index (CCEI) will not find it. The label is defensible under a broad reading of "Afriat's theorem" but is misleading under the standard econometric usage.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not any(kw in open("run.py").read() for kw in ["lambda_t", "ccei", "afriat_inequalities", "efficiency_index"])
  # PASSES on current code (proves no Afriat inequalities); FAILS if Afriat inequalities are added
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(kw in open("run.py").read() for kw in ["afriat_inequalities", "lambda_t", "ccei"]) or "GARP" in open("README.md").read().split("##")[0]
  # PASSES if either Afriat inequalities implemented OR title changed to reflect GARP checking
  ```

### Finding 2: Pseudocode omits numerical tolerance (TOL)

- **Claim source (verbatim):** "1. For each pair (i,j), set R[i,j] = 1 if p_i . x_i >= p_i . x_j." -- `README.md:53`; "4. For each reachable pair (i,j), flag a violation if p_j . x_j > p_j . x_i." -- `README.md:58`
- **Code evidence (verbatim):**
  ```python
  TOL = 1e-10

  # ...in direct_revealed_preference:
  if expenditure_i + TOL >= cost_j_at_pi:   # lax: assigns RP if within TOL
      R[i, j] = 1

  # ...in check_garp:
  if exp_j > cost_i_at_pj + TOL:            # strict: violation only if exceeds by TOL
      violations.append((i, j))
  ```
  `run.py:22, 42, 86`
- **Data evidence (if applicable):** None needed -- purely algorithmic discrepancy between pseudocode and code.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no -- TOL=1e-10 is below floating-point precision for continuous random data (prices uniform [0.5,3], income uniform [5,15]). No boundary-straddling cases arise in practice.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "TOL" in open("run.py").read() and "TOL" not in open("README.md").read() and "tolerance" not in open("README.md").read().lower()
  # PASSES on current code+README (proves omission)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "tolerance" in open("README.md").read().lower() or "1e-10" in open("README.md").read()
  # PASSES if pseudocode discloses the tolerance; FAILS on current README
  ```

### Finding 3: Static "2 violations" claim; seed-dependent at runtime

- **Claim source (verbatim):** "| Corrupted sample | 2 violations | Two chosen bundles are swapped until GARP fails |" -- `README.md:42`; "The corrupted sample fails with 2 violating pairs." -- `README.md:62`
- **Code evidence (verbatim):**
  ```python
  f"| Corrupted sample | {len(violations_inc)} violations | Two chosen bundles are swapped ...\n"
  ...
  f"The Cobb-Douglas sample passes with {len(violations_con)} violations. "
  f"The corrupted sample fails with {len(violations_inc)} violating pairs."
  ```
  `run.py:369, 389-390`
- **Data evidence (if applicable):** Simulation with `np.random.default_rng(42)`, T=10, n_goods=3: swap path fires on attempt 2, violations=`[(4, 7), (7, 4)]`, count=2. README value matches. Count is seed-dependent; different seed yields different count. needs re-run to verify for any seed other than 42.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no -- correct for committed seed=42. Risk: if seed changes without re-running `python run.py`, README count drifts silently.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert open("README.md").read().count("2 violations") >= 2  # PASSES on current committed README
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert True  # nominal HOLDS for seed=42; no fix needed unless seed changes
  ```

### Finding 4: Undisclosed fallback path in generate_inconsistent_data

- **Claim source (verbatim):** "Two chosen bundles are swapped until GARP fails" -- `README.md:42`; "After two bundles are swapped, the same price variation now creates a strict revealed-preference cycle." -- `README.md:72`
- **Code evidence (verbatim):**
  ```python
  for attempt in range(200):
      prices, quantities, alpha = generate_consistent_data(T, n_goods, rng)
      i, j = rng.choice(T, size=2, replace=False)
      quantities_bad = quantities.copy()
      quantities_bad[i] = quantities[j]
      quantities_bad[j] = quantities[i]
      satisfies, violations, _, _ = check_garp(prices, quantities_bad)
      if not satisfies:
          return prices, quantities_bad, violations

  # Fallback: construct a known violation manually
  prices = np.array([
      [1.0, 2.0, 1.0],
      [2.0, 1.0, 1.0],
      [1.0, 1.0, 2.0],
  ] + [rng.uniform(0.5, 3.0, size=n_goods).tolist() for _ in range(T - 3)])
  quantities = np.array([
      [4.0, 1.0, 2.0],
      [1.0, 4.0, 2.0],
      [2.0, 2.0, 3.0],
  ] + [(rng.dirichlet(np.ones(n_goods)) * rng.uniform(5, 15)).tolist()
       for _ in range(T - 3)])
  ```
  `run.py:122-147`
- **Data evidence (if applicable):** Simulation confirms swap path fires at attempt=2 for seed=42; fallback not reached. Fallback produces a hardcoded dataset, not a swap-corrupted dataset. If fallback fired, the "two bundles swapped" narrative would be entirely wrong.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no for seed=42. Would be result-changing if seed changed and fallback fired.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "fallback" not in open("README.md").read().lower() and "200" not in open("README.md").read()
  # PASSES (proves README does not disclose fallback logic)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "fallback" in open("README.md").read().lower() or "attempt" in open("README.md").read().lower()
  # PASSES if README discloses swap-retry loop and fallback; FAILS on current README
  ```

## Cross-cutting patterns

- Single dominant cross-cutting pattern: **label inflation**. "Afriat's test" appears in the title (README:1), solution method (README:47), and takeaway (README:88), but the code implements only GARP checking. Afriat's actual 1967 contribution was the constructive inequalities approach; GARP checking is a necessary condition that follows from his theorem but is not the algorithm he proposed. All three prose occurrences of "Afriat's test" are affected by Finding 1.
- Findings 2 and 4 share a common cause: the pseudocode and narrative describe idealized algorithm behavior, while the code has pragmatic engineering choices (numerical tolerance, 200-attempt retry with fallback) not reflected in the prose. Neither changes results for seed=42.
- All numeric claims in the README are generated dynamically via f-string interpolation at report time (run.py:363-390). This is architecturally sound but creates silent DATA DRIFT risk if README is committed without a re-run after any parameter or seed change.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 20%.** Below 50% threshold -- no halt required. Finding 1 (MISLABELED MED) is the only finding worth actioning.

1. **Finding 1 -- Violated invariant test:**
   ```python
   def test_afriat_inequalities_absent():
       src = open("run.py").read()
       assert "lambda_t" not in src and "ccei" not in src.lower() and "afriat_inequalities" not in src
   ```
   Confirm PASSES. Then either: (a) implement Afriat inequalities (lambda_t, u_t system) and add them to Equations and Results, or (b) rename tutorial to "GARP Test for Consumer Rationalizability" throughout README.md title and run.py strings.

2. **Finding 2 -- Violated invariant test:**
   ```python
   def test_tol_not_in_readme():
       readme = open("README.md").read()
       assert "TOL" not in readme and "tolerance" not in readme.lower()
   ```
   Confirm PASSES. Fix: add one sentence to Solution Method noting tolerance=1e-10 in both conditions.

3. **Finding 4 -- Violated invariant test:**
   ```python
   def test_fallback_not_disclosed():
       readme = open("README.md").read()
       assert "fallback" not in readme.lower() and "200" not in readme
   ```
   Confirm PASSES. Fix: add sentence to Solution Method noting the 200-attempt retry and hardcoded fallback.

4. After fixes, re-run `python run.py` to regenerate README.md. Re-run this skill to confirm all findings now read HOLDS and score drops to <= 10%.
