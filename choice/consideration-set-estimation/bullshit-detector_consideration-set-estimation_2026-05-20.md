# bullshit-detector — consideration-set-estimation — 2026-05-20

**Bullshit score: 65%** — One FALSE/HIGH finding (Example 2 menu-index bug produces entirely wrong numbers in Results — p(a,{a,b})=0.000 and p(b,{b,c})=0.000 instead of 0.444 and 0.500 — with narrative that self-contradicts) plus two FALSE/MED findings (full-ranking-on-point-estimate claim contradicted by committed table; KL "essentially zero" contradicted by committed CSV showing KL=1.4908), plus one DILUTED/MED (true-DGP LL unverifiable without re-run) and one DILUTED/LOW (hardcoded SE claim).

## Header
- Claim sources: `choice/consideration-set-estimation/README.md` (prose, Equations, Results, tables)
- Code / artifact root: `choice/consideration-set-estimation/run.py`
- Data artifacts: `choice/consideration-set-estimation/tables/method-comparison.csv`, `choice/consideration-set-estimation/tables/ranking-recovery.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (automated audit)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | `p(a,{a,b})=0.000` and `p(b,{b,c})=0.000` in Example 2 replication | FALSE | HIGH | yes — two of three bar-chart values wrong; narrative internally contradicts numbers |
| 2 | "Both methods recover the full ranking on the point estimate" | FALSE | MED | yes — Method 2 gets 9/10 pairs (table shows a4,a5 swapped) |
| 3 | "KL divergence of essentially zero" for Method 2 | FALSE | MED | yes — committed CSV shows KL=1.4908 (355x Method 1's 0.0042) |
| 4 | "within one or two units of true-DGP value" for Method 2 LL | DILUTED | MED | needs re-run to verify — true-DGP LL never computed |
| 5 | "attention parameters lie within one SE of truth on every alternative" | DILUTED | LOW | needs re-run to verify — hardcoded prose, not dynamically checked |
| 6 | Method 2 score pseudocode sign vs code sign | MISLABELED | LOW | no — sign flip cancels with argsort(-score); functionally equivalent |
| 7 | Closed-form choice probability formula | HOLDS | - | no |
| 8 | gamma MLE closed form given ranking (N_c / (N_c + N_b)) | HOLDS | - | no |
| 9 | Method 2 attention from singleton-with-default frequencies | HOLDS | - | no |
| 10 | Bootstrap ranking accuracy M1=100%, M2=31% | HOLDS | - | no |
| 11 | Method-comparison LL values in table | HOLDS | - | no |

## Findings

### Finding 1: Example 2 menu indices are wrong — bar chart shows p=0.000 for two of three pairwise probabilities

- **Claim source (verbatim):** "At the calibration $\gamma(a) = 4/9$, $\gamma(b) = 1/2$, $\gamma(c) = 9/10$ with $a \succ b \succ c$, the closed-form probabilities are $p(a, \{a, b\}) = 0.000$, $p(b, \{b, c\}) = 0.000$, and $p(a, \{a, c\}) = 0.444$. Both $p(a, \{a, b\})$ and $p(a, \{a, c\})$ fall below the weak-stochastic-transitivity threshold of $0.5$, while $p(b, \{b, c\})$ meets it." — `README.md:143`

- **Code evidence (verbatim):**
  ```python
  p_ab = {"a": probs_ex[1, 0], "b": probs_ex[1, 1]}      # menu {a, b}
  p_bc = {"b": probs_ex[3, 1], "c": probs_ex[3, 2]}      # menu {b, c}
  p_ac = {"a": probs_ex[2, 0], "c": probs_ex[2, 2]}      # menu {a, c}
  ```
  `run.py:350-352`

- **Data evidence:** `enumerate_menus(3)` at `run.py:262-268` generates menus by bit-mask: mask_int=1→index 0={a}, mask_int=2→index 1={b}, mask_int=3→index 2={a,b}, mask_int=4→index 3={c}, mask_int=5→index 4={a,c}, mask_int=6→index 5={b,c}. Code uses index 1 for "menu {a,b}" (actual: {b}), index 3 for "menu {b,c}" (actual: {c}), index 2 for "menu {a,c}" (actual: {a,b}). Correct indices are 2 for {a,b}, 5 for {b,c}, 4 for {a,c}. Verified by direct computation: correct values are p(a,{a,b})=4/9≈0.444, p(b,{b,c})=0.500, p(a,{a,c})=4/9≈0.444. Buggy code produces: probs_ex[1,0]=p(a,{b})=0.000, probs_ex[3,1]=p(b,{c})=0.000, probs_ex[2,0]=p(a,{a,b})=0.444 (last one accidentally correct because {a,b} at index 2 happens to give the same value as {a,c}).

- **Category:** FALSE

- **Severity:** HIGH

- **Result-changing:** yes — two of three bar-chart bars display 0.000 instead of 0.444 and 0.500. The narrative is internally contradictory: it asserts "p(b,{b,c}) meets [the 0.5 threshold]" but displays 0.000, which demonstrably does not meet 0.5. The qualitative WST-violation conclusion is accidentally correct with the true values (0.444 < 0.5 and 0.444 < 0.5 still violate WST while p(b,{b,c})=0.500 holds), but the specific numbers the reader is asked to trust are fabricated by the index error.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert probs_ex[1, 0] == 0.0  # PASSES on buggy code (menu {b}: a not present)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(p_ab["a"] - 4/9) < 1e-6 and abs(p_bc["b"] - 0.5) < 1e-6  # correct indices
  ```

---

### Finding 2: "Both methods recover the full ranking on the point estimate" is false for Method 2

- **Claim source (verbatim):** "Both methods recover the full ranking on the point estimate at this sample size." — `README.md:151`

- **Code evidence (verbatim):**
  ```python
      "Both methods recover the full ranking on the point estimate at this sample size. "
  ```
  `run.py:713` — hardcoded string literal, not conditioned on `rank_correct_m2`.

- **Data evidence:** `tables/ranking-recovery.csv:3`: `Method 2 moments,a_1 > a_2 > a_3 > a_5 > a_4,9 / 10,31%`. True ranking is a_1 > a_2 > a_3 > a_4 > a_5; a_4 and a_5 are swapped. 9/10 correct pairs is not full-ranking recovery. The table the README itself displays (line 158) shows 9/10 for Method 2.

- **Category:** FALSE

- **Severity:** MED

- **Result-changing:** yes — the comparison narrative frames Method 1 vs Method 2 as "equally accurate on the point estimate, differing only in bootstrap robustness." The committed table proves Method 2 gets the point estimate wrong on one pair.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "9 / 10" in open("tables/ranking-recovery.csv").read()  # PASSES (bug exists)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Both methods recover the full ranking" not in open("README.md").read()
  ```

---

### Finding 3: Method 2 KL "essentially zero" contradicted by committed CSV

- **Claim source (verbatim):** "Both Method 1 and Method 2 reach a log-likelihood within one or two units of the true-DGP value and a Kullback-Leibler divergence of essentially zero." — `README.md:160`

- **Code evidence (verbatim):**
  ```python
      "Both Method 1 and Method 2 reach a log-likelihood within one or two units of the true-DGP value and a Kullback-Leibler divergence of essentially zero. "
  ```
  `run.py:739` — hardcoded prose.

- **Data evidence:** `tables/method-comparison.csv:1-4`:
  ```
  Method,Log-likelihood,KL divergence to true,Captures asymmetric impact
  Method 1 MLE,-15073.4,0.0042,yes
  Method 2 moments,-15847.0,1.4908,yes
  Luce / MNL benchmark,-16234.0,2.2885,no
  ```
  Method 2 KL=1.4908; Method 1 KL=0.0042. Ratio: 355:1. Method 2 KL is 65% of Luce's KL=2.2885. "Essentially zero" is not defensible. LL gap between M1 and M2 is 773.6 units; "within one or two units" of true-DGP is separately addressed in Finding 4.

- **Category:** FALSE

- **Severity:** MED

- **Result-changing:** yes — the reader concludes Method 2 and Method 1 are near-equivalent in fit accuracy. The committed table shows Method 2 is orders of magnitude worse in KL and substantially worse in LL.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert float(pd.read_csv("tables/method-comparison.csv").iloc[1]["KL divergence to true"]) > 1.0
  # PASSES (1.4908 > 1.0 — proves the claim is wrong)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "essentially zero" not in open("README.md").read()
  ```

---

### Finding 4: "Within one or two units of true-DGP value" unverifiable — true-DGP LL not computed

- **Claim source (verbatim):** "Both Method 1 and Method 2 reach a log-likelihood within one or two units of the true-DGP value" — `README.md:160`

- **Code evidence:** No call to `log_likelihood_at(ranking_true, gamma_true, ...)` exists anywhere in `run.py`. The true-DGP LL is never computed or stored. M1 LL=-15073.4, M2 LL=-15847.0 (773.6 units apart). Since the true-DGP LL <= M1 LL by construction, M2 LL is at least 773.6 units below true-DGP. **needs re-run to verify** but the gap between M1 and M2 alone makes "within one or two units" for M2 implausible.

- **Data evidence:** No committed artifact contains true-DGP LL.

- **Category:** DILUTED

- **Severity:** MED

- **Result-changing:** needs re-run to verify

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "true_dgp" not in open("tables/method-comparison.csv").read()  # PASSES (absent)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "true_dgp" in open("tables/method-comparison.csv").read()  # disclose true-DGP LL
  ```

---

### Finding 5: "Within one standard error of truth on every alternative" is hardcoded, not dynamically checked

- **Claim source (verbatim):** "the recovered attention parameters lie within one standard error of the truth on every alternative." — `README.md:135`

- **Code evidence (verbatim):**
  ```python
      f"and the recovered attention parameters lie within one standard error of the truth on every alternative."
  ```
  `run.py:553` — fixed string. `gamma_se_m1` is computed at `run.py:334` and `gamma_m1` at `run.py:322`. The condition `all(abs(gamma_m1 - gamma_true) <= gamma_se_m1)` is never evaluated. The prose is asserted unconditionally.

- **Category:** DILUTED

- **Severity:** LOW

- **Result-changing:** no — with N=500 and 80 bootstrap reps this is likely true at this seed, but it is unverified. **needs re-run to verify** strictly.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "within one standard error" in open("run.py").read()  # hardcoded string exists
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert all(abs(gamma_m1 - gamma_true) <= gamma_se_m1)  # dynamically verified before prose
  ```

---

### Finding 6: Method 2 score pseudocode sign reversed vs code (functionally equivalent)

- **Claim source (verbatim):** "score[j] <- sum_i impact[i, j] - sum_i impact[j, i]" — `README.md:127` (pseudocode block)

- **Code evidence (verbatim):**
  ```python
      score = np.zeros(J)
      for j in range(J):
          score[j] = avg_impact[j, :].sum() - avg_impact[:, j].sum()
      ranking_hat = np.argsort(-score)
  ```
  `run.py:210-213`

  `avg_impact[i,j]` = how much j's probability rises when i is removed. Code computes `sum_k avg_impact[j,k] - sum_i avg_impact[i,j]` = `sum_i impact[j,i] - sum_i impact[i,j]` = negative of the pseudocode formula. However, `argsort(-code_score) = argsort(pseudocode_score)`, which produces the identical ranking output.

- **Category:** MISLABELED

- **Severity:** LOW

- **Result-changing:** no — ranking output identical.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert score[ranking_hat[0]] > 0  # top-ranked item has positive code score (opposite of pseudocode)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert np.array_equal(np.argsort(-score), ranking_hat)  # invariant preserved after any sign fix
  ```

## Cross-cutting patterns

- All three FALSE findings (F1, F2, F3) involve hardcoded prose strings in `run.py` that do not reflect the actual simulation output. F1 is a code bug (wrong array indices produce wrong values which then get formatted into the prose). F2 and F3 are prose bugs (claims written before the simulation was run or from a different parameterization and never corrected).
- F1 (wrong menu indices) is structurally the most dangerous: the code computes wrong values and both the figure and the narrative faithfully reproduce those wrong values. The self-contradiction (narrative says p(b,{b,c}) "meets 0.5" but the displayed value is 0.000) is the smoking gun.
- The menu enumeration index-off pattern in F1 would silently affect any other analysis that hardcodes menu indices by integer literal rather than by mask lookup. A grep for `probs_ex[` or `menus_ex[` in any sibling tutorial is warranted.
- The true-DGP LL is never stored in any artifact (F4). Every LL comparison to "true-DGP" in this or sibling tutorials should be checked for the same omission.

## TDD execution sequence (for the next agent)

0. **Bullshit score 65% (>= 50%). Stop and surface to the user before writing any fix code.**
1. **F1 (highest priority):** Write a test asserting `p_ab["a"] == 0.0` (proves the bug). Fix `run.py:350-352` to look up menu indices by mask equality rather than by hardcoded integer: use `next(i for i, m in enumerate(menus_ex) if np.array_equal(m, [True, True, False]))` for {a,b}, etc. Confirm test fails after fix. Regenerate README and figure.
2. **F2:** Fix `run.py:713` from the hardcoded "Both methods recover the full ranking" to a conditional string based on `rank_correct_m2 == n_pairs`. Regenerate README.
3. **F3:** Fix `run.py:739`. Remove "essentially zero" and "within one or two units"; replace with actual computed values (e.g., "Method 2 KL=1.49, 355x larger than Method 1's KL=0.004"). Regenerate README.
4. **F4:** Add `ll_true = log_likelihood_at(ranking_true, gamma_true, counts, menus)` to `main()` and include it in `method_table`. Then prose can make a verifiable LL comparison.
5. **F5:** Replace hardcoded string at `run.py:553` with a conditional checking `all(abs(gamma_m1 - gamma_true) <= gamma_se_m1)`.
6. **F6 (optional):** Align pseudocode at `README.md` solution-method section to match code's sign: `score[j] <- sum_i impact[j, i] - sum_i impact[i, j]`, note that `argsort(-score)` then sorts best-to-worst.
7. After all fixes: re-run `python run.py` inside the folder, then `scripts/validate_catalog.py`. Re-audit with this skill. Target score <= 20%.
