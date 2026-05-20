# bullshit-detector — normal-form-games — 2026-05-20

**Bullshit score: 10%** — all numeric results and formulas hold; one incomplete equation (max deviation gain combined quantity never defined in Equations section) at LOW severity; diagram-only cap not applied (tutorial has numeric results table).

## Header
- Claim sources: `game-theory/normal-form-games/README.md` (all sections)
- Code / artifact root: `game-theory/normal-form-games/run.py`
- Data artifacts: `game-theory/normal-form-games/tables/equilibrium-summary.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Deviation gain d1(i,j) defined; code heat map colors by it | DILUTED | LOW | no |
| 2 | Prisoner's Dilemma has no interior mixed NE | HOLDS | — | — |
| 3 | Matching Pennies: Pr(row Heads)=0.500; Pr(col Heads)=0.500; residual=0.0e+00 | HOLDS | — | — |
| 4 | Battle of the Sexes: Pr(row Opera)=0.600; Pr(col Opera)=0.400; residual=2.2e-16 | HOLDS | — | — |
| 5 | Stag Hunt: Pr(row Stag)=0.667; Pr(col Stag)=0.667; residual=4.4e-16 | HOLDS | — | — |
| 6 | Pure Nash found by testing zero-deviation profiles | HOLDS | — | — |
| 7 | Mixed residual = max absolute gap in two indifference equations | HOLDS | — | — |
| 8 | Row/col indifference equations in README match code formula | HOLDS | — | — |
| 9 | PD: mutual cooperation has higher joint payoff than mutual defection | HOLDS | — | — |
| 10 | CSV table matches README numbers exactly | HOLDS | — | — |

## Findings

### Finding 1: Heat map uses max(d1,d2) but Equations section never defines this combined quantity

- **Claim source (verbatim):** "The heat maps color each payoff table by the largest one-player deviation gain. Warmer cells have larger gains from switching action." — `README.md:87`
- **Claim source (verbatim):** "The row player's one-step deviation gain at $(i,j)$ is $d_1(i,j)=\max_{i' \in I} A_{i'j}-A_{ij}$" and analogously d2. — `README.md:17-27`
- **Code evidence (verbatim):**
  ```python
  row_gain = np.max(row_payoffs[:, j]) - row_payoffs[i, j]
  col_gain = np.max(col_payoffs[i, :]) - col_payoffs[i, j]
  gains[i, j] = max(row_gain, col_gain)
  ```
  `run.py:45-47`
- **Data evidence:** Heat map plotted from `gains` array; pseudocode step 2 says "Add (i,j) to E when max{d1(i,j),d2(i,j)}=0." — `README.md:77`
- **Category:** DILUTED — Equations section defines d1 and d2 individually but never defines the combined quantity max(d1,d2) that the figure actually displays. The pseudocode step 2 uses it without a prior definition. The claim "largest one-player deviation gain" in Results correctly describes the plotted quantity, but Equations does not introduce it as a named object.
- **Severity:** LOW — does not change any published number; the prose description in Results is accurate; a reader following only Equations → Results hits an undefined notation gap.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "max{d1" in open("game-theory/normal-form-games/README.md").read() or "\\max(d_1" in open("game-theory/normal-form-games/README.md").read()  # PASSES on current (string absent), FAILS on honest fix that defines the combined quantity
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(phrase in open("game-theory/normal-form-games/README.md").read() for phrase in ["\\max\\{d_1", "max\\{d_1", "combined deviation"])  # PASSES on honest fix, FAILS on current code
  ```

### Finding 2: Prisoner's Dilemma — no interior mixed equilibrium (HOLDS)

- **Claim source (verbatim):** "None" in Interior mixed equilibrium column. — `README.md:101`, `tables/equilibrium-summary.csv:2`
- **Code evidence (verbatim):**
  ```python
  denom_q = a[0, 0] - a[1, 0] - a[0, 1] + a[1, 1]
  if abs(denom_q) < 1e-12 or abs(denom_p) < 1e-12:
      return None
  ```
  `run.py:66,68-69` — with PD payoffs denom_q = -1 - 0 - (-3) + (-2) = 0.0 → function returns None ✓
- **Category:** HOLDS

### Finding 3: Matching Pennies — 50/50 mixed NE, residual 0.0e+00 (HOLDS)

- **Claim source (verbatim):** "Pr(row Heads)=0.500; Pr(column Heads)=0.500 | 0.0e+00" — `README.md:102`, `tables/equilibrium-summary.csv:3`
- **Code evidence (verbatim):**
  ```python
  q = (a[1, 1] - a[0, 1]) / denom_q   # (1 - (-1)) / (1-(-1)-(-1)+1) = 2/4 = 0.5
  p = (b[1, 1] - b[1, 0]) / denom_p   # (-1 - 1) / (-1-1-1+(-1)) = -2/-4 = 0.5
  ```
  `run.py:71-72` — verified analytically and numerically ✓
- **Category:** HOLDS

### Finding 4: Battle of the Sexes — p=0.600, q=0.400, residual 2.2e-16 (HOLDS)

- **Claim source (verbatim):** "Pr(row Opera)=0.600; Pr(column Opera)=0.400 | 2.2e-16" — `README.md:103`, `tables/equilibrium-summary.csv:4`
- **Code evidence (verbatim):**
  ```python
  # A=[[3,0],[0,2]], B=[[2,0],[0,3]]
  denom_q = 3 - 0 - 0 + 2 = 5
  q = (2 - 0) / 5 = 0.4
  denom_p = 2 - 0 - 0 + 3 = 5
  p = (3 - 0) / 5 = 0.6
  ```
  `run.py:66-72` — verified ✓. Float residual of 2.2e-16 is machine-epsilon noise ✓
- **Category:** HOLDS

### Finding 5: Stag Hunt — p=q=0.667, residual 4.4e-16 (HOLDS)

- **Claim source (verbatim):** "Pr(row Stag)=0.667; Pr(column Stag)=0.667 | 4.4e-16" — `README.md:104`, `tables/equilibrium-summary.csv:5`
- **Code evidence (verbatim):**
  ```python
  # A=[[4,0],[3,2]], B=[[4,3],[0,2]]
  denom_q = 4 - 3 - 0 + 2 = 3
  q = (2 - 0) / 3 = 0.6667
  denom_p = 4 - 3 - 0 + 2 = 3
  p = (2 - 0) / 3 = 0.6667
  ```
  `run.py:66-72` — verified ✓
- **Category:** HOLDS

### Finding 6: Pure Nash enumeration (HOLDS)

- **Claim source (verbatim):** Algorithm step 2: "Add (i,j) to E when max{d1(i,j),d2(i,j)} = 0." — `README.md:77`. PD: (Defect,Defect); BoS: (Opera,Opera),(Football,Football); SH: (Stag,Stag),(Hare,Hare); MP: None.
- **Code evidence (verbatim):**
  ```python
  row_best = row_payoffs[i, j] == np.max(row_payoffs[:, j])
  col_best = col_payoffs[i, j] == np.max(col_payoffs[i, :])
  if row_best and col_best:
      equilibria.append((i, j))
  ```
  `run.py:30-33` — mathematically equivalent to d1=d2=0 for these integer-valued payoffs ✓. All four games verified numerically.
- **Category:** HOLDS

### Finding 7: Mixed residual definition (HOLDS)

- **Claim source (verbatim):** "The reported mixed residual is the maximum absolute gap left in these two indifference equations." — `README.md:54`
- **Code evidence (verbatim):**
  ```python
  row_action_payoffs = row_payoffs @ np.array([q, 1.0 - q])
  col_action_payoffs = np.array([p, 1.0 - p]) @ col_payoffs
  residual = max(
      abs(row_action_payoffs[0] - row_action_payoffs[1]),
      abs(col_action_payoffs[0] - col_action_payoffs[1]),
  )
  ```
  `run.py:76-81` — residual = max|E[row action 0] - E[row action 1]|, |E[col action 0] - E[col action 1]|| ✓
- **Category:** HOLDS

### Finding 8: Indifference equation formulas (HOLDS)

- **Claim source (verbatim):** "$A_{11}q + A_{12}(1-q) = A_{21}q + A_{22}(1-q)$, $B_{11}p + B_{21}(1-p) = B_{12}p + B_{22}(1-p)$" — `README.md:48-50`
- **Code evidence (verbatim):**
  ```python
  denom_q = a[0, 0] - a[1, 0] - a[0, 1] + a[1, 1]
  denom_p = b[0, 0] - b[0, 1] - b[1, 0] + b[1, 1]
  q = (a[1, 1] - a[0, 1]) / denom_q
  p = (b[1, 1] - b[1, 0]) / denom_p
  ```
  `run.py:66-72` — algebraically solves the stated indifference equations ✓. 1-indexed README notation maps correctly to 0-indexed numpy arrays.
- **Category:** HOLDS

### Finding 9: PD joint payoff claim (HOLDS)

- **Claim source (verbatim):** "mutual defection is stable even though mutual cooperation gives more total payoff" — `README.md:87`
- **Code evidence:** A_pd[0,0]+B_pd[0,0] = -1+(-1) = -2 (CC); A_pd[1,1]+B_pd[1,1] = -2+(-2) = -4 (DD). -2 > -4 ✓ `run.py:95-96`
- **Category:** HOLDS

### Finding 10: CSV table matches README numbers (HOLDS)

- **Data evidence:** `tables/equilibrium-summary.csv` rows match `README.md:101-104` verbatim for all four games across all five columns ✓
- **Category:** HOLDS

## Cross-cutting patterns

- Tutorial is numerically clean. All four game payoff matrices, all equilibrium computations, and all table values are consistent across `run.py`, `README.md`, and `tables/equilibrium-summary.csv`.
- Single DILUTED finding (Finding 1) is a documentation gap, not a computation error: the combined quantity max(d1,d2) is used in figures and pseudocode but never formally defined in the Equations section. This is the only surface where a careful reader would notice a notation mismatch.
- No parametric leaks, no stale numbers, no mislabeled algorithms, no unimplemented claims.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10% — below 50% threshold. No halt required.**
1. Finding 1 only: add a definition of the combined quantity max{d1(i,j), d2(i,j)} to the Equations section of `run.py`'s `add_equations` block, regenerate `README.md`, and confirm the notation gap is closed.
2. No pytest tests needed for purely documentary gaps, but a sanity check: `assert "max" in readme_equations_section and "d_1" in readme_equations_section and "d_2" in readme_equations_section` after regeneration.
3. Re-run this skill on the updated tutorial; expect score 0%.
