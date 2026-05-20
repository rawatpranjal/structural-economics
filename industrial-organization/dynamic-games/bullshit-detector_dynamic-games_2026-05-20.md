# bullshit-detector -- dynamic-games -- 2026-05-20

**Bullshit score: 35%** -- DILUTED at MED severity: the tutorial's central "pure-strategy MPE" claim rests on a fallback code path (argmax joint payoff when no pure NE exists) that is never disclosed, and the "waits at top rung" results claim is broader than the 4-row evidence table covers.

## Header

- Claim sources: `industrial-organization/dynamic-games/README.md`
- Code / artifact root: `industrial-organization/dynamic-games/run.py`
- Data artifacts: `industrial-organization/dynamic-games/tables/policy-by-state.csv`
- Seed audit (if any): none
- Run by: bullshit-detector agent (Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Output is a pure-strategy MPE at every state | DILUTED | MED | yes -- if fallback fires at any of 21 unverified states, the output is not a NE there |
| 2 | Firm 1 waits at the top rung (all q1=4 states) | DILUTED | MED | yes -- top-rung invest vs wait depends on value gap vs kappa; only (4,4) verified |
| 3 | Damping alpha undisclosed; pseudocode names it without value | DATA DRIFT | LOW | no -- results are reproducible only if reader guesses alpha=0.35 |
| 4 | flow_profit formula, transition probs, G_i, beta, kappa | HOLDS | -- | -- |
| 5 | Table values (README vs CSV, 4 states) | HOLDS | -- | -- |
| 6 | Convergence criterion max Bellman residual < 1e-8 | HOLDS | -- | -- |

## Findings

### Finding 1: "pure-strategy MPE" claim is diluted by undisclosed no-NE fallback

- **Claim source (verbatim):** "A pure-strategy Markov-perfect equilibrium is a policy $a^{\ast}(\omega)=(a_1^{\ast}(\omega),a_2^{\ast}(\omega))$ and values satisfying $G_i(a_i^{\ast},a_j^{\ast};\omega,V)\geq G_i(a_i,a_j^{\ast};\omega,V)$ for all $a_i\in\{0,1\}$" -- `README.md:30-35`
- **Claim source (verbatim, pseudocode):** "Find pure Nash equilibria of the 2-by-2 state game. Select the equilibrium with the largest joint payoff if there is a tie." -- `README.md:62-63` (Solution Method pseudocode)
- **Code evidence (verbatim):**
  ```python
  def select_equilibrium(pay1: np.ndarray, pay2: np.ndarray) -> tuple[int, int]:
      equilibria = []
      for a1 in [0, 1]:
          for a2 in [0, 1]:
              if pay1[a1, a2] >= pay1[1 - a1, a2] - 1e-10 and pay2[a1, a2] >= pay2[a1, 1 - a2] - 1e-10:
                  equilibria.append((a1, a2))
      if not equilibria:
          total = pay1 + pay2
          idx = np.unravel_index(np.argmax(total), total.shape)
          return int(idx[0]), int(idx[1])
      return max(equilibria, key=lambda a: pay1[a] + pay2[a])
  ```
  `run.py:52-62`
- **Data evidence:** `tables/policy-by-state.csv` shows `Max deviation gain = 0.00e+00` for 4 states: (0,0), (1,2), (2,1), (4,4). The remaining 21 of 25 states have no row in the CSV. If the fallback at lines 58-61 fired at any of those states, the returned action profile is not a pure-strategy NE -- the "gain" would be positive, not zero. The CSV cannot confirm this.
- **Category:** DILUTED
- **Severity:** MED -- the fallback is logically coherent (picks joint-welfare-maximizing profile) but is categorically different from a Nash equilibrium. Disclosing it would not change the computed numbers in this calibration, but the claim "pure-strategy MPE" is false at any state where the fallback fires.
- **Result-changing:** yes -- if the fallback fires at any state, `V` at that state is not a fixed point of the best-response correspondence (deviation gains > 0 there). The tutorial's headline claim ("deviation gains are zero at the reported actions") would then be false at that state. needs re-run to verify.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "if not equilibria" in inspect.getsource(select_equilibrium)
  # PASSES on current code (fallback exists); FAILS if fallback is removed
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "if not equilibria" not in inspect.getsource(select_equilibrium) or "fallback" in open("README.md").read().lower()
  # PASSES when either fallback is removed or README discloses it; FAILS on current code
  ```

---

### Finding 2: "waits at top rung" claim broader than evidence

- **Claim source (verbatim):** "Firm 1 invests at every interior quality state. It waits at the top rung because the ladder cap binds." -- `README.md:74`
- **Code evidence (verbatim):**
  ```python
  def transition_probs(q: int, action: int, q_max: int) -> dict[int, float]:
      if action == 1:
          up = min(q + 1, q_max)
          return {up: 0.62, q: 0.38} if up != q else {q: 1.0}
      down = max(q - 1, 0)
      return {down: 0.12, q: 0.88} if down != q else {q: 1.0}
  ```
  `run.py:22-27`

  At q1=4 (q_max=4): `up = min(5,4) = 4`, so `up == q`, returning `{4: 1.0}`. Investing at the cap: stay at 4 with probability 1, cost kappa=2.2. Waiting at cap: stay at 4 with prob 0.88, drop to 3 with prob 0.12, no cost. Whether wait is optimal requires `0.12 * beta * (V[4,q2,0] - V[3,q2,0]) < kappa=2.2`. For large quality leads (e.g., q2=0), V[4,0,0] - V[3,0,0] could be large enough that firm 1 prefers to invest to protect its lead.

- **Data evidence:** `tables/policy-by-state.csv` contains only (0,0), (1,2), (2,1), (4,4). States (4,0), (4,1), (4,2), (4,3) are absent. The CSV cannot verify the "waits at top rung" claim for these states.
- **Category:** DILUTED
- **Severity:** MED -- the claim "waits at the top rung" is stated as a universal result over all states with q1=4 (5 states). Only 1 of those 5 states appears in the evidence table.
- **Result-changing:** yes -- if firm 1 invests at (4,0) or (4,1) to protect its quality lead, the "Firm 1 waits at top rung" statement in Results is false. The figure caption would be the primary mislead; the investment-policy figure would show "Invest" cells in the q1=4 row. needs re-run to verify.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert len(set(states_shown_at_q1_4)) == 1  # only (4,4) in CSV
  # PASSES on current CSV; FAILS if all 5 top-rung states are added
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert all(policy[4, q2, 0] == 0 for q2 in range(5))
  # PASSES if firm 1 waits at all (4,q2); FAILS if any top-rung state has invest
  ```

---

### Finding 3: Damping alpha=0.35 undisclosed

- **Claim source (verbatim):** "Update V_i^{n+1} = alpha T_i V^n + (1-alpha) V_i^n.  (alpha: iteration damping weight)" -- `README.md:65`
- **Code evidence (verbatim):**
  ```python
  V = 0.35 * V_new + 0.65 * V
  ```
  `run.py:80`
- **Data evidence:** Model Setup table (`README.md:39-49`) has no row for alpha. The pseudocode names alpha as a free parameter without stating its value. A reader trying to reproduce the tutorial must either run the code or guess alpha=0.35.
- **Category:** DATA DRIFT -- pseudocode and Model Setup table disagree (pseudocode introduces a parameter; table omits it). Code is internally consistent.
- **Severity:** LOW -- alpha does not appear in any reported number claim; it affects convergence speed not the equilibrium itself. Reproducibility is impaired but results are not falsified.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "alpha" not in open("README.md").read() or "0.35" in open("README.md").read()
  # FAILS on current README (alpha named but 0.35 never stated)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "alpha" not in open("README.md").read() or any("0.35" in line and "alpha" in line for line in open("README.md"))
  # PASSES when README states alpha=0.35 explicitly; FAILS on current README
  ```

---

### Finding 4 (HOLDS): Economic model formulas and parameters

- flow_profit: `run.py:15-19` matches `README.md:18` exactly (M=14, eta=0.75, lambda=0.35).
- Transition probs: `run.py:25,27` matches `README.md:23` (0.62 invest, 0.12 depreciate).
- G_i payoff: `run.py:47-48` matches `README.md:28` (flow profit minus kappa*a_i plus beta*EV).
- beta=0.90, kappa=2.2: `run.py:65` defaults match `README.md:44-45`.
- Convergence: `run.py:79,82` computes Bellman residual max|V_new-V| < 1e-8, matching pseudocode.
- **Category:** HOLDS

### Finding 5 (HOLDS): Table values README vs CSV

All four rows in `README.md:90-95` match `tables/policy-by-state.csv` exactly: values (60.03, 78.59, 58.87, 78.52), policies, and 0.00e+00 deviation gains.

- **Category:** HOLDS

## Cross-cutting patterns

- The tutorial's two MED-severity findings both stem from the same root: insufficient evidence coverage in the Results section. The table has 4 rows out of 25 states. A 5x5 full-policy table (or at minimum all top-rung states) would eliminate both the "waits at top rung" ambiguity and provide broader deviation-gain verification.
- The undisclosed no-NE fallback (`run.py:58-61`) is the only code path that can invalidate the "pure-strategy MPE" claim. A one-line assert in the solve loop (`assert equilibria, f"No pure NE at ({q1},{q2})"`) would make the claim self-enforcing during every run and eliminate the ambiguity entirely.
- The damping alpha=0.35 and the quality payoff lambda=0.35 share the same numeric value. The Model Setup table documents lambda but silently omits alpha. A reader scanning for "0.35" in the table would find only lambda, reinforcing the illusion that alpha is not a free parameter.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35% (below 50%). Surface to user before writing fix code, but forward work may continue.**

1. **Finding 1 -- violated invariant test:**
   ```python
   import inspect
   from industrial_organization.dynamic_games import run as dg
   def test_fallback_exists():
       assert "if not equilibria" in inspect.getsource(dg.select_equilibrium)
   ```
   This PASSES now (confirms fallback exists). Commit as the red test.

2. **Finding 1 -- honest-fix pass condition:**
   ```python
   def test_fallback_disclosed_or_absent():
       src = inspect.getsource(dg.select_equilibrium)
       readme = open("README.md").read().lower()
       assert "if not equilibria" not in src or "fallback" in readme or "no pure" in readme
   ```
   This FAILS now. Fix: either (a) remove fallback and assert its absence in tests, or (b) add a sentence in Solution Method disclosing the fallback. Option (a) is safer -- if no pure NE ever fires in this calibration, removing it is harmless and the MPE claim is then unconditional.

3. **Finding 2 -- violated invariant test:**
   ```python
   def test_top_rung_coverage():
       import pandas as pd
       csv = pd.read_csv("tables/policy-by-state.csv")
       top_rung_rows = [r for r in csv["State"] if r.startswith("(4,")]
       assert len(top_rung_rows) == 1  # only (4,4) currently
   ```
   PASSES now. Fix: add all 5 top-rung rows to the results table in `run.py:270`.

4. **Finding 2 -- honest-fix pass condition:**
   ```python
   def test_all_top_rung_wait(sol):
       policy = sol["policy"]
       assert all(policy[4, q2, 0] == 0 for q2 in range(5)), \
           f"Firm 1 invests at (4,{[q2 for q2 in range(5) if policy[4,q2,0]==1]})"
   ```
   Run `solve_game()`, pass result here. PASSES if the claim is correct; FAILS if firm 1 invests at any (4,q2).

5. **Finding 3 -- fix:** Add a row to the Model Setup table: `| Damping weight | alpha=0.35 | Controls iteration step size |`.

6. After all fixes, re-run `python run.py`, regenerate `README.md` and `tables/policy-by-state.csv`, and re-run this skill on the refreshed artifacts to confirm score drops to <=10%.
