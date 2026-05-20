# bullshit-detector - first-price-auctions - 2026-05-20

**Bullshit score: 15%** - one DILUTED/MED finding: prose claims the grid best response "sits on the analytic bid" for the focal case (n=3, v=0.8), but the residuals table the same README publishes shows n=3 has max error 1.583e-04, and the actual focal gap is 1.333e-04; all other claims HOLD against code and data.

## Header
- Claim sources: `game-theory/first-price-auctions/README.md` (Overview, Equations, Model Setup, Solution Method, Results prose)
- Code / artifact root: `game-theory/first-price-auctions/run.py`
- Data artifacts: `game-theory/first-price-auctions/tables/auction-summary.csv`, `game-theory/first-price-auctions/tables/best-response-residuals.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Equilibrium bid is b*(v)=(n-1)/n * v | HOLDS | - | no |
| 2 | Shading = v/n | HOLDS | - | no |
| 3 | Win probability = x(bhat)^(n-1), x = min(n/(n-1)*bhat, 1) | HOLDS | - | no |
| 4 | Expected payoff = (v-bhat)*x(bhat)^(n-1) | HOLDS | - | no |
| 5 | Deviation grid: 2,001 bids per value | HOLDS | - | no |
| 6 | Check values: 19 values in [0.05, 0.95] | HOLDS | - | no |
| 7 | Focal: n=3, v=0.8, 2 rivals | HOLDS | - | no |
| 8 | "Grid best response sits on the analytic bid" (focal case) | DILUTED | MED | no (prose only; table discloses real gap) |
| 9 | Shading table values (1/n at v=1) | HOLDS | - | no |
| 10 | Best-response residuals table matches CSV | HOLDS | - | no |

## Findings

### Finding 1: "The grid best response sits on the analytic bid" (focal case n=3, v=0.8)

- **Claim source (verbatim):** "For value 0.8 with 2 rivals, the payoff curve peaks at the analytic bid. Lower bids raise margin only when they still win. Higher bids buy win probability at a higher payment. The grid best response sits on the analytic bid." - `README.md:81-83`

- **Code evidence (verbatim):**
  ```python
  bids = np.linspace(0.0, focal_value, 300)
  payoffs = expected_payoff(focal_value, bids, focal_n)
  br_bid, br_payoff = grid_best_response(focal_value, focal_n)
  eq_bid = equilibrium_bid(focal_value, focal_n)

  fig2, ax2 = plt.subplots()
  ax2.plot(bids, payoffs, label=f"Expected payoff, v={focal_value:.1f}")
  ax2.axvline(eq_bid, color="black", linestyle=":", label=f"Exact bid {eq_bid:.3f}")
  ax2.scatter(br_bid, br_payoff, color="crimson", zorder=5, label=f"Grid BR {br_bid:.3f}")
  ```
  `run.py:172-180`

  `grid_best_response` uses `n_grid=2001` (default, `run.py:38`). For n=3, v=0.8:
  - `eq_bid = (3-1)/3 * 0.8 = 0.533333...`
  - Grid bids = `linspace(0, 0.8, 2001)`, closest grid point to eq_bid = 0.533200
  - `|br_bid - eq_bid| = 1.333e-04` (reproduced by replicating `run.py` logic)

- **Data evidence:** `tables/best-response-residuals.csv:4` shows `3,1.583e-04` - max residual for n=3 is 1.583e-04 (at v=0.95), confirming non-zero gap for n=3 at all check values. The focal gap (1.333e-04) is also non-zero. The same README that makes the "sits on" claim publishes the gap in its own residuals table.

- **Category:** DILUTED - code does compute the deviation check correctly and the residuals table honestly reports non-zero error; the prose overstates coincidence between BR and eq_bid for the focal case.

- **Severity:** MED - the residuals table in the same document contradicts the prose; no reader trusting only the table is misled. The gap is 0.017% of the bid axis and visually invisible in the figure.

- **Result-changing:** no - the residuals table is the authoritative numeric artifact; prose is misleading but does not change any published number.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(grid_best_response(0.8, 3)[0] - equilibrium_bid(0.8, 3)) < 1e-10  # PASSES on current code (returns 1.333e-04 gap, so assert FAILS) — wait, inverted: this FAILS on current code, meaning...
  ```

  Corrected form (passes on buggy code, fails on honest fix):
  ```python
  assert abs(grid_best_response(0.8, 3)[0] - equilibrium_bid(0.8, 3)) > 1e-10
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "approximately" in run.focal_results_prose or abs(grid_best_response(0.8, 3)[0] - equilibrium_bid(0.8, 3)) < 1e-10
  ```

  Simpler form targeting the prose fix specifically:
  ```python
  assert "near" in results_prose_focal or "close to" in results_prose_focal  # passes when prose hedges; fails when prose says "sits on"
  ```

## Cross-cutting patterns

- The tutorial's own residuals table (`best-response-residuals.csv`, `README.md:96-103`) already discloses the n=3 non-zero gap (1.583e-04). The prose claim ("sits on the analytic bid") contradicts this in the same document. This is an internal consistency failure: the figure and table are honest; the prose overwrites their message.
- All six formula claims (bid rule, shading, win probability, expected payoff, grid count, check-value count) ground cleanly in code with no discrepancy. The codebase is faithful to the economic model.
- The `np.maximum(value - bid, 0.0)` guard in `expected_payoff` (`run.py:35`) is not in the README equations. It is never binding because the bid grid is `linspace(0, value)`, so this is a code defensiveness measure, not a discrepancy.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15% (< 50%).** No halt required. Single prose fix needed.
1. Turn the violated invariant into a pytest: confirm `abs(grid_best_response(0.8, 3)[0] - equilibrium_bid(0.8, 3)) > 1e-10` passes on current code.
2. Turn the honest-fix pass condition into a second pytest: confirm the prose string for the focal results no longer says "sits on the analytic bid" after the fix.
3. Fix: replace "The grid best response sits on the analytic bid." (`run.py:189`) with a phrase that acknowledges the grid gap, e.g. "The grid best response lies within one grid spacing of the analytic bid." Then re-run `python run.py` to regenerate `README.md`.
4. After fix, re-run this skill to confirm the finding reads HOLDS and bullshit score drops to 0-10%.
