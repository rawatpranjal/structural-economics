# bullshit-detector -- sequence-space-jacobian-hank -- recheck -- 2026-05-20

**Bullshit score: 0%** -- all four Stage-3 findings resolved; prose, code, and CSV artifacts are consistent.

## Header
- Claim sources: `heterogeneous-agents/sequence-space-jacobian-hank/README.md`
- Code / artifact root: `heterogeneous-agents/sequence-space-jacobian-hank/run.py`
- Data artifacts: `heterogeneous-agents/sequence-space-jacobian-hank/tables/diagnostics.csv`
- Seed audit: `bullshit-detector_sequence-space-jacobian-hank_2026-05-20.md`
- Run by: claude-sonnet-4-6 (bullshit-detector skill, Stage-3 recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Original finding (short) | Category | Resolution | Recheck result |
|---|--------------------------|----------|------------|----------------|
| 1 | Anticipation-curve figure mislabeled as J^{C,r}_{0,s} columns | MISLABELED | Prose fixed | HOLDS |
| 2 | Complexity prose implied O(T\|state\|) total; sweep is O(T^2\|state\|) | DILUTED | Prose fixed | HOLDS |
| 3 | Q1/Q5 consumption ratio not in any committed artifact | DATA DRIFT | Row added to CSV | HOLDS |
| 4 | HANK vs RA inflation comparison not in any committed artifact | DATA DRIFT | Rows added to CSV | HOLDS |

## Findings

### Finding 1: Anticipation-curve figure label -- RESOLVED

- **Original claim (verbatim):** "These curves are the columns of $J^{C, r}_{0, s}$ before the forward distribution propagation." -- `README.md:201` (original)
- **Fixed prose (verbatim):** "Each curve is the skill-averaged date-0 policy perturbation $dc(a)$ for a unit $r$ pulse at lag $s$ -- the raw anticipation curve before it is integrated against the steady-state distribution to form a Jacobian entry." -- `README.md:203` (recheck)
- **Code evidence:** `run.py:1226` confirms `"skill-averaged date-0 policy perturbation $dc(a)$ for a unit "` in `add_results` string. Old string `"columns of $J^{C, r}_{0, s}$"` absent from `run.py` (grep returns no output).
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding1_violated_invariant` FAILS (fix applied -- old label gone).
- **Honest-fix test:** `test_finding1_honest_fix` PASSES.

### Finding 2: Forward-sweep complexity prose -- RESOLVED

- **Original claim (verbatim):** "anticipation curves are translation-invariant, so they are computed once and then convolved with the time-varying input path during the forward sweep." (implied O(T|state|) total)
- **Fixed prose (verbatim):** "The forward distribution sweep above then runs $T$ separate passes of length $T$, restarting $delta_D$ for each pulse date, so the sweep costs $O(T^2\,|state|)$ in total." -- `README.md:172-173` (recheck); `run.py:1155` contains `"O(T^2"`.
- **Code evidence:** `grep -n "O(T^2" run.py` returns line 1155.
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding2_violated_invariant` FAILS (fix applied).
- **Honest-fix test:** `test_finding2_honest_fix` PASSES.

### Finding 3: Q1/Q5 peak consumption ratio -- RESOLVED

- **Data evidence:** `diagnostics.csv` row 20: `Q1/Q5 peak consumption ratio, 3.77`. Value 3.77 satisfies `abs(3.77 - 4.0) < 1.5` (honest-fix tolerance). README prose says "close to four times" -- consistent with 3.77.
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding3_violated_invariant` FAILS (row now present).
- **Honest-fix test:** `test_finding3_honest_fix` PASSES.

### Finding 4: HANK vs RA inflation direction -- RESOLVED

- **Data evidence:** `diagnostics.csv` rows: `Peak HANK inflation response (annualized %), -0.688`; `Peak RA NK inflation response (annualized %), -0.249`. `abs(-0.688) = 0.688 > abs(-0.249) = 0.249` -- HANK inflation magnitude exceeds RA, confirming the README prose claim.
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding4_violated_invariant` FAILS (row now present).
- **Honest-fix test:** `test_finding4_honest_fix` PASSES.

## Cross-cutting patterns

None. All four findings were presentation-layer issues (wrong notation, implicit complexity claim, missing CSV rows). The underlying economic model, EGM solver, fake-news Jacobian, and equilibrium solve remain faithfully implemented and unchanged.

## TDD execution sequence (for the next agent)

Stage-3 fixes are complete. No further action required. Bullshit score is 0%.
