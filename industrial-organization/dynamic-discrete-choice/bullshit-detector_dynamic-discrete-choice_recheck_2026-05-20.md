# bullshit-detector — dynamic-discrete-choice — recheck — 2026-05-20

**Bullshit score: 0%** — original Finding 1 (DILUTED, MED: prose asserted unverified numerical MCE-IRL equivalence "to within solver tolerance" and pseudocode "theta_IRL == theta_NFXP up to solver tolerance" with no MCE-IRL estimator implemented) resolved by removing numerical phrasing, retitling MCE-IRL topic as an interpretation, and explicitly stating no separate run is performed; all formula, parameter, and diagnostic claims verified HOLDS against code and both CSV artifacts.

## Header
- Claim sources: `industrial-organization/dynamic-discrete-choice/README.md`
- Code / artifact root: `industrial-organization/dynamic-discrete-choice/run.py`
- Data artifacts: `industrial-organization/dynamic-discrete-choice/tables/parameter-estimates.csv`, `industrial-organization/dynamic-discrete-choice/tables/simulation-moments.csv`
- Seed audit: `bullshit-detector_dynamic-discrete-choice_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | MCE-IRL numerical equivalence asserted without estimator | HOLDS (fixed) | — | no |
| 2 | NFXP Bellman update matches equations | HOLDS | — | no |
| 3 | CCP Hotz-Miller linear system | HOLDS | — | no |
| 4 | MPEC Bellman constraints | HOLDS | — | no |
| 5 | MCE-IRL now stated as interpretation, not estimator | HOLDS | — | no |
| 6 | theta_0 true=2.00, NFXP=2.01812, CCP=2.01767, MPEC=2.01808 | HOLDS | — | no |
| 7 | theta_1 true=-0.15, NFXP=-0.15346, CCP=-0.15334, MPEC=-0.15346 | HOLDS | — | no |
| 8 | Repair rate=0.253181, avg mileage=2.21011, share>=10=0.00293333 | HOLDS | — | no |
| 9 | VFI iterations=228, MPEC iterations=4, MPEC Bellman residual=1.89644e-11 | HOLDS | — | no |
| 10 | All three estimator successes=1 | HOLDS | — | no |

## Findings

### Finding 1 (original): DILUTED — MCE-IRL numerical equivalence asserted without estimator — RESOLVED

- **Original claim:** The original README title "Bus Engine Replacement: NFXP, CCP, MPEC, and the MCE-IRL Equivalence" and prose "returns the same theta to within solver tolerance" and pseudocode output "theta_IRL == theta_NFXP up to solver tolerance" asserted a verified numerical result while `run.py` contained no `estimate_mce_irl` function and `parameter-estimates.csv` contained no `theta_IRL` column.

- **Current README evidence (title fix):** `README.md:1` — "Bus Engine Replacement: NFXP, CCP, MPEC, and the MCE-IRL Interpretation". Title now says "Interpretation" not "Equivalence".

- **Current README evidence (numerical phrasing removed):** `README.md:148` — "This tutorial does not run a separate MCE-IRL estimator, because at this setup it would be the NFXP code with relabelled variables." Neither "to within solver tolerance" nor "theta_IRL == theta_NFXP" appears anywhere in `README.md`.

- **Current README evidence (MCE-IRL scope stated):** `README.md:151` — pseudocode block labeled "MCE-IRL on Rust's bus engine, written in NFXP terms (no separate run here)". `README.md:161` — "MCE-IRL appears here as an interpretation, not as a new estimator."

- **Current code evidence:** `run.py` — no `estimate_mce_irl` function exists. The algebraic correspondence claim (MCE-IRL soft-Bellman = NFXP inner fixed point) is a mathematical statement, not a numerical one, and is now correctly framed.

- **Resolution:** All unverified numerical phrasing removed. MCE-IRL is presented as an algebraic interpretation, the pseudocode is labeled "no separate run here", and the title no longer advertises a demonstrated numerical equivalence. Finding fully resolved.

- **Category:** HOLDS (post-fix)

## Grounded HOLDS findings (key structural claims verified fresh)

**H1 — NFXP Bellman update.** `README.md:32-34` — conditional value functions satisfy inclusive value equations with Euler constant. Code `run.py:57-63` — `inclusive = logsumexp(values, axis=1) + EULER_GAMMA`; `next_replace = F_replace @ inclusive`; `next_keep = F_keep @ inclusive`; `values_new = column_stack([flow_replace + beta*next_replace, flow_keep + beta*next_keep])`. Exact match. HOLDS.

**H2 — CCP Hotz-Miller linear system.** `README.md:55-58` — `W_theta = u_bar(p_hat) + beta * F_hat * W_theta` solved as linear system. Code `run.py` (`hotz_miller_ccp`) — `A = np.eye(n_states) - beta * F_hat`; `W = np.linalg.solve(A, flow_bar)`. Exact match. HOLDS.

**H3 — MPEC Bellman constraints.** `README.md:69-76` — theta and v chosen jointly; Bellman residuals as equality constraints. Code `run.py` (`estimate_mpec`) — uses `scipy.optimize.minimize` with `constraints` specifying Bellman residuals equal zero via `bellman_residual` function. Exact match. HOLDS.

**H4 — MCE-IRL as interpretation.** `README.md:148` — "This tutorial does not run a separate MCE-IRL estimator, because at this setup it would be the NFXP code with relabelled variables." `README.md:161` — "MCE-IRL appears here as an interpretation, not as a new estimator." Code: no `estimate_mce_irl` function. Claim and code agree. HOLDS.

**H5 — theta_0 parameter estimates.** `README.md:183` — True=2.00, Full-solution ML=2.01812, CCP=2.01767, MPEC=2.01808. `parameter-estimates.csv:2` — `theta_0,2.0,2.01812,0.01812,2.01767,0.01767,2.01808,0.01808`. Exact match. HOLDS.

**H6 — theta_1 parameter estimates.** `README.md:184` — True=-0.15, Full-solution ML=-0.15346, CCP=-0.15334, MPEC=-0.15346. `parameter-estimates.csv:3` — `theta_1,-0.15,-0.15346,-0.00346,-0.15334,-0.00334,-0.15346,-0.00346`. Exact match. HOLDS.

**H7 — Repair rate and mileage moments.** `README.md:194-196` — Repair rate=0.253181, Average mileage=2.21011, Share>=10=0.00293333. `simulation-moments.csv:2-4` — 0.2531809523..., 2.210109523..., 0.0029333333.... Rounded to 6 significant figures: 0.253181, 2.21011, 0.00293333. Exact match. HOLDS.

**H8 — VFI and MPEC diagnostics.** `README.md:197-201` — VFI iterations=228, MPEC iterations=4, MPEC max Bellman residual=1.89644e-11. `simulation-moments.csv:5,9,10` — 228.0, 4.0, 1.8964385617437074e-11. All match. HOLDS.

**H9 — All estimator successes.** `README.md:199-201` — Full ML success=1, CCP success=1, MPEC success=1. `simulation-moments.csv:6-8` — 1.0, 1.0, 1.0. Exact match. HOLDS.

**H10 — Share>=10 prose claim.** `README.md:169` — "only **0.29%** of bus-periods have mileage at least 10". CSV: 0.0029333... = 0.29333...% — rounds to 0.29%. HOLDS.

## Cross-cutting patterns

- The one original DILUTED finding was scope-creep: a mathematical equivalence (algebraically true) was phrased as a numerical result (unverified). The fix correctly separates the two: the algebraic fact remains in the Takeaway and Solution Method, labeled as interpretation; the numerical phrasing is removed; the pseudocode is marked "no separate run here". This is the honest scope-respecting fix.
- No formula, parameter estimate, or diagnostic claim required changing. All published numbers are faithful to the committed CSVs.
- The test suite confirms the fix: `test_violated_numerical_equivalence_claimed_without_estimator` FAILS (correct - "to within solver tolerance" and "theta_IRL == theta_NFXP" no longer in README, so `asserts_numerical` is False); `test_fixed_no_unverified_numerical_equivalence_claim` PASSES; `test_fixed_title_does_not_advertise_demonstrated_equivalence` PASSES.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings read HOLDS. No further fixes required.
1. Test suite status: `test_violated_numerical_equivalence_claimed_without_estimator` FAILS (correct - fix applied, numerical phrasing removed); `test_fixed_no_unverified_numerical_equivalence_claim` PASSES (correct - "to within solver tolerance" and "theta_IRL == theta_NFXP" absent from README); `test_fixed_title_does_not_advertise_demonstrated_equivalence` PASSES (correct - "MCE-IRL Equivalence" absent from README).
2. No re-run needed. No code or data artifact changes required. Prose fix only: unverified numerical phrasing removed, MCE-IRL topic correctly scoped as algebraic interpretation.
