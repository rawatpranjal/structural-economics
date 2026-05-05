# Tutorial QC: industrial-organization/dynamic-games

## Verdict

minor issues

The tutorial is economics-first, reproducible, and coherent with its generated artifacts. The only issue I found is a notation collision in the pseudocode: `lambda` is used as the value-iteration relaxation weight even though $\lambda$ is already the direct quality payoff in the equations and model setup.

## Scorecard

| Dimension | Status | Notes |
|---|---|---|
| Crux and intuition | Pass | Explains the economic object as state-contingent quality investment in a two-firm Markov-perfect dynamic game. Computation serves the MPE policy and continuation-value object. |
| Pseudocode and method clarity | Minor issue | The algorithm is symbolic and problem-specific, but the relaxation weight uses `lambda`, colliding with the payoff parameter $\lambda$. |
| Results and writeup coherence | Pass | Claims about investment at non-top states, quality leadership, symmetric states, and zero deviation gains match `run.py` and `tables/policy-by-state.csv`. Visible captions are absent; exposition sits in the surrounding text. |
| Reproducibility | Pass | Static QC, repro QC, and catalog validation all exited 0. Repro regenerated no changed artifacts. |
| Root catalog row | Pass | The title links to `industrial-organization/dynamic-games/`; the thumbnail links to `figures/investment-policy.png`, a useful full policy figure; the row description helps distinguish the tutorial from neighboring dynamic IO examples. |

## Evidence

- Claim checked: "firm 1 invests at every interior quality state and waits only at the top rung." Recomputed policy from `run.solve_game()` has firm 1 policy `[0, 0, 0, 0, 0]` on the top rung `q_1=4` and investment at all states with `q_1<4`. This matches the Results prose and investment-policy figure.
- Claim checked: selected state `(1,2)` has Firm 1 value `58.87`, Firm 2 value `78.59`, value advantage `-19.72`, and max deviation gain `0.00e+00`. Recomputed values are `58.8680`, `78.5866`, `-19.7186`, and `0.0000e+00`, matching the CSV and README table after rounding.
- Claim checked: selected state `(4,4)` has both firms waiting, values `78.52` and `78.52`, value advantage `0.00`, and max deviation gain `0.00e+00`. Recomputed values are `78.5244`, `78.5244`, `0.0000`, and `0.0000e+00`, matching after rounding.
- Repro check: `qc_repro` reports `exit_code: 0`, `timed_out: false`, `changed: []`, `restored: true`, and stdout with `Converged in 576 iterations`, final error `9.82e-09`, and largest one-step deviation gain `0.00e+00`.
- Static check: `qc_subject` finds all standard sections present, pseudocode present, three figure references present, and no missing figures.
- Catalog check: root README row links the thumbnail `industrial-organization/dynamic-games/figures/thumb.png` to `industrial-organization/dynamic-games/figures/investment-policy.png`; both files exist, and the full figure is a non-thumbnail PNG.

## Findings

1. Minor: the Solution Method pseudocode uses `lambda` for the relaxation update, `V_i^{n+1} = lambda T_i V^n + (1-lambda) V_i^n`, while the Equations and Model Setup already use $\lambda$ for the direct quality payoff. This is a notation mismatch in the one place where the tutorial should align algorithmic notation with the model notation.

## Recommended Follow-Up Edits

- Rename the relaxation weight in the pseudocode to a distinct symbol such as `rho`, `omega_relax`, or "relaxation weight", or define it explicitly as unrelated to the payoff parameter $\lambda$.

## Commands Run

| Command | Exit status |
|---|---:|
| `python3 scripts/qc_subject.py industrial-organization --only dynamic-games --json --out /tmp/qc-static-dynamic-games.json` | 0 |
| `python3 scripts/qc_repro.py industrial-organization --only dynamic-games --timeout 300 --restore --out /tmp/qc-repro-dynamic-games.json` | 0 |
| `python3 scripts/validate_catalog.py` | 0 |
| `python3 scripts/validate_catalog.py` | 0 |

