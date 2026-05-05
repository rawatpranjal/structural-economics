# Tutorial QC: industrial-organization/dynamic-entry-exit

## Verdict

major issues

The tutorial is reproducible, economics-first, and its displayed numerical
tables match the generated artifacts. The major issue is method clarity: the
README presents exit as a logit rule using the expected next-state continuation
value, but `run.py` reports exit probabilities from a same-state continuation
approximation. Because those probabilities drive the transition matrix,
stationary distribution, and policy plots, the approximation should either be
made explicit or replaced by a solver consistent with the stated equations.

## Scorecard

| Dimension | Status | Notes |
|---|---|---|
| Crux and intuition | Pass | The tutorial clearly separates sunk entry costs from incumbent exit option value and keeps the economic object in front of the computation. |
| Pseudocode and method clarity | Major issue | The pseudocode is symbolic and problem-specific, but it does not disclose the same-state continuation approximation used to compute reported exit probabilities. |
| Results and writeup coherence | Pass with caveat | README numbers match `run.py`, the CSV tables, and regenerated output. The caveat is that the reported exit probabilities are not the probabilities implied by the README's displayed Delta equation. |
| Reproducibility | Pass | Static QC, repro QC, and catalog validation all exited 0. Repro regenerated no changed artifacts. |
| Root catalog row | Pass | The title links to `industrial-organization/dynamic-entry-exit/`; the thumbnail links to `figures/value-function.png`, a useful full figure; the row helps distinguish this dynamic IO tutorial from nearby entries. |

## Evidence

- Claim checked: "The value iteration converged in **667 iterations** with sup-norm error **9.94e-09**." Re-running the model gives 667 iterations and error `9.94071669425e-09`, matching the README and `qc_repro` stdout after rounding.
- Claim checked: the equilibrium statistics table reports `E[N] = 7.98`, standard deviation `0.15`, modal `N = 8`, static zero-profit `N = 10.3`, profit at mean `0.790`, net profit `0.290`, HHI `1250`, expected incumbent exit probability `0.0027`, expected exits `0.021`, expected entry `0.02`, and max stationary simulation gap `5.42e-04`. Recomputed values are `7.97883632`, `0.14528966`, `8`, `10.31370850`, `0.79012346`, `0.29012346`, `1250`, `0.00265012`, `0.02116368`, `0.02116368`, and `0.00054213694`, matching after rounding.
- Claim checked: the selected-state row for `N = 10` reports profit `0.529`, net profit `0.029`, `V(N) = 3.960`, exit probability `0.0221`, and expected entry `0`. Recomputed values are `0.52892562`, `0.02892562`, `3.95954765`, `0.02208561`, and `0.00`, matching after rounding.
- Claim checked: the selected-state row for `N = 30` reports profit `0.067`, net profit `-0.433`, `V(N) = 1.562`, exit probability `0.2592`, and expected entry `0`. Recomputed values are `0.06659729`, `-0.43340271`, `1.56187392`, `0.25915261`, and `0.00`, matching after rounding.
- Method check: README lines 21-40 define exit using `Delta(N) = pi(N) - f + beta E[V(N_{t+1}) | N_t=N, stay]`, while `run.py` lines 144-156 compute `_exit_prob` with `EV_approx = V[N - 1]`. The value update itself integrates over rival survivors at `run.py` lines 94-108, so the reported exit probabilities are not generated from the same continuation object as the displayed equation.
- Magnitude check for the method mismatch: applying the README's expected-continuation Delta to the final policy gives `p_exit(15) = 0.083714` versus the reported `0.108483`, `p_exit(20) = 0.137776` versus `0.178266`, and `p_exit(30) = 0.209743` versus `0.259153`.
- Caption check: visible captions are absent. Each figure has concise alt text and surrounding prose that interprets the figure, which is consistent with the repo convention.
- Catalog check: root README line 106 links the thumbnail `industrial-organization/dynamic-entry-exit/figures/thumb.png` to `industrial-organization/dynamic-entry-exit/figures/value-function.png`; both files exist, and the full figure is a readable 917 x 704 PNG showing the value function, sunk cost line, and static zero-profit marker.

## Findings

1. Major: the exit-probability equation and the reported exit-probability calculation are not aligned. The tutorial tells the reader that exit follows the logit rule using the expected continuation value under the stochastic transition, but `_exit_prob` uses a same-state continuation approximation. This is not just a notation issue: the approximation changes reported exit probabilities by about 2.5 percentage points at `N = 15` and 4.9 percentage points at `N = 30`.

## Recommended Follow-Up Edits

- Either change the solver so exit probabilities are computed from the same expected-continuation object used in the displayed Delta equation, or explicitly document the same-state approximation in the Equations and Solution Method sections.
- If the approximation is retained, add a short diagnostic or sentence explaining why it is acceptable for the tutorial's low-mass high-`N` states.

## Commands Run

| Command | Exit status |
|---|---:|
| `python3 scripts/qc_subject.py industrial-organization --only dynamic-entry-exit --json --out /tmp/qc-static-dynamic-entry-exit.json` | 0 |
| `python3 scripts/qc_repro.py industrial-organization --only dynamic-entry-exit --timeout 300 --restore --out /tmp/qc-repro-dynamic-entry-exit.json` | 0 |
| `python3 scripts/validate_catalog.py` | 0 |
| Python numerical spot-check importing `industrial-organization/dynamic-entry-exit/run.py` | 0 |
