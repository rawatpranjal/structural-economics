# Tutorial QC: industrial-organization/dynamic-discrete-choice

## Verdict

pass

The tutorial is economics-first, reproducible, and coherent with its generated artifacts. I found no required fixes: the dynamic replacement intuition is clear, both NFXP and CCP steps are explained with problem-specific pseudocode, the displayed numbers match `run.py` and the CSVs, and the catalog row is useful.

## Scorecard

| Dimension | Status | Notes |
|---|---|---|
| Crux and intuition | Pass | The tutorial keeps the economic object in front: replacement changes future mileage states, so observed hazards combine current keep costs and continuation values. |
| Pseudocode and method clarity | Pass | The Solution Method gives separate symbolic algorithms for nested fixed-point likelihood and Hotz-Miller CCP estimation, using notation that matches the Equations section. |
| Results and writeup coherence | Pass | Prose claims match the generated figures, `tables/parameter-estimates.csv`, `tables/simulation-moments.csv`, and fresh recomputation from `run.py`. Visible captions are absent; figure exposition appears in nearby text. |
| Reproducibility | Pass | Static QC, repro QC, and catalog validation all exited 0. Repro reported no changed artifacts. |
| Root catalog row | Pass | The title links to `industrial-organization/dynamic-discrete-choice/`; the thumbnail opens `figures/value-and-ccp.png`, a useful full figure; the row description helps a reader distinguish this tutorial from neighboring dynamic IO examples. |

## Evidence

- Claim checked: "Very high mileage states are scarce: in this simulation only **0.29%** of bus-periods have mileage at least 10." Recomputed share is `0.002933333333333`, matching the README percent and `tables/simulation-moments.csv`.
- Claim checked: the diagnostics table reports repair rate `0.253181`, average mileage `2.21011`, high-mileage share `0.00293333`, and VFI iterations `228`. Fresh recomputation gives `0.253180952380952`, `2.210109523809524`, `0.002933333333333`, and `228`, matching after rounding.
- Claim checked: the parameter table reports full-solution ML `(2.01812, -0.15346)` and CCP `(2.01767, -0.15334)`. Recomputed estimates are `(2.01812141, -0.15346282)` and `(2.01766692, -0.15333595)`, matching after rounding.
- Claim checked: "replacement is rare for fresh engines and rises sharply once mileage makes keeping the engine costly." The true replacement probability rises from `0.119203` at the lowest mileage state to `0.851689` at the highest mileage state.
- Claim checked: "The full-solution and CCP policies are almost on top of the truth over the states that carry most of the simulated likelihood." The largest absolute policy deviation from the truth is `0.006813` for full-solution ML and `0.006549` for CCP over observed states.
- Static check: `qc_subject` finds all standard sections present, pseudocode present, three figure references present, and no missing figures.
- Repro check: `qc_repro` reports `exit_code: 0`, `timed_out: false`, `changed: []`, `restored: true`, and stdout with VFI convergence in `228` iterations, simulated repair rate `0.253`, and the same full-solution and CCP estimates.
- Catalog check: root README line 107 links the thumbnail `industrial-organization/dynamic-discrete-choice/figures/thumb.png` to `industrial-organization/dynamic-discrete-choice/figures/value-and-ccp.png`; both files exist, and the full figure is a readable 1632 x 654 PNG showing conditional values and the replacement hazard.

## Findings

No findings.

## Recommended Follow-Up Edits

No required follow-up edits for this QC pass.

## Commands Run

| Command | Exit status |
|---|---:|
| `python3 scripts/qc_subject.py industrial-organization --only dynamic-discrete-choice --json --out /tmp/qc-static-dynamic-discrete-choice.json` | 0 |
| `python3 scripts/qc_repro.py industrial-organization --only dynamic-discrete-choice --timeout 300 --restore --out /tmp/qc-repro-dynamic-discrete-choice.json` | 0 |
| `python3 scripts/validate_catalog.py` | 0 |
| Python numerical spot-check importing `industrial-organization/dynamic-discrete-choice/run.py` | 0 |
