# Tutorial QC: industrial-organization/three-part-tariffs

## Verdict

pass

The tutorial is economics-first, reproducible, and coherent with its generated artifacts. I found no required fixes: it explains the allowance as a state variable in a finite-horizon usage problem, gives symbolic problem-specific pseudocode, the numerical claims match `run.py` and `tables/plan-summary.csv`, and the catalog row is useful.

## Scorecard

| Dimension | Status | Notes |
|---|---|---|
| Crux and intuition | Pass | The tutorial centers the economic object: a three-part broadband tariff makes current usage forward-looking because today consumption changes remaining allowance and future overage exposure. |
| Pseudocode and method clarity | Pass | The Solution Method gives a finite-horizon backward-induction algorithm and plan-choice aggregation using the same plan, type, state, and value notation as the Equations section. |
| Results and writeup coherence | Pass | Prose claims match the generated figures, the CSV table, and fresh recomputation from `run.py`. Visible figure captions are absent; figure exposition appears in nearby text. |
| Reproducibility | Pass | Static QC, repro QC, and catalog validation all exited 0. Repro reported no changed artifacts. |
| Root catalog row | Pass | The title links to `industrial-organization/three-part-tariffs/`; the thumbnail opens `figures/usage-policy.png`, a useful full policy figure; the description helps a reader choose the tutorial. |

## Evidence

- Claim checked: "The finer-grid benchmark matches total usage within **0.00 GB**, while the largest cumulative-path gap is **3.00 GB**." Fresh recomputation gives focal total usage `85.0`, fine-grid total usage `85.0`, total usage gap `0.0`, and maximum cumulative-path gap `3.0`, matching the README.
- Claim checked: "net consumer value differs from the finer-grid solution by **0.408**, about **0.4%** of the baseline value." Fresh recomputation gives focal value `103.66609954537134`, fine-grid value `104.07459096074413`, absolute gap `0.40849141537279365`, and relative gap `0.003940453216280314`, matching after rounding.
- Claim checked: the plan-choice table reports Metered share `0.100`, average usage `45.000`, revenue `30.000`, and value `52.384`; Three-part share `0.690`, usage `82.971`, revenue `46.000`, and value `99.233`; Unlimited share `0.210`, usage `110.714`, revenue `52.000`, and value `163.701`. These match `tables/plan-summary.csv` and fresh recomputation after rounding.
- Claim checked: "Low-usage types choose the low fixed fee, middle types value the allowance, and high-usage types pay for unlimited access." Fresh recomputation assigns `h=3.0` to Metered, `h=3.5,4.0,4.5,5.0` to Three-part, and `h=5.6,6.2` to Unlimited.
- Claim checked: "Near the end of the cycle, the same remaining allowance has less option value, so the policy relaxes." At cumulative usage `C=80` under the three-part plan, the policy uses `1.0` GB on day 1 and `3.0` GB on day 30, consistent with the visual and prose.
- Static check: `qc_subject` finds all standard sections present, pseudocode present, three figure references present, and no missing figures.
- Repro check: `qc_repro` reports `exit_code: 0`, `timed_out: false`, `changed: []`, `restored: true`, and stdout with the same plan-choice summary as the CSV.
- Catalog check: root README line 108 links the thumbnail `industrial-organization/three-part-tariffs/figures/thumb.png` to `industrial-organization/three-part-tariffs/figures/usage-policy.png`; both files exist, and the full figure is a readable 1030 x 704 PNG showing the forward-looking usage policy.

## Findings

No findings.

## Recommended Follow-Up Edits

No required follow-up edits for this QC pass.

## Commands Run

| Command | Exit status |
|---|---:|
| `python3 scripts/qc_subject.py industrial-organization --only three-part-tariffs --json --out /tmp/qc-static-three-part-tariffs.json` | 0 |
| `python3 scripts/qc_repro.py industrial-organization --only three-part-tariffs --timeout 300 --restore --out /tmp/qc-repro-three-part-tariffs.json` | 0 |
| `python3 scripts/validate_catalog.py` | 0 |
| Python numerical spot-check importing `industrial-organization/three-part-tariffs/run.py` | 0 |
| Figure/thumbnail file check for `figures/thumb.png` and `figures/usage-policy.png` | 0 |
