# Tutorial QC: industrial-organization/nash-in-nash

## Verdict

minor issues

The tutorial is economics-first, reproducible, and coherent with its generated artifacts. The only issue I found is catalog-adjacent: `figures/thumb.png` exists and the root row opens a useful full figure, but the thumbnail itself is 200 x 86 rather than the 200 x 150 size specified in `CLAUDE.md`.

## Scorecard

| Dimension | Status | Notes |
|---|---|---|
| Crux and intuition | Pass | The tutorial centers the economic object: a failed hospital-insurer link changes network value, enrollment, outside options, and the transfer implied by Nash-in-Nash bargaining. |
| Pseudocode and method clarity | Pass | The Solution Method gives symbolic, problem-specific pseudocode for bilateral and merged-system disagreement networks, using the same $G^{-hd}$, $G^{-Hd}$, $q_d(G)$, $\Delta_{hd}$, $S_{hd}$, and $w_{hd}$ notation as the Equations section. |
| Results and writeup coherence | Pass | The claims match `run.py`, the generated figures, and both CSV tables. Visible figure captions are absent; exposition appears in the surrounding Results text and the image alt text is concise. |
| Reproducibility | Pass | Static QC, repro QC, and catalog validation all exited 0. Repro reported no changed artifacts for this tutorial. |
| Root catalog row | Minor issue | The title still links to `industrial-organization/nash-in-nash/`, and the thumbnail opens the useful full `figures/negotiated-prices.png` figure. The thumbnail file is nonblank but is 200 x 86, not the repo-standard 200 x 150. |

## Evidence

- Claim checked: "Hospital 1 receives higher transfers because it is the higher-quality hospital." Fresh formula recomputation gives H1 transfers `2.097` and `2.300`, versus H2 transfers `1.603` and `1.749`, matching the README and `tables/nash-in-nash-results.csv`.
- Claim checked: "Insurer 2 pays somewhat more because its higher premium gives it a larger per-enrollee margin." Fresh recomputation gives H1-I2 `2.300` > H1-I1 `2.097` and H2-I2 `1.749` > H2-I1 `1.603`, matching the README table.
- Claim checked: "Dropping Hospital 1 hurts more than dropping Hospital 2." Demand losses are `233.4` and `222.2` for Hospital 1, versus `146.6` and `141.8` for Hospital 2, matching the prose and table.
- Claim checked: "Losing the merged system is much worse because the insurer has no in-network hospital, so the negotiated payment is higher." Fresh recomputation gives no-system demands `10.4` and `8.6`; separate transfers `3.700` and `4.048`; merged-system transfers `4.529` and `4.780`; and changes `22.4%` and `18.1%`, matching `tables/merged-system-results.csv`.
- Static check: `qc_subject` finds all standard sections present, pseudocode present, three figure references present, and no missing figures.
- Repro check: `qc_repro` reports `exit_code: 0`, `timed_out: false`, `changed: []`, `restore_requested: true`, and `restored: true`.
- Catalog check: root README line 109 links the thumbnail `industrial-organization/nash-in-nash/figures/thumb.png` to `industrial-organization/nash-in-nash/figures/negotiated-prices.png`; both files exist, and the full figure is a readable 1632 x 698 PNG.

## Findings

1. Minor: `industrial-organization/nash-in-nash/figures/thumb.png` does not match the thumbnail size contract in `CLAUDE.md`. The file is nonblank and usable in the catalog, but it is 200 x 86 rather than 200 x 150.

## Recommended Follow-Up Edits

- Regenerate or crop `figures/thumb.png` to the repo-standard 200 x 150 dimensions while keeping it linked to the full negotiated-prices figure.

## Commands Run

| Command | Exit status |
|---|---:|
| `python3 scripts/qc_subject.py industrial-organization --only nash-in-nash --json --out /tmp/qc-static-nash-in-nash.json` | 0 |
| `python3 scripts/qc_repro.py industrial-organization --only nash-in-nash --timeout 300 --restore --out /tmp/qc-repro-nash-in-nash.json` | 0 |
| `python3 scripts/validate_catalog.py` | 0 |
| Python numerical spot-check of the tutorial formulas | 0 |
| Figure and thumbnail file/dimension check | 0 |
