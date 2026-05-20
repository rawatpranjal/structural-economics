# bullshit-detector — theory-of-the-firm — recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. Prior residual DILUTED/LOW finding (Model Setup table "marginal investment return" phrasing) is resolved: `run.py:156` now reads "Share of revenue $\\theta x$ captured by the investor under governance $g$". All numeric claims re-verified by independent computation.

## Header
- Claim sources: `industrial-organization/theory-of-the-firm/README.md` (126 lines, read in full)
- Code / artifact root: `industrial-organization/theory-of-the-firm/run.py` (292 lines, read in full)
- Data artifacts: `industrial-organization/theory-of-the-firm/tables/governance-comparison.csv` (13 lines, read in full)
- Seed audit: `bullshit-detector_theory-of-the-firm_recheck_2026-05-20.md` (prior recheck, score 10%)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, second recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "of revenue θx" in Equations and Model Setup table | HOLDS | — | — |
| 2 | FOC b_g(s)θ - x = 0 | HOLDS | — | — |
| 3 | x_g(s) = b_g(s)θ | HOLDS | — | — |
| 4 | W_g(s) = θx_g - 0.5x_g² - F_g(s) | HOLDS | — | — |
| 5 | W* = 0.5θ² | HOLDS | — | — |
| 6 | All six incentive-schedule coefficients | HOLDS | — | — |
| 7 | All six governance-cost coefficients | HOLDS | — | — |
| 8 | All 12 table rows (inc, inv, surplus, eff, chosen) | HOLDS | — | — |
| 9 | Thresholds s≈0.21 and s≈0.37 | HOLDS | — | — |
| 10 | Model Setup b_g(s) gloss names revenue, not marginal value | HOLDS | — | — |

## Findings

### Finding 1 (prior residual): Model Setup table phrasing — RESOLVED

- **Prior claim (buggy):** "Share of the marginal investment return captured by the investor under governance $g$" — prior `README.md:55` / `run.py:156`
- **Current code evidence (verbatim):**
  ```python
  "| $b_g(s)$ | Share of revenue $\\theta x$ captured by the investor under governance $g$ |\n"
  ```
  `run.py:156`
- **Current README (verbatim):** "| $b_g(s)$ | Share of revenue $\theta x$ captured by the investor under governance $g$ |" — `README.md:55`
- The phrase "marginal investment return" is absent from `run.py`. The phrase "Share of revenue $\\theta x$" matches the Equations section at `README.md:22` and the code `investment = theta * incentive` at `run.py:39`. **RESOLVED.**
- **Category:** HOLDS

---

### Finding 2: FOC b_g(s)θ - x = 0 — HOLDS

- **Claim source (verbatim):** "The private first-order condition is $$b_g(s)\theta - x = 0,$$" — `README.md:23-24`
- **Code evidence:** `run.py:39`: `investment = theta * incentive`. With `incentive = np.clip(b_g(s), 0.05, 1.0)` from `run.py:38`: `x = theta * b_g(s)` ↔ FOC `b_g*theta - x = 0`. ✓
- **Category:** HOLDS

---

### Finding 3: Investment formula x_g(s) = b_g(s)θ — HOLDS

- **Claim source (verbatim):** "which gives $$x_g(s) = b_g(s)\theta$$" — `README.md:26`
- **Code evidence:** `run.py:39`: `investment = theta * incentive`. ✓
- **Category:** HOLDS

---

### Finding 4: Surplus formula W_g(s) — HOLDS

- **Claim source (verbatim):** "$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$" — `README.md:29`
- **Code evidence:** `run.py:40`: `surplus = theta * investment - 0.5 * investment**2 - values["governance_cost"]`. Exact match. ✓
- **Category:** HOLDS

---

### Finding 5: First-best surplus W* = 0.5θ² — HOLDS

- **Claim source (verbatim):** "$$W^{\ast}=\frac{1}{2}\theta^2$$" — `README.md:45`
- **Code evidence:** `run.py:36`: `first_best_surplus = 0.5 * theta**2`. With θ=4: `0.5 * 16 = 8.0`. Independent recomputation: 8.0. ✓
- **Category:** HOLDS

---

### Finding 6: Incentive schedule coefficients — HOLDS

- **Claim source (verbatim):** "$b_{\text{spot}}(s)=0.72-0.55s,\quad b_{\text{contract}}(s)=0.72-0.25s,\quad b_{\text{integration}}(s)=0.74-0.03s.$" — `README.md:32-34`
- **Code evidence:** `run.py:22-32` — all six coefficients (intercepts 0.72, 0.72, 0.74; slopes -0.55, -0.25, -0.03) match exactly. Clip at `run.py:38` (`np.clip(..., 0.05, 1.0)`) never activates on s∈[0,1] with these schedules (minimum raw value `b_spot(1) = 0.17 > 0.05`). ✓
- **Category:** HOLDS

---

### Finding 7: Governance cost coefficients — HOLDS

- **Claim source (verbatim):** "$F_{\text{spot}}(s)=0.02+0.04s,\quad F_{\text{contract}}(s)=0.38+0.03s,\quad F_{\text{integration}}(s)=1.05-0.35s.$" — `README.md:37-39`
- **Code evidence:** `run.py:23-32` — all six coefficients (intercepts 0.02, 0.38, 1.05; slopes 0.04, 0.03, -0.35) match exactly. ✓
- **Category:** HOLDS

---

### Finding 8: All 12 table rows — HOLDS

- **Claim source:** `README.md:103-115` and `tables/governance-comparison.csv:1-13`
- **Verification:** Independent recomputation of all 12 rows for s ∈ {0.0, 0.3, 0.5, 1.0} and three regimes. Every incentive share, investment, surplus, efficiency ratio, and chosen-regime flag matches to displayed precision. Representative spot-check: `s=0.3, Long-term contract: inc=0.65, inv=2.58, surp=6.60, eff=82.5%, chosen=yes` — computed W=6.6028, rounds to 6.60 at 2dp; chosen correctly because 6.60 > 6.48 (integration) > 6.38 (spot). ✓
- **Category:** HOLDS

---

### Finding 9: Threshold claims s≈0.21 and s≈0.37 — HOLDS

- **Claim source (verbatim):** "spot exchange wins for $s\lesssim 0.21$. Long-term contracts win for $0.21\lesssim s\lesssim 0.37$. Vertical integration wins for $s\gtrsim 0.37$." — `README.md:82`
- **Verification:** On the 10001-point fine grid, Spot→Long-term transition at s=0.2059 (→ 0.21 at 2dp); Long-term→Integration transition at s=0.3742 (→ 0.37 at 2dp). Thresholds embedded dynamically via f-string at `run.py:184-188` from `regions[0][1]` and `regions[2][0]`. Cannot drift from code output. ✓
- **Category:** HOLDS

---

### Finding 10: Model Setup b_g(s) gloss — HOLDS

- **Current code evidence (verbatim):**
  ```python
  "| $b_g(s)$ | Share of revenue $\\theta x$ captured by the investor under governance $g$ |\n"
  ```
  `run.py:156`
- The phrase "marginal investment return" is absent from `run.py` (grep confirms). "Share of revenue $\\theta x$" is consistent with the Equations section at `run.py:122` ("capture share $b_g(s)$ of revenue $\\theta x$") and the code's revenue-sharing computation at `run.py:39`. ✓
- **Category:** HOLDS

---

## Cross-cutting patterns

- The residual finding from the prior recheck (DILUTED/LOW: Model Setup table using "marginal investment return") is fully resolved. The fix reached both the Equations string (`run.py:122`) and the Model Setup table string (`run.py:156`). Both now say "of revenue $\\theta x$".
- All numeric claims remain clean. The dynamic threshold insertion at `run.py:184-188` prevents drift between code output and README for threshold values.
- No FALSE, no DILUTED, no DATA DRIFT, no UNIMPLEMENTED findings in this pass.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings HOLDS. No action required.
1. The test `test_finding1_recheck_model_setup_table_not_marginal` in `tests/test_theory-of-the-firm.py` passes: `"marginal investment return" not in src` and `r"Share of revenue $\\theta x$ captured by the investor" in src`. This is the honest-fix condition and it passes.
2. The violated-invariant test `test_finding1_violated_invariant_code_uses_revenue_sharing` passes: `r"b_g(s)\theta - x = 0" in src` and no marginal-value FOC form present.
3. No further fixes needed.
