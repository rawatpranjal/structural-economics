# bullshit-detector — fred-macro-data — 2026-05-20

**Bullshit score: 15%** — One DILUTED/LOW finding: the Equations section reuses the Greek letter sigma for two numerically distinct quantities (DGP scaling vector vs HP-cycle standard deviation); prose warns but the equations themselves are silent. No FALSE, UNIMPLEMENTED, or result-changing gaps found.

## Header

- Claim sources: `time-series/fred-macro-data/README.md` (Overview, Equations, Model Setup, Results, Takeaway)
- Code / artifact root: `time-series/fred-macro-data/run.py`
- Data artifacts: `time-series/fred-macro-data/tables/business-cycle-stats.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, 2026-05-20)
- Date: 2026-05-20
- Diagram-only cap applied: no

---

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | sigma denotes two distinct quantities in same Equations section | DILUTED | LOW | no |
| 2 | DGP formula s_t = rho*s_{t-1} + sqrt(1-rho^2)*eps_t, eps~N(0,C) | HOLDS | - | - |
| 3 | HP filter solves (I + lambda K'K) tau = y | HOLDS | - | - |
| 4 | K is correct (T-2) x T second-difference matrix | HOLDS | - | - |
| 5 | T=200, T_B=5000, lambda=1600 match code | HOLDS | - | - |
| 6 | Series means, stds, persistence, corr matrix match code | HOLDS | - | - |
| 7 | All table numbers match tables/business-cycle-stats.csv | HOLDS | - | - |
| 8 | Okun regression direction: c_u on c_g | HOLDS | - | - |
| 9 | Okun slope -0.142 consistent with table moments | HOLDS | - | - |
| 10 | Unemployment most persistent cycle (autocorr 0.649) | HOLDS | - | - |
| 11 | Sign claims (countercyclical unemp, procyclical CPI/Fed) | HOLDS | - | - |
| 12 | Innovation correlation matrix correctly labelled (not cycle corr) | HOLDS | - | - |
| 13 | Benchmark uses same process for sampling variation | HOLDS | - | - |

---

## Findings

### Finding 1: sigma overload — DGP scaling and HP-cycle std dev share the same symbol in Equations

- **Claim source (verbatim):** "The vector $s_t$ is a standardized latent state and $\sigma$ is the 4-vector of series standard deviations (3.0, 1.5, 1.5, 3.0); note that $\sigma_j = \mathrm{sd}(c_{j,t})$ below is the HP-cycle standard deviation, a numerically distinct quantity." — `README.md:31`

- **Code evidence (verbatim):**
  ```python
  stds = np.array([3.0, 1.5, 1.5, 3.0])
  # ...
  data = means[None, :] + stds[None, :] * standardized   # sigma used as DGP scaling
  # ...
  vols = cycles_df.std()                                  # cycle std dev, != stds above
  ```
  `run.py:69`, `run.py:93`, `run.py:120`

- **Data evidence:** `tables/business-cycle-stats.csv` row 2: `GDP growth,2.896,...` — GDP-growth cycle volatility is 2.896, not 3.0 (the DGP sigma). The two quantities are numerically different for all series (e.g. GDP 2.896 vs 3.0, Unemployment 0.975 vs 1.5).

- **Category:** DILUTED — the Equations section uses $\sigma$ in the DGP block (`y_t = \mu + \sigma \odot s_t`, where sigma = [3.0,1.5,1.5,3.0]) and then again in the moments block (`\sigma_j = \mathrm{sd}(c_{j,t})`, where sigma_j is the HP-cycle std, e.g. 2.896 for GDP). The prose at README:31 warns the reader that these are "numerically distinct quantities," but the equations themselves carry no distinguishing notation (e.g. no $\sigma^y$ vs $\sigma^c$), so a reader parsing only the equation block has no signal.

- **Severity:** LOW — the warning is present in the surrounding prose; no result is misreported; the confusion is notational only.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "numerically distinct" in open("time-series/fred-macro-data/README.md").read()  # PASSES on current code (warning exists but equations share symbol); FAILS if equations distinguished sigma^y from sigma^c
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert re.search(r'\\sigma\^[{]?[yc]', open("time-series/fred-macro-data/run.py").read()) is not None  # PASSES if DGP sigma gets a distinguishing superscript; FAILS on current code
  ```

---

### Findings 2-13: All HOLDS

The following claims were grounded against code and data and found faithful. Verbatim evidence listed for each.

**Finding 2 — DGP formula.**
- Claim: `README.md:24-29` — `s_t = rho odot s_{t-1} + sqrt(1-rho^2) odot eps_t`, `eps_t ~ N(0,C)`, `y_t = mu + sigma odot s_t`.
- Code: `run.py:85-93` — `innovation_scale = np.sqrt(1.0 - persistence**2)`, `standardized[t] = persistence * standardized[t-1] + innovation_scale * innovations[t]`, `data = means[None,:] + stds[None,:] * standardized`. Matches element-wise.

**Finding 3 — HP filter linear system.**
- Claim: `README.md:41-43` — `(I + lambda K'K) tau_j = y_j`.
- Code: `run.py:55` — `trend = spsolve(I + lamb * K.T @ K, y)`. Exact match.

**Finding 4 — K matrix construction.**
- Claim: `README.md:40-43` — second-difference penalty `sum [(tau_{t+1}-tau_t)-(tau_t-tau_{t-1})]^2` over T-2 terms.
- Code: `run.py:51-54` — `K = spdiags(np.array([e, -2.0*e, e]), np.array([0,1,2]), T-2, T)`. Verified by constructing K for T=6: produces correct second-difference rows `[1,-2,1]` starting at column i for row i.

**Finding 5 — Scalar parameters.**
- Claim: `README.md:67-70` — T=200, T_B=5000, lambda=1600.
- Code: `run.py:177-179` — `T=200`, `T_benchmark=5000`, `lamb_hp=1600.0`. Exact match. T=200 quarters = 50 years (200/4=50): HOLDS.

**Finding 6 — Series primitives.**
- Claim: `README.md:74-88` — means [2.5, 2.0, 5.5, 4.0], stds [3.0, 1.5, 1.5, 3.0], persistence [0.30, 0.70, 0.85, 0.80], innovation corr matrix as stated.
- Code: `run.py:68-80` — `means=np.array([2.5,2.0,5.5,4.0])`, `stds=np.array([3.0,1.5,1.5,3.0])`, `persistence=np.array([0.30,0.70,0.85,0.80])`, `corr=np.array([[1.0,0.2,-0.6,0.3],[0.2,1.0,-0.3,0.5],[-0.6,-0.3,1.0,-0.2],[0.3,0.5,-0.2,1.0]])`. All entries match.

**Finding 7 — Table numbers.**
- Claim: `README.md:138-143` — moment table with specific values.
- Data: `tables/business-cycle-stats.csv:1-5` — all six column values for all four series match the README table exactly (e.g. GDP Vol=2.896, Unemp Corr=-0.423, Unemp Autocorr=0.649).

**Finding 8 — Okun regression direction.**
- Claim: `README.md:55-60` — `c_{u,t} = alpha_O + beta_O c_{g,t} + e_t` (unemployment on GDP).
- Code: `run.py:138-141` — `x = cycles_df["GDP_growth"]`, `y = cycles_df["Unemployment"]`, `slope, intercept = np.polyfit(x, y, 1)`. x=GDP, y=unemployment: direction matches.

**Finding 9 — Okun slope -0.142.**
- Claim: `README.md:149` — "The Okun slope is -0.142."
- Verification: implied slope = corr * sd_u / sd_g = -0.423 * 0.975 / 2.896 = -0.142 (from CSV moments). Consistent. Exact polyfit value needs re-run to verify but is consistent with grounded moments.

**Finding 10 — Unemployment most persistent.**
- Claim: `README.md:149` — "Unemployment is the most persistent cycle."
- Data: `tables/business-cycle-stats.csv:4` — Unemployment Autocorr=0.649 > Fed funds=0.599 > CPI=0.483 > GDP=0.282. HOLDS.

**Finding 11 — Sign claims.**
- Claim: `README.md:131-132` — "Unemployment is countercyclical. Inflation and the policy rate are procyclical."
- Data: `tables/business-cycle-stats.csv` — Corr(Unemp,GDP)=-0.423 (negative=countercyclical), Corr(CPI,GDP)=+0.174, Corr(Fed,GDP)=+0.187 (both positive=procyclical). HOLDS.

**Finding 12 — Innovation correlation matrix labelling.**
- Claim: `README.md:81` — heading "Innovation correlation matrix C" (not cycle correlation).
- Code: `run.py:73-80` — variable named `corr`, passed to `rng.multivariate_normal(np.zeros(4), corr, size=T)`. Correctly characterised as innovation-level; cycle correlations differ due to persistence attenuation and HP filtering.

**Finding 13 — Benchmark sampling variation claim.**
- Claim: `README.md:134` — "Benchmark columns use the same process, so they show sampling variation rather than validation with real data."
- Code: `run.py:191-198` — `benchmark_df = generate_synthetic_macro_data(T=T_benchmark, seed=2026, dated=False)` — same function, same parameters, different seed and length. HOLDS.

---

## Cross-cutting patterns

- No cross-cutting issues found. The single DILUTED finding (sigma notation overload) is isolated to the Equations section and does not propagate to code, tables, or Takeaway.
- The tutorial consistently uses the same random seed (42 for the 50-year sample, 2026 for the benchmark), which means all committed numbers in README and the CSV are reproducible from the committed code.
- The distinction between innovation correlation (C, the DGP parameter) and cycle correlation (the reported moments table) is correctly maintained throughout code and prose. The only gap is in the LaTeX notation of the Equations section.

---

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Below the 50% halt threshold. One DILUTED/LOW finding. Safe to continue work.

1. **Finding 1 (DILUTED — sigma overload):** Turn the violated invariant into a passing test:
   ```python
   # test_sigma_notation.py
   import re, pathlib
   src = pathlib.Path("time-series/fred-macro-data/run.py").read_text()
   # Current code: only one sigma symbol in equations; no superscript distinction
   assert re.search(r'\\sigma\^[{]?[yc]', src) is None  # PASSES on current code
   ```
   Then write the honest-fix pass condition test:
   ```python
   assert re.search(r'\\sigma\^[{]?[yc]', src) is not None  # FAILS on current code
   ```

2. The fix is cosmetic: distinguish the DGP sigma from the moments sigma in the equations block (e.g. `\sigma^{y}` for the DGP scaling vector and `\sigma^{c}` for the cycle standard deviation), and update the prose warning at README:31 to reference the new notation. No code changes required.

3. After the notation fix, re-run `python scripts/validate_catalog.py` and re-run this skill. The expected new bullshit score is 0-10%.
