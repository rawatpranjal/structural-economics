# Structural Economics: Computational Examples

A library of executable models in computational and structural economics. Every model is self-contained, runs with `python run.py`, and produces a documented report with equations, solutions, visualizations, and economic takeaways.

**Built with:** Python, JAX, NumPy, SciPy, Matplotlib | **License:** MIT

## Quick Start

```bash
pip install -r requirements.txt
cd dynamic-programming/cake-eating
python run.py
# -> generates README.md + figures/ + tables/
```

---

## Dynamic Programming

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="dynamic-programming/cake-eating/figures/thumb.png" width="120"> | [**Cake Eating**](dynamic-programming/cake-eating/) | Value Function Iteration | Optimal depletion: consume fraction $(1-\beta)$ each period |
| <img src="dynamic-programming/optimal-growth/figures/thumb.png" width="120"> | [**Optimal Growth**](dynamic-programming/optimal-growth/) | VFI | Capital converges to steady state $k_{ss} = (\alpha\beta A)^{1/(1-\alpha)}$ |
| <img src="dynamic-programming/solow-growth/figures/thumb.png" width="120"> | [**Solow Growth**](dynamic-programming/solow-growth/) | Simulation | Exogenous savings drives long-run output per capita |
| <img src="dynamic-programming/consumption-savings/figures/thumb.png" width="120"> | [**Consumption-Savings**](dynamic-programming/consumption-savings/) | VFI + Markov income | Precautionary savings under income uncertainty |
| <img src="dynamic-programming/job-search-mccall/figures/thumb.png" width="120"> | [**Job Search (McCall)**](dynamic-programming/job-search-mccall/) | VFI | Reservation wage: be more selective when patient |
| <img src="dynamic-programming/diamond-mortensen-pissarides/figures/thumb.png" width="120"> | [**DMP Search & Matching**](dynamic-programming/diamond-mortensen-pissarides/) | Log-linearization | Beveridge curve and the Shimer puzzle |
| <img src="dynamic-programming/asset-pricing/figures/thumb.png" width="120"> | [**Lucas Asset Pricing**](dynamic-programming/asset-pricing/) | Functional VFI | Price-dividend ratio depends on risk aversion |
| <img src="dynamic-programming/rbc/figures/thumb.png" width="120"> | [**Real Business Cycles**](dynamic-programming/rbc/) | VFI + simulation | Investment is 4x more volatile than output |
| <img src="dynamic-programming/aiyagari/figures/thumb.png" width="120"> | [**Aiyagari (1994)**](dynamic-programming/aiyagari/) | VFI + GE bisection | Precautionary savings push $r < 1/\beta - 1$ |
| <img src="dynamic-programming/ode-methods/figures/thumb.png" width="120"> | [**ODE Methods**](dynamic-programming/ode-methods/) | Phase diagrams | Solow, Ramsey saddle paths, Lotka-Volterra cycles |

## Heterogeneous Agents

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="heterogeneous-agents/vfi-deterministic/figures/thumb.png" width="120"> | [**VFI Deterministic**](heterogeneous-agents/vfi-deterministic/) | VFI | With $\beta R < 1$, agent runs down assets to zero |
| <img src="heterogeneous-agents/vfi-iid-income/figures/thumb.png" width="120"> | [**VFI with IID Income**](heterogeneous-agents/vfi-iid-income/) | VFI | Income risk generates precautionary savings buffer |
| <img src="heterogeneous-agents/endogenous-grid-points/figures/thumb.png" width="120"> | [**Endogenous Grid Points**](heterogeneous-agents/endogenous-grid-points/) | EGP (Carroll 2006) | 10x faster than VFI by inverting the Euler equation |
| <img src="heterogeneous-agents/egp-aiyagari/figures/thumb.png" width="120"> | [**EGP-Aiyagari**](heterogeneous-agents/egp-aiyagari/) | EGP + GE bisection | EGP makes GE loop feasible; $r < 1/\beta - 1$ in equilibrium |
| <img src="heterogeneous-agents/envelope-equation-iteration/figures/thumb.png" width="120"> | [**Envelope Equation**](heterogeneous-agents/envelope-equation-iteration/) | EEI | Third approach: iterate on $V'(a)$ via envelope theorem |

## Industrial Organization

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="industrial-organization/bertrand-logit-demand/figures/thumb.png" width="120"> | [**Bertrand-Nash Logit**](industrial-organization/bertrand-logit-demand/) | Fixed point | Mergers raise prices; cost efficiencies can offset |
| <img src="industrial-organization/blp-random-coefficients/figures/thumb.png" width="120"> | [**BLP Random Coefficients**](industrial-organization/blp-random-coefficients/) | Contraction mapping | Heterogeneous preferences relax IIA |
| <img src="industrial-organization/effective-hhi/figures/thumb.png" width="120"> | [**Effective HHI**](industrial-organization/effective-hhi/) | Index computation | $\Delta \text{HHI} = 2 s_1 s_2 \times 10000$ for merger screening |
| <img src="industrial-organization/static-games/figures/thumb.png" width="120"> | [**Static Games**](industrial-organization/static-games/) | Nash equilibrium | Cournot output between monopoly and competition |
| <img src="industrial-organization/collusion-detection/figures/thumb.png" width="120"> | [**Collusion Detection**](industrial-organization/collusion-detection/) | Repeated games | $\delta^* = 9/17$ for duopoly; more firms = harder to collude |
| <img src="industrial-organization/dynamic-entry-exit/figures/thumb.png" width="120"> | [**Dynamic Entry/Exit**](industrial-organization/dynamic-entry-exit/) | Dynamic programming | Entry costs sustain above-competitive profits |
| <img src="industrial-organization/nash-in-nash/figures/thumb.png" width="120"> | [**Nash-in-Nash**](industrial-organization/nash-in-nash/) | Bilateral bargaining | Incremental value = leverage in vertical markets |

## Choice Models

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="choice/revealed-preference-afriat/figures/thumb.png" width="120"> | [**Revealed Preference**](choice/revealed-preference-afriat/) | Afriat + Warshall | GARP: testable implication of utility maximization |
| <img src="choice/garp-warshall/figures/thumb.png" width="120"> | [**GARP & Warshall**](choice/garp-warshall/) | Transitive closure | Bronars power increases with observations |
| <img src="choice/preference-recoverability/figures/thumb.png" width="120"> | [**Preference Recoverability**](choice/preference-recoverability/) | Afriat numbers | Bound indifference curves without functional form |
| <img src="choice/logit-discrete-choice/figures/thumb.png" width="120"> | [**Logit Discrete Choice**](choice/logit-discrete-choice/) | MLE | IIA property: cross-elasticities proportional to shares |
| <img src="choice/bayesian-learning/figures/thumb.png" width="120"> | [**Bayesian Learning**](choice/bayesian-learning/) | Bayes' rule | Optimal classifier; ML converges to Bayes |

## Optimal Control

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="optimal-control/phase-diagrams/figures/thumb.png" width="120"> | [**Phase Diagrams**](optimal-control/phase-diagrams/) | Linearization | Ramsey steady state is a saddle point |
| <img src="optimal-control/finite-difference/figures/thumb.png" width="120"> | [**Finite Differences**](optimal-control/finite-difference/) | Upwind FD | Implicit scheme for HJB equations in continuous time |
| <img src="optimal-control/continuous-cake-eating/figures/thumb.png" width="120"> | [**Continuous Cake Eating**](optimal-control/continuous-cake-eating/) | Pontryagin | Shadow price rises at rate $\rho$ in current value |
| <img src="optimal-control/ramsey-growth/figures/thumb.png" width="120"> | [**Ramsey Growth**](optimal-control/ramsey-growth/) | Shooting method | Saddle path found by bisection on initial consumption |

## Continuous Time

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="continuous-time/huggett-incomplete-markets/figures/thumb.png" width="120"> | [**Huggett (1993)**](continuous-time/huggett-incomplete-markets/) | HJB + KFE | Precautionary savings push $r < \rho$ in equilibrium |
| <img src="continuous-time/hjb-growth/figures/thumb.png" width="120"> | [**HJB Growth**](continuous-time/hjb-growth/) | Upwind FD | Continuous-time growth via finite differences |

## Time Series

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="time-series/fred-macro-data/figures/thumb.png" width="120"> | [**FRED Macro Data**](time-series/fred-macro-data/) | HP filter | Okun's law, Phillips curve in cyclical components |
| <img src="time-series/stock-watson/figures/thumb.png" width="120"> | [**Stock-Watson Factors**](time-series/stock-watson/) | PCA | One factor explains 57% of variance; 28% forecast gain |

---

## Structure

Each model folder is self-contained:

```
topic-area/
  model-name/
    run.py          # Single script: model + solve + generate report
    README.md       # Auto-generated: equations, solutions, figures, takeaways
    figures/        # PNG plots (value functions, policies, simulations)
    tables/         # CSV data (comparison tables, statistics)
```

Shared utilities live in `lib/`:

```
lib/
  grids.py          # Asset/state grid construction
  discretize.py     # Tauchen, Rouwenhorst for AR(1) processes
  interpolate.py    # Linear interpolation (JAX-compatible)
  vfi.py            # Generic VFI solver
  plotting.py       # Consistent matplotlib academic style
  output.py         # ModelReport class (auto-generates README.md)
```

Original files are preserved in `_legacy/`.

## License

MIT License. Copyright (c) 2021 Pranjal Rawat.
