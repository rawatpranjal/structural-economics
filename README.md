# Computational Economics

This repository gives graduate students and researchers short, executable examples in computational economics. Tutorials are organized by economic question and can be run from their folders with `python run.py`.

## Contents

- [Quick Start](#quick-start)
- [Dynamic Programming](#dynamic-programming)
- [Macroeconomics](#macroeconomics)
- [Industrial Organization](#industrial-organization)
- [Choice and Demand](#choice-and-demand)
- [Computational Game Theory](#computational-game-theory)
- [Time Series and Data](#time-series-and-data)
- [Computational Methods](#computational-methods)
- [Other Code Repositories](#other-code-repositories)

## Quick Start

```bash
pip install -r requirements.txt
cd dynamic-programming/cake-eating
python run.py
# -> generates README.md + figures/ + tables/
```

## Dynamic Programming

These tutorials start from one-state decision problems and build toward risk, search, asset pricing, business cycles, and general equilibrium.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="dynamic-programming/shock-discretization/figures/thumb.png" width="160">](dynamic-programming/shock-discretization/figures/stationary-mass.png) | **[Discretizing Persistent Shocks](dynamic-programming/shock-discretization/)** | Approximate persistent income or productivity shocks before putting them in a Bellman equation. Tauchen and Rouwenhorst turn an AR(1) into a finite Markov chain. |
| [<img src="dynamic-programming/cake-eating/figures/thumb.png" width="160">](dynamic-programming/cake-eating/figures/value-function.png) | **[Finite-Resource Cake Eating](dynamic-programming/cake-eating/)** | Solve the canonical finite-resource consumption problem. Value function iteration is checked against the closed-form Euler rule. |
| [<img src="dynamic-programming/optimal-growth/figures/thumb.png" width="160">](dynamic-programming/optimal-growth/figures/value-function.png) | **[Optimal Growth by Value Function Iteration](dynamic-programming/optimal-growth/)** | Solve the planner's one-capital growth problem by value function iteration. The log-utility case gives a closed-form benchmark for the computed policy. |
| [<img src="dynamic-programming/solow-growth/figures/thumb.png" width="160">](dynamic-programming/solow-growth/figures/solow-diagram.png) | **[Solow Growth and Conditional Convergence](dynamic-programming/solow-growth/)** | Iterate the Solow law of motion for capital per effective worker. Saving, depreciation, and technology growth determine convergence. |
| [<img src="dynamic-programming/consumption-savings/figures/thumb.png" width="160">](dynamic-programming/consumption-savings/figures/value-functions.png) | **[Income Risk and Buffer-Stock Saving](dynamic-programming/consumption-savings/)** | Solve household saving with persistent income risk and a borrowing limit. The policy shows how prudence and liquidity constraints create buffer-stock behavior. |
| [<img src="dynamic-programming/job-search-mccall/figures/thumb.png" width="160">](dynamic-programming/job-search-mccall/figures/accept-vs-reject.png) | **[McCall Job Search and the Reservation Wage](dynamic-programming/job-search-mccall/)** | Solve job search as a reservation-wage problem. The Bellman equation reduces to a scalar fixed point that makes optimal stopping easy to audit. |
| [<img src="dynamic-programming/asset-pricing/figures/thumb.png" width="160">](dynamic-programming/asset-pricing/figures/asset-price-function.png) | **[Lucas Tree Prices and the Stochastic Discount Factor](dynamic-programming/asset-pricing/)** | Price a Lucas tree from the household Euler equation. The stochastic discount factor links dividend risk and mean reversion to equilibrium prices. |
| [<img src="dynamic-programming/rbc/figures/thumb.png" width="160">](dynamic-programming/rbc/figures/comovements.png) | **[RBC Capital, Labor, and Business-Cycle Moments](dynamic-programming/rbc/)** | Solve a representative-household RBC model with endogenous labor on a global capital-TFP grid. Nonlinear policies show how productivity risk moves output, investment, and hours. |
| [<img src="dynamic-programming/diamond-mortensen-pissarides/figures/thumb.png" width="160">](dynamic-programming/diamond-mortensen-pissarides/figures/productivity-tightness.png) | **[DMP Search, Vacancies, and Unemployment](dynamic-programming/diamond-mortensen-pissarides/)** | Link productivity to vacancies and unemployment through match surplus and free entry. The tutorial compares a local rule with a global fixed point. |
| [<img src="dynamic-programming/aiyagari/figures/thumb.png" width="160">](dynamic-programming/aiyagari/figures/capital-market.png) | **[Aiyagari Saving and Capital-Market Clearing](dynamic-programming/aiyagari/)** | Clear the capital market in an incomplete-markets economy. Household saving policies aggregate into a capital supply curve and pin down the interest rate. |

## Macroeconomics

This section covers heterogeneous households, DSGE models, nonlinear global solutions, and continuous-time control.

### Heterogeneous Agents

These tutorials focus on incomplete-markets households and equilibrium interest rates.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="heterogeneous-agents/endogenous-grid-points/figures/thumb.png" width="160">](heterogeneous-agents/endogenous-grid-points/figures/consumption-policy.png) | **[Buffer-Stock Saving by Endogenous Grid Points](heterogeneous-agents/endogenous-grid-points/)** | Solve a buffer-stock saving problem without an inner asset-choice search. EGP works backward from next-period assets through the Euler equation, then simulates MPCs under income risk. |
| [<img src="heterogeneous-agents/envelope-equation-iteration/figures/thumb.png" width="160">](heterogeneous-agents/envelope-equation-iteration/figures/consumption-policy.png) | **[Envelope-Equation Iteration for Buffer-Stock Saving](heterogeneous-agents/envelope-equation-iteration/)** | Use the envelope condition as the update rule in the same buffer-stock environment. EEI iterates on marginal continuation values, then compares the policy with EGP and grid VFI. |
| [<img src="heterogeneous-agents/huggett-incomplete-markets/figures/thumb.png" width="160">](heterogeneous-agents/huggett-incomplete-markets/figures/bond-market.png) | **[Huggett Equilibrium and the Risk-Free Rate](heterogeneous-agents/huggett-incomplete-markets/)** | Clear a one-bond incomplete-markets economy with income risk and a borrowing limit. The HJB and stationary density equations deliver the interest rate that makes aggregate bond demand zero. |

### Linearized DSGE

These tutorials log-linearize DSGE models around steady state and solve the rational-expectations transition. They use coefficient matching or Klein-style QZ, the same first-order logic behind standard DSGE solvers.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="dsge/rbc/figures/thumb.png" width="160">](dsge/rbc/figures/irf-tfp-shock.png) | **[RBC TFP Shocks and Capital Propagation](dsge/rbc/)** | A productivity shock raises output today, while investment carries part of the response into future capital. The tutorial solves the local RBC transition with first-order perturbation and checks it against a nonlinear path. |
| [<img src="dsge/nkdsge/figures/thumb.png" width="160">](dsge/nkdsge/figures/irf-monetary-shock.png) | **[Sticky-Price Monetary Transmission in a New Keynesian DSGE](dsge/nkdsge/)** | Trace how policy-rate wedges and natural-rate demand shocks move output and inflation when prices are sticky. Coefficient matching solves the log-linear equilibrium, with Klein QZ checking determinacy. |
| [<img src="dsge/assetNews/figures/thumb.png" width="160">](dsge/assetNews/figures/irf-surprise-vs-news.png) | **[Lucas-Tree Dividend News and Asset Prices](dsge/assetNews/)** | Price a tree claim when investors learn about future dividends before cash flows arrive. A first-order pricing rule separates expected payoffs from stochastic discounting. |
| [<img src="dsge/rbc-with-labor/figures/thumb.png" width="160">](dsge/rbc-with-labor/figures/irf-tfp-shock.png) | **[RBC Labor Supply and TFP Shocks](dsge/rbc-with-labor/)** | Trace how endogenous hours, consumption, and investment transmit a productivity shock in an RBC model. Klein QZ solves the log-linear rational-expectations system and delivers the local impulse responses. |

### Global Nonlinear DSGE

These tutorials solve macro models on grids so constraints, taxes, and risk sharing remain visible.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="global-dsge/rbc-capital-tax/figures/thumb.png" width="160">](global-dsge/rbc-capital-tax/figures/steady-state-tax.png) | **[Capital Taxes and Saving in a Global RBC Model](global-dsge/rbc-capital-tax/)** | Study how a rebated capital-income tax lowers saving by taxing the return to future capital. A global grid solution with Euler refinement separates aggregate feasibility from the private after-tax return. |
| [<img src="global-dsge/rbc-irreversible-investment/figures/thumb.png" width="160">](global-dsge/rbc-irreversible-investment/figures/policy-functions.png) | **[Capital Overhang from Irreversible Investment in RBC](global-dsge/rbc-irreversible-investment/)** | Study a business-cycle economy where installed capital cannot be undone after a bad productivity draw. Global value function iteration tracks the kink at zero investment and the overhang in high-capital recession states. |
| [<img src="global-dsge/heaton-lucas/figures/thumb.png" width="160">](global-dsge/heaton-lucas/figures/equity-premium-and-distribution.png) | **[Heaton-Lucas Risk Sharing and Equity Premia](global-dsge/heaton-lucas/)** | Study why incomplete risk sharing makes wealth shares matter for equity premia. STPFI solves the implicit wealth-share transition and portfolio constraints in one global system. |

### Continuous-Time Macro and Optimal Control

These examples cover HJB equations, phase diagrams, shooting, and shadow prices.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="optimal-control/hjb-growth/figures/thumb.png" width="160">](optimal-control/hjb-growth/figures/value-function.png) | **[Ramsey Capital Accumulation by HJB Upwinding](optimal-control/hjb-growth/)** | Study the planner's continuous-time consumption-investment tradeoff. An upwind HJB computes the shadow value of capital and the path back to the Ramsey steady state. |
| [<img src="optimal-control/phase-diagrams/figures/thumb.png" width="160">](optimal-control/phase-diagrams/figures/phase-diagram.png) | **[Ramsey Consumption Choice and Saddle Paths](optimal-control/phase-diagrams/)** | Study the planner's initial consumption choice in continuous-time growth. A phase diagram and stable-arm integration show why only one path satisfies the long-run boundary condition. |
| [<img src="optimal-control/ramsey-growth/figures/thumb.png" width="160">](optimal-control/ramsey-growth/figures/phase-diagram.png) | **[Ramsey Saving by Saddle-Path Shooting](optimal-control/ramsey-growth/)** | Study how a Ramsey planner chooses initial consumption when capital is inherited. Shooting adjusts the jump variable until the transition reaches the stable path. |
| [<img src="optimal-control/continuous-cake-eating/figures/thumb.png" width="160">](optimal-control/continuous-cake-eating/figures/consumption-path.png) | **[Fixed-Resource Consumption and Pontryagin Shadow Prices](optimal-control/continuous-cake-eating/)** | Study how a planner spreads a fixed resource over an infinite horizon. Pontryagin's costate equation prices the remaining stock and gives the smooth depletion path. |

## Industrial Organization

The IO section covers firm boundaries, vertical relationships, demand, pricing, production, mergers, collusion, bargaining, and industry dynamics.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="industrial-organization/theory-of-the-firm/figures/thumb.png" width="160">](industrial-organization/theory-of-the-firm/figures/investment-incentives.png) | **[Firm Boundaries, Hold-Up, and Vertical Integration](industrial-organization/theory-of-the-firm/)** | Study when ownership should move inside the firm for relationship-specific investment. A grid comparison weighs incentive gains against contracting and hierarchy costs. |
| [<img src="industrial-organization/vertical-relationships/figures/thumb.png" width="160">](industrial-organization/vertical-relationships/figures/price-quantity.png) | **[Double Marginalization in Vertical Supply Chains](industrial-organization/vertical-relationships/)** | Compare a manufacturer-retailer channel with the integrated benchmark. Backward induction shows how linear wholesale pricing raises perceived marginal cost, while a two-part tariff restores quantity through a fixed fee. |
| [<img src="industrial-organization/vertical-contracts/figures/thumb.png" width="160">](industrial-organization/vertical-contracts/figures/assortment-selection.png) | **[Vending Assortments Under Vertical Contracts](industrial-organization/vertical-contracts/)** | Study how rebates and slotting fees affect product availability in a vending channel. Exact enumeration compares the retailer's preferred assortment under each contract. |
| [<img src="industrial-organization/bertrand-logit-demand/figures/thumb.png" width="160">](industrial-organization/bertrand-logit-demand/figures/price-comparison.png) | **[Differentiated-Products Merger Pricing with Logit Demand](industrial-organization/bertrand-logit-demand/)** | Study how common ownership changes markups in a four-product market. A logit-demand Bertrand FOC solve turns diversion and cost efficiencies into counterfactual prices. |
| [<img src="industrial-organization/logit-supply-side/figures/thumb.png" width="160">](industrial-organization/logit-supply-side/figures/estimation-comparison.png) | **[Cereal Demand and Markup Recovery from Prices](industrial-organization/logit-supply-side/)** | Recover markups and marginal costs in a differentiated-products market with endogenous prices. Berry inversion, IV/2SLS, and Bertrand-Nash pricing FOCs connect shares, ownership, and prices to costs. |
| [<img src="industrial-organization/blp-random-coefficients/figures/thumb.png" width="160">](industrial-organization/blp-random-coefficients/figures/observed-vs-predicted-shares.png) | **[Differentiated-Products Demand with BLP](industrial-organization/blp-random-coefficients/)** | Study how consumer heterogeneity changes substitution in differentiated-products markets. A BLP contraction and IV/GMM estimate taste dispersion before comparing elasticities with plain logit. |
| [<img src="industrial-organization/production-functions-markups/figures/thumb.png" width="160">](industrial-organization/production-functions-markups/figures/production-estimates.png) | **[Production Elasticities and Firm Markups](industrial-organization/production-functions-markups/)** | Recover firm markups from production elasticities and variable-input shares. A proxy-control regression corrects the productivity bias created when firms choose inputs after seeing their efficiency. |
| [<img src="industrial-organization/effective-hhi/figures/thumb.png" width="160">](industrial-organization/effective-hhi/figures/hhi-vs-nfirms.png) | **[Market Concentration Screens with HHI](industrial-organization/effective-hhi/)** | Measure market concentration for merger screening from firm ownership shares. HHI and effective firm counts give the screen, while a small Bertrand pricing exercise separates concentration from price effects. |
| [<img src="industrial-organization/collusion-detection/figures/thumb.png" width="160">](industrial-organization/collusion-detection/figures/profits-by-regime.png) | **[Repeated-Game Cartels and Price Screens](industrial-organization/collusion-detection/)** | Study when firms can keep a cartel together in a repeated Cournot market. Closed-form incentive constraints set the sustainability threshold, and a simulated price path shows what a screen can and cannot say. |
| [<img src="industrial-organization/dynamic-games/figures/thumb.png" width="160">](industrial-organization/dynamic-games/figures/investment-policy.png) | **[Dynamic Games and Markov-Perfect Investment](industrial-organization/dynamic-games/)** | Model firms that invest in quality while watching rivals' states. Future rivalry makes today's investment payoff depend on the whole industry state. |
| [<img src="industrial-organization/dynamic-games-estimation/figures/thumb.png" width="160">](industrial-organization/dynamic-games-estimation/figures/ccp-heatmaps.png) | **[Dynamic Games Estimation from First-Stage CCPs](industrial-organization/dynamic-games-estimation/)** | Estimate a small investment game from first-stage policies and forward values. The likelihood avoids solving a full MPE at every trial parameter. |
| [<img src="industrial-organization/dynamic-entry-exit/figures/thumb.png" width="160">](industrial-organization/dynamic-entry-exit/figures/value-function.png) | **[Dynamic Entry and Exit in Oligopoly](industrial-organization/dynamic-entry-exit/)** | Separate entry costs from exit decisions in a dynamic market. A stable firm count can still hide turnover and option-value hysteresis. |
| [<img src="industrial-organization/dynamic-discrete-choice/figures/thumb.png" width="160">](industrial-organization/dynamic-discrete-choice/figures/value-and-ccp.png) | **[Bus Engine Replacement and Dynamic Choice](industrial-organization/dynamic-discrete-choice/)** | Estimate a replacement decision by NFXP, CCP, and MPEC. The same hazard can be handled by nested fixed points or Bellman constraints. |
| [<img src="industrial-organization/three-part-tariffs/figures/thumb.png" width="160">](industrial-organization/three-part-tariffs/figures/usage-policy.png) | **[Three-Part Tariffs and Forward-Looking Broadband Demand](industrial-organization/three-part-tariffs/)** | Study plan choice when data allowances make usage forward-looking. The allowance has value before the cap is actually reached. |
| [<img src="industrial-organization/nash-in-nash/figures/thumb.png" width="160">](industrial-organization/nash-in-nash/figures/negotiated-prices.png) | **[Nash-in-Nash Hospital-Insurer Bargaining](industrial-organization/nash-in-nash/)** | Solve bilateral bargaining over hospital-insurer transfers. Network outside options matter because losing a system is not the same as losing one hospital. |
| [<img src="industrial-organization/merger-simulation/figures/thumb.png" width="160">](industrial-organization/merger-simulation/figures/price-comparison.png) | **[Differentiated-Products Merger Simulation](industrial-organization/merger-simulation/)** | Simulate post-merger pricing with differentiated products. Screens such as GUPPI help, but the solved equilibrium depends on substitution and efficiencies. |

## Choice and Demand

Choice and demand focuses on revealed preference, learning, and choice models.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="choice/revealed-preference-afriat/figures/thumb.png" width="160">](choice/revealed-preference-afriat/figures/budget-lines-consistent.png) | **[Afriat's Revealed-Preference Test](choice/revealed-preference-afriat/)** | Test finite choice data without assuming a demand curve. The tutorial shows how revealed-preference cycles become a rationalizability failure. |
| [<img src="choice/preference-recoverability/figures/thumb.png" width="160">](choice/preference-recoverability/figures/budget-lines.png) | **[Preference Bounds from Revealed Choices](choice/preference-recoverability/)** | Use observed choices to bound the utilities that could explain them. Finite data restrict preferences without identifying one exact demand system. |
| [<img src="choice/money-pump-index/figures/thumb.png" width="160">](choice/money-pump-index/figures/money-pump-cycle.png) | **[Money Pump Index for Revealed Preference](choice/money-pump-index/)** | Put an economic size on revealed-preference violations. The same pass-fail result can hide very different expenditure losses. |
| [<img src="choice/houtman-maks-rational-subsets/figures/thumb.png" width="160">](choice/houtman-maks-rational-subsets/figures/conflict-graph.png) | **[Houtman-Maks Rational Cores](choice/houtman-maks-rational-subsets/)** | Find the largest subset of choices that can be rationalized. This helps distinguish a few inconsistent observations from broad inconsistency. |
| [<img src="choice/revealed-price-preference/figures/thumb.png" width="160">](choice/revealed-price-preference/figures/price-cost-ratios.png) | **[Revealed Price Preference](choice/revealed-price-preference/)** | Ask whether price regimes can be ranked from the bundles they make affordable. Bundle choices can look rational even when price-regime rankings fail. |
| [<img src="choice/logit-discrete-choice/figures/thumb.png" width="160">](choice/logit-discrete-choice/figures/log-likelihood-surface.png) | **[Plain Logit Demand and IIA](choice/logit-discrete-choice/)** | Estimate a baseline product-choice model and inspect its substitution restriction. Lost buyers from one product get reallocated mechanically across the rest. |
| [<img src="choice/maximum-score-binary-choice/figures/thumb.png" width="160">](choice/maximum-score-binary-choice/figures/score-objectives.png) | **[Maximum Score Binary Choice](choice/maximum-score-binary-choice/)** | Estimate a binary-choice index with a nonsmooth classification criterion. Scale normalization and smoothing make the semiparametric target visible. |
| [<img src="choice/bayesian-learning/figures/thumb.png" width="160">](choice/bayesian-learning/figures/belief-evolution.png) | **[Bayesian Learning and Sequential Investment](choice/bayesian-learning/)** | Track how signals update beliefs and change the decision to invest or wait. Waiting is valuable when beliefs are still uncertain enough to move. |
| [<img src="choice/urn-behavioral-mixtures/figures/thumb.png" width="160">](choice/urn-behavioral-mixtures/figures/bayes-likelihood-ratio.png) | **[Bayesian Urn Classification and Behavioral Mixtures](choice/urn-behavioral-mixtures/)** | Use likelihood ratios to classify urn states and finite mixtures to recover latent decision rules from repeated choices. |
| [<img src="choice/risk-aversion-monotone-choice/figures/thumb.png" width="160">](choice/risk-aversion-monotone-choice/figures/risky-choice-fits.png) | **[Risk Aversion and Monotone Stochastic Choice](choice/risk-aversion-monotone-choice/)** | Estimate lottery choice with unconstrained, structural, and monotonicity-constrained logits. Shape restrictions discipline noisy row shares. |
| [<img src="choice/nested-logit/figures/thumb.png" width="160">](choice/nested-logit/figures/elasticity-heatmap.png) | **[Nested Logit Demand and Within-Nest Substitution](choice/nested-logit/)** | Group products into nests so substitution can be stronger among close alternatives. The nesting parameter changes where lost buyers go. |

## Computational Game Theory

These tutorials introduce computational methods to solve game theoretic equilibria.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="game-theory/normal-form-games/figures/thumb.png" width="160">](game-theory/normal-form-games/figures/pure-deviation-gains.png) | **[Normal-Form Games and Nash Equilibrium Checks](game-theory/normal-form-games/)** | Read finite games as no-profitable-deviation conditions. Pure and mixed equilibria are checked directly against payoff tables. |
| [<img src="game-theory/static-games/figures/thumb.png" width="160">](game-theory/static-games/figures/cournot-best-response.png) | **[Cournot Oligopoly and Best-Response Dynamics](game-theory/static-games/)** | Solve a quantity-setting game and watch best responses converge. Equilibrium is where each firm is happy with its quantity given the others. |
| [<img src="game-theory/first-price-auctions/figures/thumb.png" width="160">](game-theory/first-price-auctions/figures/bid-functions.png) | **[First-Price Auctions and Bid Shading](game-theory/first-price-auctions/)** | Show how bidders trade off a higher win probability against a lower surplus if they win. More competition reduces bid shading. |
| [<img src="game-theory/quantal-response-equilibrium/figures/thumb.png" width="160">](game-theory/quantal-response-equilibrium/figures/qre-path.png) | **[Entry Game QRE and Noisy Best Responses](game-theory/quantal-response-equilibrium/)** | Let players make payoff-sensitive mistakes in a small entry game. Finite precision smooths behavior while keeping a fixed-point discipline. |

## Time Series and Data

These tutorials cover stochastic processes, macroeconomic data, and forecasting.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="time-series/fred-macro-data/figures/thumb.png" width="160">](time-series/fred-macro-data/figures/time-series.png) | **[FRED-Style Macro Data and Business-Cycle Moments](time-series/fred-macro-data/)** | Turn GDP, inflation, unemployment, and policy-rate series into business-cycle moments. The tutorial is about the data objects behind macro targets. |
| [<img src="time-series/ar-processes/figures/thumb.png" width="160">](time-series/ar-processes/figures/ar1-irfs.png) | **[Persistent Shocks and Multiplier-Accelerator Dynamics](time-series/ar-processes/)** | Study how persistence changes shock half-lives, spectra, and income dynamics. The coefficient is an economic timing assumption as well as a statistical parameter. |
| [<img src="time-series/stock-watson/figures/thumb.png" width="160">](time-series/stock-watson/figures/factor-comparison.png) | **[Stock-Watson Diffusion Index Forecasts](time-series/stock-watson/)** | Extract a common factor from a large macro panel and use it for forecasting. The example compares factor forecasts with simpler benchmarks. |

## Computational Methods

These tutorials are standalone references for optimization, approximation, simulation, filtering, and sampling.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="computational-methods/numerical-optimization/figures/thumb.png" width="160">](computational-methods/numerical-optimization/figures/optimizer-paths.png) | **[Multimodal Likelihood Optimization](computational-methods/numerical-optimization/)** | Optimize an objective with more than one plausible mode. Starting values matter because different optimizers can settle in different basins. |
| [<img src="computational-methods/simulation-based-estimation/figures/thumb.png" width="160">](computational-methods/simulation-based-estimation/figures/criterion-surfaces.png) | **[Simulation-Based Estimation: MSM and Indirect Inference](computational-methods/simulation-based-estimation/)** | Estimate one stochastic search DGP by moment matching and auxiliary-model matching. Criterion surfaces show simulation noise and identification. |
| [<img src="computational-methods/projection-methods/figures/thumb.png" width="160">](computational-methods/projection-methods/figures/chebyshev-basis.png) | **[Chebyshev Projection for a Growth Policy](computational-methods/projection-methods/)** | Approximate a smooth growth policy with a small set of coefficients. Euler errors away from the fitted nodes show whether the approximation is reliable. |
| [<img src="computational-methods/perturbation-linearization/figures/thumb.png" width="160">](computational-methods/perturbation-linearization/figures/local-approximations.png) | **[Perturbation Around a Steady State](computational-methods/perturbation-linearization/)** | Linearize nonlinear dynamics near a steady state. Higher-order terms show where curvature and asymmetric responses start to matter. |
| [<img src="computational-methods/metropolis-hastings/figures/thumb.png" width="160">](computational-methods/metropolis-hastings/figures/mh-walk.png) | **[Metropolis-Hastings for a Bimodal Posterior](computational-methods/metropolis-hastings/)** | Diagnose how a finite MCMC run explores a two-mode posterior. Proposal tuning changes acceptance, mode switching, and effective sample size. |
| [<img src="computational-methods/kalman-filter/figures/thumb.png" width="160">](computational-methods/kalman-filter/figures/simulated-signal.png) | **[Kalman Filtering a Latent Economic State](computational-methods/kalman-filter/)** | Filter a latent economic state from noisy signals. Prediction errors update the state estimate, uncertainty, and likelihood together. |
| [<img src="computational-methods/particle-filter/figures/thumb.png" width="160">](computational-methods/particle-filter/figures/filter-comparison.png) | **[Particle Filtering Latent Economic States](computational-methods/particle-filter/)** | Approximate latent-state filtering with weighted simulated particles. Proposal choices drive weight collapse and Monte Carlo error. |

## Other Code Repositories

- [QuantEcon](https://github.com/QuantEcon) - Open-source lectures and libraries for quantitative economics.
- [John Stachurski GitHub](https://github.com/jstac) - Repositories on computational economics, stochastic dynamics, and econometric theory.
- [Dynamic Structural Econometrics (DSE 2023)](https://github.com/dseconf/DSE2023) - Summer-school materials on solving and estimating dynamic models.
- [New York Fed DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl) - Julia tools for solving and estimating DSGE models.
- [GDSGE](https://github.com/gdsge/gdsge) - Toolbox for global nonlinear DSGE solution.
- [DSGE_mod](https://github.com/JohannesPfeifer/DSGE_mod) - Dynare `.mod` files for canonical DSGE models.
- [Chris Conlon Grad IO](https://github.com/chrisconlon/Grad-IO) - PhD empirical IO course materials.
- [Kenneth Train Software](https://eml.berkeley.edu/~train/software.html) - Mixed logit and discrete-choice estimation code.
- [Victor Aguirregabiria Computer Code](https://sites.google.com/view/victoraguirregabiriaswebsite/computer-code?authuser=0) - Dynamic discrete-choice and dynamic-game estimation code.
- [EconRL](https://github.com/SimonHashtag/EconRL) - Bibliography of economics and finance papers using reinforcement learning.
- [CompEcon](https://github.com/fediskhakov/CompEcon) - Foundations of Computational Economics course materials.
- [CompEcon Fall 2017](https://github.com/ScPo-CompEcon/CoursePack) - Sciences Po computational economics course pack.
- [EmpiricalIO](https://github.com/kohei-kawaguchi/EmpiricalIO) - Empirical IO lecture notes, assignments, and R code.
- [Quantitative Macro Models](https://github.com/hessjacob/Quantitative-Macro-Models) - Python implementations of heterogeneous-agent and firm-dynamics models.
- [Benjamin Moll Codes](https://benjaminmoll.com/codes/) - Finite-difference codes for continuous-time heterogeneous-agent models.
- [Archive of Empirical Dynamic Programming Research](https://github.com/CForg/Archive-of-Empirical-Dynamic-Programming-Research) - Replication materials for empirical dynamic programming.
- [Dynamical Systems](https://github.com/JuliaDynamics/DynamicalSystems.jl) - Julia library for nonlinear dynamics and time-series analysis.
- [Heterogeneous Agents Bayes](https://github.com/BASEforHANK) - Julia toolboxes for Bayesian HANK solution and estimation.
- [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) - Interactive book on Kalman and Bayesian filters.
- [OpenSourceEcon CompMethods](https://github.com/OpenSourceEcon/CompMethods) - Executable Jupyter Book on computational methods for economists.
