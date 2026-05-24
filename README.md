# Computational Economics

This repository gives graduate students and researchers short, executable models from computational and structural economics. Each example is motivated by a by distinct economic question or computational method. Each can be run from its folders with `python run.py`.

This repo was sourced from ideas and contributions of PhD colleagues [Shahzoor Safdar](https://github.com/shahzoor), [Simon Lebastard](https://github.com/slebastard), [Max Blesch](https://github.com/MaxBlesch), [Enoch H. Kang](https://sites.google.com/view/hyunwookkang), [Hoang Nguyen](https://github.com/huuhoang2211), [Kathryn Nicholson](https://sites.google.com/gwmail.gwu.edu/kathrynnicholson/home), and [Weipeng Zhang](https://gufaculty360.georgetown.edu/s/contact/0033600001WDdwXAAT/weipeng-zhang). It also draws from the coursework from professors [John Rust](https://editorialexpress.com/jrust/), [Nathan Miller](http://www.nathanhmiller.org/), [Harry Paarsch](https://sites.google.com/site/hjpaarsch/), [Sanjog Misra](https://sanjogmisra.com/), [Toshihiko Mukoyama](https://sites.google.com/view/toshimukoyama/home), [Mark Huggett](https://sites.google.com/georgetown.edu/mark-huggett/home), [Dan Cao](https://dan-cao.facultysite.georgetown.edu/), and [Benjamin Moll](https://benjaminmoll.com/). At the bottom you will find a wider list of resources. 

Happy programming and building! =]

## Contents

- [Quick Start](#quick-start)
- [Numerical Methods](#numerical-methods)
- [Dynamic Programming](#dynamic-programming)
- [Macroeconomics](#macroeconomics)
- [Industrial Organization](#industrial-organization)
- [Structural Econometrics](#structural-econometrics)
- [Bayesian Methods](#bayesian-methods)
- [Choice and Demand](#choice-and-demand)
- [Computational Game Theory](#computational-game-theory)
- [Time Series](#time-series)
- [Agent-Based Models](#agent-based-models)
- [Selected External Resources](#selected-external-resources)

## Quick Start

```bash
pip install -r requirements.txt
cd dynamic-programming/cake-eating
python run.py
# -> generates README.md + figures/ + tables/
```

## Numerical Methods

Here we cover common tools the rest of the repo uses. This covers solving `f(x) = 0`, finding extrema of `f(x)`, and approximating a function.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="numerical-methods/root-finding/figures/thumb.png" width="160">](numerical-methods/root-finding/figures/trajectories.png) | **[Scalar Root Finding for Equilibrium Rates](numerical-methods/root-finding/)** | Three classic methods for finding the root of a function, compared on a stylized bond market. |
| [<img src="numerical-methods/interpolation/figures/thumb.png" width="160">](numerical-methods/interpolation/figures/target-vs-fit.png) | **[Off-Grid Function Approximation by Interpolation](numerical-methods/interpolation/)** | Three ways to approximate a function from its values at a finite set of points, tested on smooth and kinked targets. |
| [<img src="numerical-methods/quadrature/figures/thumb.png" width="160">](numerical-methods/quadrature/figures/error-vs-nodes.png) | **[Numerical Quadrature: Gauss-Hermite Nodes for Conditional Expectations](numerical-methods/quadrature/)** | Computing expectations under a Gaussian density using a handful of optimally placed evaluation points. |
| [<img src="numerical-methods/neural-networks-regression/figures/thumb.png" width="160">](numerical-methods/neural-networks-regression/figures/fitted-surface.png) | **[Feedforward Neural Networks for Regression and Density Approximation](numerical-methods/neural-networks-regression/)** | A small neural network learns a noisy production surface, introducing the four building blocks reused across the catalog's deep-learning tutorials. |
| [<img src="numerical-methods/scalar-optimization-monopoly-pricing/figures/thumb.png" width="160">](numerical-methods/scalar-optimization-monopoly-pricing/figures/profit-curve.png) | **[Scalar Optimization for Monopoly Pricing](numerical-methods/scalar-optimization-monopoly-pricing/)** | Three ways to find the profit-maximizing price for a monopolist, benchmarked against the closed-form solution. |
| [<img src="numerical-methods/constrained-optimization-kkt/figures/thumb.png" width="160">](numerical-methods/constrained-optimization-kkt/figures/simplex-paths.png) | **[Constrained Optimization and KKT Conditions](numerical-methods/constrained-optimization-kkt/)** | How a planner splits a fixed budget across three projects when one is too weak to fund, and what the shadow prices reveal. |
| [<img src="numerical-methods/simulated-likelihood/figures/thumb.png" width="160">](numerical-methods/simulated-likelihood/figures/bias-variance-vs-R.png) | **[Simulated Maximum Likelihood, Common Random Numbers, and Halton Sequences](numerical-methods/simulated-likelihood/)** | Estimating a model with no closed-form likelihood by averaging over fixed Halton draws that keep the objective smooth. |
| [<img src="numerical-methods/fixed-point-acceleration/figures/thumb.png" width="160">](numerical-methods/fixed-point-acceleration/figures/share-fit.png) | **[Fixed-Point Iteration and Acceleration](numerical-methods/fixed-point-acceleration/)** | Three ways to solve a fixed-point equation, compared on the demand-inversion problem behind modern industrial organization. |
| [<img src="numerical-methods/global-search-multistart/figures/thumb.png" width="160">](numerical-methods/global-search-multistart/figures/profit-surface.png) | **[Global Search and Multi-Start Diagnostics](numerical-methods/global-search-multistart/)** | Why a single optimizer run can quietly miss the true global maximum, and three diagnostics that bound how badly it missed. |

## Dynamic Programming

These tutorials start from one-state decision problems and build toward risk, search, asset pricing, business cycles, and general equilibrium.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="dynamic-programming/shock-discretization/figures/thumb.png" width="160">](dynamic-programming/shock-discretization/figures/stationary-mass.png) | **[Discretizing Persistent Shocks](dynamic-programming/shock-discretization/)** | Two methods for replacing a persistent income or productivity process with a finite-state Markov chain that a dynamic program can use. |
| [<img src="dynamic-programming/cake-eating/figures/thumb.png" width="160">](dynamic-programming/cake-eating/figures/value-function.png) | **[Finite-Resource Cake Eating](dynamic-programming/cake-eating/)** | Three classic methods for solving the cake-eating problem, benchmarked against the closed-form consumption rule. |
| [<img src="dynamic-programming/optimal-growth/figures/thumb.png" width="160">](dynamic-programming/optimal-growth/figures/value-function.png) | **[Optimal Growth by Value Function Iteration](dynamic-programming/optimal-growth/)** | How a planner splits output between consumption today and capital tomorrow, recovering the textbook saving rule on a grid. |
| [<img src="computational-methods/projection-methods/figures/thumb.png" width="160">](computational-methods/projection-methods/figures/chebyshev-basis.png) | **[Growth-Model Capital Policy by Chebyshev Projection](computational-methods/projection-methods/)** | Approximating a planner's saving rule with a handful of smooth polynomial coefficients instead of a dense grid of values. |
| [<img src="computational-methods/smolyak-sparse-grids/figures/thumb.png" width="160">](computational-methods/smolyak-sparse-grids/figures/grid-size-scaling.png) | **[High-Dimensional Growth Policy by Smolyak Sparse Grids](computational-methods/smolyak-sparse-grids/)** | Beating the curse of dimensionality on a multi-sector growth problem by keeping only the grid points that carry new information. |
| [<img src="dynamic-programming/q-learning-growth/figures/thumb.png" width="160">](dynamic-programming/q-learning-growth/figures/policy-comparison.png) | **[Stochastic Optimal Growth by Q-Learning](dynamic-programming/q-learning-growth/)** | Learning the saving rule in a stochastic growth economy from sampled experience alone, without knowing how productivity moves. |
| [<img src="dynamic-programming/solow-growth/figures/thumb.png" width="160">](dynamic-programming/solow-growth/figures/solow-diagram.png) | **[Solow Growth and Conditional Convergence](dynamic-programming/solow-growth/)** | Why higher saving rates raise the level of long-run income but not its growth rate, and why poor countries can converge to rich ones. |
| [<img src="dynamic-programming/consumption-savings/figures/thumb.png" width="160">](dynamic-programming/consumption-savings/figures/value-functions.png) | **[Buffer-Stock Saving with Persistent Income by VFI](dynamic-programming/consumption-savings/)** | How households facing persistent income shocks and no borrowing build a saving buffer and spend out of new income near the limit. |
| [<img src="dynamic-programming/job-search-mccall/figures/thumb.png" width="160">](dynamic-programming/job-search-mccall/figures/accept-vs-reject.png) | **[McCall Job Search and the Reservation Wage](dynamic-programming/job-search-mccall/)** | The wage cutoff an unemployed worker will accept, and how patience and unemployment benefits push it higher. |
| [<img src="dynamic-programming/asset-pricing/figures/thumb.png" width="160">](dynamic-programming/asset-pricing/figures/asset-price-function.png) | **[Lucas Tree I: SDF Baseline by Scaled-Price Iteration](dynamic-programming/asset-pricing/)** | Pricing a Lucas tree that pays a random dividend, and how risk aversion changes the price-to-dividend ratio over the business cycle. |
| [<img src="dynamic-programming/rbc/figures/thumb.png" width="160">](dynamic-programming/rbc/figures/comovements.png) | **[RBC Capital, Labor, and Business-Cycle Moments](dynamic-programming/rbc/)** | How productivity shocks generate the textbook business-cycle pattern of smooth consumption, volatile investment, and procyclical hours. |
| [<img src="dynamic-programming/diamond-mortensen-pissarides/figures/thumb.png" width="160">](dynamic-programming/diamond-mortensen-pissarides/figures/productivity-tightness.png) | **[DMP Search, Vacancies, and Unemployment](dynamic-programming/diamond-mortensen-pissarides/)** | Why search-and-matching models struggle to generate the observed swings in vacancies and unemployment unless match surplus is small. |
| [<img src="dynamic-programming/aiyagari/figures/thumb.png" width="160">](dynamic-programming/aiyagari/figures/capital-market.png) | **[Aiyagari Saving and Capital-Market Clearing](dynamic-programming/aiyagari/)** | How uninsured income risk and a no-borrowing rule push the equilibrium interest rate below the impatience benchmark. |

## Macroeconomics

This section covers heterogeneous households, DSGE models, nonlinear global solutions, and continuous-time control.

### Heterogeneous Agents

These tutorials focus on incomplete-markets households and equilibrium interest rates.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="heterogeneous-agents/kolmogorov-forward-equation/figures/thumb.png" width="160">](heterogeneous-agents/kolmogorov-forward-equation/figures/stationary-density-ou.png) | **[Kolmogorov Forward Equation and the Stationary Wealth Distribution](heterogeneous-agents/kolmogorov-forward-equation/)** | The long-run cross section of wealth in an incomplete-markets economy, recovered by tracking how mass flows across asset levels. |
| [<img src="heterogeneous-agents/endogenous-grid-points/figures/thumb.png" width="160">](heterogeneous-agents/endogenous-grid-points/figures/consumption-policy.png) | **[Buffer-Stock Saving with IID Income by EGP](heterogeneous-agents/endogenous-grid-points/)** | How a household with risky income saves as a buffer, computed by inverting the consumption-saving rule instead of grid search. |
| [<img src="heterogeneous-agents/envelope-equation-iteration/figures/thumb.png" width="160">](heterogeneous-agents/envelope-equation-iteration/figures/consumption-policy.png) | **[Buffer-Stock Saving with Persistent Income by Envelope-Equation Iteration](heterogeneous-agents/envelope-equation-iteration/)** | How a household saves when income shocks persist, computed by updating the marginal value of wealth without solving the full value function. |
| [<img src="heterogeneous-agents/huggett-incomplete-markets/figures/thumb.png" width="160">](heterogeneous-agents/huggett-incomplete-markets/figures/bond-market.png) | **[Huggett Equilibrium and the Risk-Free Rate](heterogeneous-agents/huggett-incomplete-markets/)** | Why the risk-free rate sits below the rate of time preference when households face uninsurable income risk and trade one bond in zero net supply. |
| [<img src="heterogeneous-agents/aiyagari-hact/figures/thumb.png" width="160">](heterogeneous-agents/aiyagari-hact/figures/capital-market.png) | **[Continuous-time Aiyagari and the Mean-Field Game](heterogeneous-agents/aiyagari-hact/)** | Aiyagari's capital-market equilibrium recast as a game where each household reacts to an interest rate that the whole population's saving collectively determines. |
| [<img src="heterogeneous-agents/huggett-aggregate-risk-srl/figures/thumb.png" width="160">](heterogeneous-agents/huggett-aggregate-risk-srl/figures/policy-consumption.png) | **[Structural Reinforcement Learning for Huggett with Aggregate Risk](heterogeneous-agents/huggett-aggregate-risk-srl/)** | Solving a Huggett economy with both individual and economy-wide income risk by training a saving rule that clears the bond market without tracking the full wealth distribution. |
| [<img src="heterogeneous-agents/sequence-space-jacobian-hank/figures/thumb.png" width="160">](heterogeneous-agents/sequence-space-jacobian-hank/figures/irf-comparison.png) | **[Sequence-Space Jacobian for One-Asset HANK](heterogeneous-agents/sequence-space-jacobian-hank/)** | How an interest-rate cut ripples through an economy of households who differ in wealth, by linearizing each model block around steady state and stacking them into a single matrix equation. |

### Linearized DSGE

These tutorials log-linearize DSGE models around steady state and solve the rational-expectations transition. They use coefficient matching or Klein-style QZ, the same first-order logic behind standard DSGE solvers.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="dsge/blanchard-kahn-determinacy/figures/thumb.png" width="160">](dsge/blanchard-kahn-determinacy/figures/bk-heatmap.png) | **[Blanchard-Kahn Conditions and Saddle-Path Selection in Linear Rational Expectations](dsge/blanchard-kahn-determinacy/)** | Counting eigenvalues to decide when a linearized macro model has one stable equilibrium, no stable equilibrium, or room for sunspot equilibria driven by pure noise. |
| [<img src="dsge/rbc/figures/thumb.png" width="160">](dsge/rbc/figures/irf-fixed-labor.png) | **[Linearized RBC by Perturbation and QZ (with and without endogenous labor)](dsge/rbc/)** | How a productivity shock moves output, investment, and hours in a real business cycle model, with cases for fixed and flexible labor. |
| [<img src="dsge/nkdsge/figures/thumb.png" width="160">](dsge/nkdsge/figures/irf-monetary-shock.png) | **[Sticky-Price Monetary Transmission in a New Keynesian DSGE](dsge/nkdsge/)** | How a central bank rate hike lowers output and inflation in an economy where firms cannot freely reprice. |
| [<img src="dsge/behavioral-nk/figures/thumb.png" width="160">](dsge/behavioral-nk/figures/forward-guidance-attenuation.png) | **[Cognitive Discounting in a Behavioral New Keynesian Model](dsge/behavioral-nk/)** | How forward guidance about future rates loses bite when households and firms put less weight on news about the distant future. |
| [<img src="dsge/assetNews/figures/thumb.png" width="160">](dsge/assetNews/figures/irf-surprise-vs-news.png) | **[Lucas Tree II: Dividend News by Linearized Pricing](dsge/assetNews/)** | How an asset's price moves today when investors learn news about future dividends before the cash flows arrive. |
| [<img src="computational-methods/perturbation-linearization/figures/thumb.png" width="160">](computational-methods/perturbation-linearization/figures/local-approximations.png) | **[Aggregate Adjustment Around a Steady State](computational-methods/perturbation-linearization/)** | Approximating the path back to steady state with a local Taylor expansion when the true nonlinear transition is too costly to solve exactly. |

### Global Nonlinear DSGE

These tutorials solve macro models on grids so constraints, taxes, and risk sharing remain visible.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="global-dsge/rbc-capital-tax/figures/thumb.png" width="160">](global-dsge/rbc-capital-tax/figures/steady-state-tax.png) | **[Capital Taxes and Saving in a Global RBC Model](global-dsge/rbc-capital-tax/)** | How a capital-income tax depresses long-run saving in a stochastic growth economy, with the policy solved on a full productivity grid. |
| [<img src="global-dsge/rbc-irreversible-investment/figures/thumb.png" width="160">](global-dsge/rbc-irreversible-investment/figures/policy-functions.png) | **[Capital Overhang from Irreversible Investment in RBC](global-dsge/rbc-irreversible-investment/)** | Why a bad productivity draw leaves the economy stuck with too much capital when investment cannot be reversed. |
| [<img src="global-dsge/heaton-lucas/figures/thumb.png" width="160">](global-dsge/heaton-lucas/figures/equity-premium-and-distribution.png) | **[Heaton-Lucas Risk Sharing and Equity Premia](global-dsge/heaton-lucas/)** | Why two households who cannot fully insure each other against shocks end up demanding a risk premium on stocks that varies with their wealth share. |
| [<img src="global-dsge/deep-learning-optimal-growth/figures/thumb.png" width="160">](global-dsge/deep-learning-optimal-growth/figures/policy-comparison.png) | **[Deep Learning for Optimal Growth](global-dsge/deep-learning-optimal-growth/)** | Benchmarking a neural-network saving rule against the textbook closed-form solution of an optimal growth model. |

### Continuous-Time Macro and Optimal Control

These examples cover HJB equations, phase diagrams, shooting, and shadow prices.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="optimal-control/upwind-finite-differences/figures/thumb.png" width="160">](optimal-control/upwind-finite-differences/figures/value-and-drift.png) | **[Upwind Finite Differences for First-Order PDEs and the HJB State Constraint](optimal-control/upwind-finite-differences/)** | How a planner on a bounded capital interval picks consumption when the value function's slope changes sign and the lower boundary may bind. |
| [<img src="optimal-control/hjb-growth/figures/thumb.png" width="160">](optimal-control/hjb-growth/figures/value-function.png) | **[Ramsey Capital Accumulation by HJB Upwinding](optimal-control/hjb-growth/)** | Ramsey capital accumulation solved by turning the continuous Bellman equation into a sparse linear system that delivers consumption at every capital level. |
| [<img src="optimal-control/phase-diagrams/figures/thumb.png" width="160">](optimal-control/phase-diagrams/figures/phase-diagram.png) | **[Ramsey Consumption Choice and Saddle Paths](optimal-control/phase-diagrams/)** | The Ramsey economy's unique stable consumption path traced from the saddle steady state outward through phase-plane analysis. |
| [<img src="optimal-control/ramsey-growth/figures/thumb.png" width="160">](optimal-control/ramsey-growth/figures/phase-diagram.png) | **[Ramsey Saving by Saddle-Path Shooting](optimal-control/ramsey-growth/)** | Picking the right initial consumption jump in a Ramsey economy by guessing, simulating, and adjusting until the path lands on the long-run steady state. |

## Industrial Organization

The IO section covers firm boundaries, vertical relationships, demand, pricing, production, mergers, collusion, bargaining, and industry dynamics.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="industrial-organization/theory-of-the-firm/figures/thumb.png" width="160">](industrial-organization/theory-of-the-firm/figures/investment-incentives.png) | **[Firm Boundaries, Hold-Up, and Vertical Integration](industrial-organization/theory-of-the-firm/)** | How ownership choice shapes whether suppliers invest in assets that only one buyer can use. |
| [<img src="industrial-organization/vertical-relationships/figures/thumb.png" width="160">](industrial-organization/vertical-relationships/figures/price-quantity.png) | **[Double Marginalization in Vertical Supply Chains](industrial-organization/vertical-relationships/)** | Why an independent retailer adds a markup on top of the manufacturer's markup, and what contract restores joint profits. |
| [<img src="industrial-organization/vertical-contracts/figures/thumb.png" width="160">](industrial-organization/vertical-contracts/figures/assortment-selection.png) | **[Vending Assortments Under Vertical Contracts](industrial-organization/vertical-contracts/)** | How manufacturers use rebates and slotting fees to win scarce shelf space in a vending machine. |
| [<img src="industrial-organization/bertrand-ownership-matrix/figures/thumb.png" width="160">](industrial-organization/bertrand-ownership-matrix/figures/prices-pre-post.png) | **[Multi-Product Bertrand-Nash Pricing and the Ownership Matrix](industrial-organization/bertrand-ownership-matrix/)** | How multi-product firms set prices once they internalize sales they steal from their own brands, and how a merger changes those prices. |
| [<img src="industrial-organization/logit-supply-side/figures/thumb.png" width="160">](industrial-organization/logit-supply-side/figures/estimation-comparison.png) | **[Cereal Demand and Markup Recovery from Prices](industrial-organization/logit-supply-side/)** | How to recover the hidden marginal costs and markups of cereal brands when only prices and market shares are observed. |
| [<img src="industrial-organization/blp-random-coefficients/figures/thumb.png" width="160">](industrial-organization/blp-random-coefficients/figures/observed-vs-predicted-shares.png) | **[Differentiated-Products Demand with BLP](industrial-organization/blp-random-coefficients/)** | How buyers with different tastes substitute between competing products when one of them raises its price. |
| [<img src="industrial-organization/production-functions-markups/figures/thumb.png" width="160">](industrial-organization/production-functions-markups/figures/production-estimates.png) | **[Production Elasticities and Firm Markups](industrial-organization/production-functions-markups/)** | How to back out firm-level markups from production data when accountants never report marginal cost. |
| [<img src="industrial-organization/dynamic-games/figures/thumb.png" width="160">](industrial-organization/dynamic-games/figures/investment-policy.png) | **[Quality-Ladder Dynamic Game: Solving the MPE](industrial-organization/dynamic-games/)** | Two firms invest to climb a quality ladder while watching each other, and how their equilibrium investment rule is computed. |
| [<img src="industrial-organization/dynamic-games-estimation/figures/thumb.png" width="160">](industrial-organization/dynamic-games-estimation/figures/ccp-heatmaps.png) | **[Quality-Ladder Dynamic Game: Estimating with CCPs](industrial-organization/dynamic-games-estimation/)** | How to recover the costs and rewards behind firms' investment behavior on a quality ladder directly from observed choices. |
| [<img src="industrial-organization/dynamic-entry-exit/figures/thumb.png" width="160">](industrial-organization/dynamic-entry-exit/figures/value-function.png) | **[Entry, Exit, and Market Structure in Oligopoly](industrial-organization/dynamic-entry-exit/)** | Why sunk entry costs lock in the number of firms in a market, and what the long-run firm count looks like. |
| [<img src="industrial-organization/nash-in-nash/figures/thumb.png" width="160">](industrial-organization/nash-in-nash/figures/negotiated-prices.png) | **[Hospital-Insurer Network Bargaining](industrial-organization/nash-in-nash/)** | How hospital systems extract higher payments from insurers by threatening to drop the whole network at once. |
| [<img src="industrial-organization/merger-simulation/figures/thumb.png" width="160">](industrial-organization/merger-simulation/figures/hhi-vs-nfirms.png) | **[Merger Pricing: Concentration Screens, Diversion, and Bertrand-Nash Equilibrium](industrial-organization/merger-simulation/)** | Whether a merger between close rivals raises prices, and how three different screens for harm agree or disagree. |
| [<img src="industrial-organization/online-pricing-partial-identification/figures/thumb.png" width="160">](industrial-organization/online-pricing-partial-identification/figures/active-prices.png) | **[Online Pricing with Revealed-Preference Bounds](industrial-organization/online-pricing-partial-identification/)** | How a seller posting prices online can rule out bad prices fast by using the fact that lower prices always sell at least as well. |

## Structural Econometrics

Structural econometrics focuses on estimating economic primitives from observed choices, transitions, and policies. These tutorials connect likelihoods, dynamic programs, revealed decisions, and reward recovery.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="structural-econometrics/gmm-foundations/figures/thumb.png" width="160">](structural-econometrics/gmm-foundations/figures/efficiency-gain.png) | **[GMM Foundations: Moment Conditions, Identification, and Optimal Weighting](structural-econometrics/gmm-foundations/)** | How moment conditions translate a structural model into a parameter estimate, and how two-step weighting makes the optimal choice feasible. |
| [<img src="industrial-organization/dynamic-discrete-choice/figures/thumb.png" width="160">](industrial-organization/dynamic-discrete-choice/figures/value-and-ccp.png) | **[Bus Engine Replacement: NFXP, CCP, MPEC, and the MCE-IRL Equivalence](industrial-organization/dynamic-discrete-choice/)** | When should a bus manager replace an aging engine, and four equivalent ways to estimate that rule from observed replacement choices. |
| [<img src="structural-econometrics/keane-wolpin-career-choice/figures/thumb.png" width="160">](structural-econometrics/keane-wolpin-career-choice/figures/choice-shares.png) | **[Keane-Wolpin Career Choice by Emax Approximation](structural-econometrics/keane-wolpin-career-choice/)** | When does a young worker leave school for blue-collar or white-collar work, solved by backward induction with a fitted continuation-value surface to keep the state space manageable. |
| [<img src="structural-econometrics/dcegm-retirement-saving/figures/thumb.png" width="160">](structural-econometrics/dcegm-retirement-saving/figures/branch-consumption.png) | **[Retirement and Saving by Discrete-Continuous EGM](structural-econometrics/dcegm-retirement-saving/)** | When does an older household retire and how much should it save, solved by separating the continuous saving problem on each branch and stitching them through an upper envelope. |
| [<img src="structural-econometrics/auction-valuation-recovery/figures/thumb.png" width="160">](structural-econometrics/auction-valuation-recovery/figures/recovered-cdf.png) | **[Recovering Auction Values from First-Price Bids](structural-econometrics/auction-valuation-recovery/)** | How to recover hidden private values from observed sealed bids by inverting the equilibrium bidding rule. |
| [<img src="structural-econometrics/q-learning-bus-engine/figures/thumb.png" width="160">](structural-econometrics/q-learning-bus-engine/figures/replacement-hazard.png) | **[Rust Bus Replacement by Soft Q-Learning and DQN](structural-econometrics/q-learning-bus-engine/)** | The same bus replacement rule recovered two ways: tabular soft Q-learning without a transition matrix, and a neural value function trained on sampled transitions. |
| [<img src="choice/maximum-score-binary-choice/figures/thumb.png" width="160">](choice/maximum-score-binary-choice/figures/score-objectives.png) | **[Binary Participation with Maximum Score](choice/maximum-score-binary-choice/)** | How to estimate a binary participation rule when the error distribution is unknown, by searching for the index that classifies the most choices correctly. |
| [<img src="choice/bayesian-learning/figures/thumb.png" width="160">](choice/bayesian-learning/figures/belief-evolution.png) | **[Sequential Investment Under Bayesian Learning](choice/bayesian-learning/)** | When should a firm invest in a project of unknown quality, given that each new signal sharpens beliefs and waiting is itself a costly option. |
| [<img src="computational-methods/numerical-optimization/figures/thumb.png" width="160">](computational-methods/numerical-optimization/figures/optimizer-paths.png) | **[Latent-Regime Likelihoods and Optimizer Basins](computational-methods/numerical-optimization/)** | How a likelihood with two equally good parameter regions can fool a single optimizer, and why restart grids and global search are needed to detect the second basin. |
| [<img src="computational-methods/simulation-based-estimation/figures/thumb.png" width="160">](computational-methods/simulation-based-estimation/figures/criterion-surfaces.png) | **[Estimating a Search Acceptance Rule by Simulation](computational-methods/simulation-based-estimation/)** | How to estimate a job-search acceptance rule three ways when the likelihood is hard to write but the model is easy to simulate. |
| [<img src="structural-econometrics/adversarial-estimation/figures/thumb.png" width="160">](structural-econometrics/adversarial-estimation/figures/smm-vs-adversarial.png) | **[Adversarial Structural Estimation](structural-econometrics/adversarial-estimation/)** | How to let an adversarial classifier pick the moments for a structural estimator, recovering optimally-weighted simulated moments with a logistic critic and maximum likelihood with a neural one. |
| [<img src="choice/mixed-logit-simulation/figures/thumb.png" width="160">](choice/mixed-logit-simulation/figures/choice-fit.png) | **[Mixed Logit Demand with Simulated Likelihood](choice/mixed-logit-simulation/)** | How to estimate demand when shoppers differ in price sensitivity and quality taste, by averaging logit probabilities over simulated tastes to break restrictive substitution patterns. |
| [<img src="structural-econometrics/rum-choice-networks/figures/thumb.png" width="160">](structural-econometrics/rum-choice-networks/figures/choice-fit.png) | **[Choice Prediction with RUMnets](structural-econometrics/rum-choice-networks/)** | How to enrich a discrete-choice model with a flexible neural utility surface while keeping choice probabilities consistent with random utility maximization. |
| [<img src="choice/sequential-search-ursu/figures/thumb.png" width="160">](choice/sequential-search-ursu/figures/search-and-choice-fit.png) | **[Consumer Search with Sequential Inspection Costs](choice/sequential-search-ursu/)** | How to estimate per-product search costs from observed inspection paths and purchases, using an optimal sequential-inspection rule and simulated moments. |
| [<img src="spatial-economics/allen-arkolakis/figures/thumb.png" width="160">](spatial-economics/allen-arkolakis/figures/equilibrium-wages-population.png) | **[Allen-Arkolakis Spatial Equilibrium on a Grid](spatial-economics/allen-arkolakis/)** | How wages and population shares are determined across many locations when trade costs, productivity spillovers, and congestion all interact through a fixed-point equilibrium. |
| [<img src="structural-econometrics/bayesian-dsge-hmc/figures/thumb.png" width="160">](structural-econometrics/bayesian-dsge-hmc/figures/posterior-irfs.png) | **[Bayesian DSGE Estimation by HMC/NUTS](structural-econometrics/bayesian-dsge-hmc/)** | How to recover the posterior over deep parameters of a small New Keynesian model by chaining a differentiable rational-expectations solver, a Kalman filter, and gradient-based Monte Carlo sampling. |

## Bayesian Methods

These tutorials teach Bayesian inference as a researcher's tool, ordered from closed-form conjugate examples through Monte Carlo machinery to function-space Bayes.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="bayesian-methods/bayesian-foundations/figures/thumb.png" width="160">](bayesian-methods/bayesian-foundations/figures/beta-posteriors.png) | **[Bayesian Foundations: Priors, Likelihoods, and Conjugate Posteriors](bayesian-methods/bayesian-foundations/)** | Conjugate prior-likelihood pairs deliver closed-form posteriors whose mean is a precision-weighted average of prior and data. |
| [<img src="computational-methods/metropolis-hastings/figures/thumb.png" width="160">](computational-methods/metropolis-hastings/figures/conjugate-posterior.png) | **[Posterior Sampling: Random-walk Metropolis-Hastings](computational-methods/metropolis-hastings/)** | Sample a two-mode posterior with no closed form using gradient-free random-walk proposals, benchmarked against a conjugate sanity check. |
| [<img src="computational-methods/mcmc-diagnostics/figures/thumb.png" width="160">](computational-methods/mcmc-diagnostics/figures/trace-plots.png) | **[MCMC Chain Diagnostics: ESS, R-hat, and Integrated Autocorrelation Time](computational-methods/mcmc-diagnostics/)** | Three diagnostics catch three failure modes hidden from trace plots: wasted draws, disagreement across chains, drift or heavy tails. |
| [<img src="time-series/minnesota-svar/figures/thumb.png" width="160">](time-series/minnesota-svar/figures/policy-shock-irfs.png) | **[Monetary Policy SVARs with Minnesota Priors](time-series/minnesota-svar/)** | Shrinkage stabilizes a small monetary-policy vector autoregression and a recursive ordering isolates the policy-rate shock. |
| [<img src="computational-methods/kalman-filter/figures/thumb.png" width="160">](computational-methods/kalman-filter/figures/simulated-signal.png) | **[Nowcasting a Latent Business-Cycle State by Kalman Filtering](computational-methods/kalman-filter/)** | Recursive Gaussian updating tracks a hidden activity state from a single noisy indicator each period. |
| [<img src="computational-methods/particle-filter/figures/thumb.png" width="160">](computational-methods/particle-filter/figures/filter-comparison.png) | **[Nowcasting Hidden Economic States by Particle Filtering](computational-methods/particle-filter/)** | Weighted simulations approximate the filtered distribution of a hidden state, and a weight-concentration diagnostic flags when most simulations carry no information. |
| [<img src="computational-methods/hamiltonian-monte-carlo/figures/thumb.png" width="160">](computational-methods/hamiltonian-monte-carlo/figures/posterior-coverage.png) | **[Hamiltonian Monte Carlo on a Banana Posterior](computational-methods/hamiltonian-monte-carlo/)** | Gradient-guided trajectories sample a curved banana posterior an order of magnitude more efficiently than isotropic random-walk proposals. |
| [<img src="bayesian-methods/neural-posterior-brock-hommes/figures/thumb.png" width="160">](bayesian-methods/neural-posterior-brock-hommes/figures/posterior-pairs.png) | **[Likelihood-Free Bayes by Neural Posterior Estimation](bayesian-methods/neural-posterior-brock-hommes/)** | A normalizing flow trained on simulator output returns a joint posterior over four parameters of an asset-pricing model with no tractable likelihood. |
| [<img src="numerical-methods/gaussian-processes/figures/thumb.png" width="160">](numerical-methods/gaussian-processes/figures/posterior-fit.png) | **[Gaussian Process Regression and Uncertainty Quantification](numerical-methods/gaussian-processes/)** | A Gaussian process gives a closed-form posterior over functions with credible bands, and marginal likelihood picks the smoothness scale from data. |
| [<img src="numerical-methods/bayesian-optimization/figures/thumb.png" width="160">](numerical-methods/bayesian-optimization/figures/bo-iterations.png) | **[Bayesian Optimization with a Gaussian-Process Surrogate](numerical-methods/bayesian-optimization/)** | A surrogate posterior plus expected improvement finds the global optimum of an expensive function in roughly thirty evaluations, against thousands for random or annealing baselines. |

## Choice and Demand

Choice and demand focuses on revealed preference, learning, and choice models.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="choice/revealed-preference-afriat/figures/thumb.png" width="160">](choice/revealed-preference-afriat/figures/budget-lines-consistent.png) | **[Consumer Rationalizability with Afriat's Test](choice/revealed-preference-afriat/)** | Whether a finite set of budget choices could have come from one stable utility function, checked by closing the revealed-preference graph. |
| [<img src="choice/preference-recoverability/figures/thumb.png" width="160">](choice/preference-recoverability/figures/budget-lines.png) | **[Recovering Preference Bounds from Budget Choices](choice/preference-recoverability/)** | Recovering the tightest upper-contour boundary through a chosen bundle by fitting Afriat inequalities to budget data. |
| [<img src="choice/money-pump-index/figures/thumb.png" width="160">](choice/money-pump-index/figures/money-pump-cycle.png) | **[Revealed-Preference Cycles and the Money Pump Index](choice/money-pump-index/)** | Pricing a consumer's inconsistency by finding the worst revealed-preference cycle and reporting its average budget slack. |
| [<img src="choice/houtman-maks-rational-subsets/figures/thumb.png" width="160">](choice/houtman-maks-rational-subsets/figures/conflict-graph.png) | **[Rationalizable Choice Cores with Houtman-Maks](choice/houtman-maks-rational-subsets/)** | Finding the largest subset of choices that survives the rationality test by removing the fewest conflicting observations. |
| [<img src="choice/revealed-price-preference/figures/thumb.png" width="160">](choice/revealed-price-preference/figures/price-cost-ratios.png) | **[Price-Regime Revealed Preference](choice/revealed-price-preference/)** | Ranking tax or tariff regimes from cross-budget cost comparisons when only one bundle is observed under each price schedule. |
| [<img src="choice/logit-discrete-choice/figures/thumb.png" width="160">](choice/logit-discrete-choice/figures/log-likelihood-surface.png) | **[Product Demand with Plain Logit and IIA](choice/logit-discrete-choice/)** | Fitting price and quality tastes from product choices, then watching one product's lost buyers spread to rivals in proportion to existing market shares. |
| [<img src="choice/urn-behavioral-mixtures/figures/thumb.png" width="160">](choice/urn-behavioral-mixtures/figures/bayes-likelihood-ratio.png) | **[Are People Bayesian? Decision-Rule Mixtures via EM](choice/urn-behavioral-mixtures/)** | Sorting subjects into Bayesian updaters versus simpler cutoff rules by running an EM mixture on repeated urn-classification choices. |
| [<img src="choice/risk-aversion-monotone-choice/figures/thumb.png" width="160">](choice/risk-aversion-monotone-choice/figures/risky-choice-fits.png) | **[Lottery Risk Aversion with Monotone Choice](choice/risk-aversion-monotone-choice/)** | Estimating risk aversion from a ladder of binary lotteries while enforcing that the share choosing the riskier option rises with its odds of winning. |
| [<img src="choice/convex-time-budget-present-bias/figures/thumb.png" width="160">](choice/convex-time-budget-present-bias/figures/identification-profile.png) | **[Estimating Present Bias from Convex Time Budgets](choice/convex-time-budget-present-bias/)** | Separating present bias from long-run patience by exploiting front-end-delay variation in intertemporal allocation experiments. |
| [<img src="choice/consideration-set-estimation/figures/thumb.png" width="160">](choice/consideration-set-estimation/figures/menu-removal-asymmetry.png) | **[Stochastic Choice and Random Consideration Sets](choice/consideration-set-estimation/)** | Recovering both a preference ranking and per-product attention probabilities from how choice frequencies shift when the menu changes. |
| [<img src="choice/probability-distortion-mixture/figures/thumb.png" width="160">](choice/probability-distortion-mixture/figures/weighting-functions.png) | **[Heterogeneous Probability Distortion via Finite-Mixture EM](choice/probability-distortion-mixture/)** | Sorting subjects into expected-utility types versus probability-distorting types using an EM mixture on certainty-equivalent data. |
| [<img src="choice/nested-logit/figures/thumb.png" width="160">](choice/nested-logit/figures/elasticity-heatmap.png) | **[Cereal Demand with Nested Logit Substitution](choice/nested-logit/)** | Estimating cereal demand where substitution stays mostly within product groups, with instruments correcting for price endogeneity. |
| [<img src="choice/weitzman-search-rule/figures/thumb.png" width="160">](choice/weitzman-search-rule/figures/reservation-values.png) | **[Pandora's Box: Optimal Sequential Search and the Weitzman Reservation-Value Rule](choice/weitzman-search-rule/)** | Why opening boxes in decreasing reservation value is exactly optimal when each box hides a random reward and inspection is costly. |

## Computational Game Theory

These tutorials introduce computational methods to solve game theoretic equilibria.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="game-theory/normal-form-games/figures/thumb.png" width="160">](game-theory/normal-form-games/figures/pure-deviation-gains.png) | **[Finite Strategic Games and Nash Equilibrium Checks](game-theory/normal-form-games/)** | Locating pure Nash equilibria by scanning unilateral deviation gains, and recovering mixed equilibria from indifference conditions. |
| [<img src="game-theory/static-games/figures/thumb.png" width="160">](game-theory/static-games/figures/cournot-best-response.png) | **[Cournot Quantity Competition and Best-Response Iteration](game-theory/static-games/)** | Two firms iterating best responses to a fixed point, recovering the Cournot quantity that the closed-form first-order condition predicts. |
| [<img src="game-theory/first-price-auctions/figures/thumb.png" width="160">](game-theory/first-price-auctions/figures/bid-functions.png) | **[First-Price Auctions, Bid Shading, and Deviation Checks](game-theory/first-price-auctions/)** | Why bidders shade below their value in a sealed-bid auction, with a type-by-type deviation grid confirming the closed-form bid rule. |
| [<img src="game-theory/regret-matching/figures/thumb.png" width="160">](game-theory/regret-matching/figures/time-average-convergence.png) | **[Regret Matching and No-Regret Dynamics](game-theory/regret-matching/)** | A learning rule where players reweight actions by past regret and the time-average play settles on a correlated equilibrium. |
| [<img src="game-theory/cfr-asymmetric-auction/figures/thumb.png" width="160">](game-theory/cfr-asymmetric-auction/figures/bid-functions-asymmetric.png) | **[Asymmetric First-Price Auctions by Counterfactual Regret Minimization](game-theory/cfr-asymmetric-auction/)** | Solving for equilibrium bid functions when bidders draw from different value distributions, with regret-based learning replacing the broken closed form. |
| [<img src="game-theory/deep-optimal-auctions/figures/thumb.png" width="160">](game-theory/deep-optimal-auctions/figures/learned-mechanism.png) | **[Deep Learning for Optimal Auction Design](game-theory/deep-optimal-auctions/)** | Training a neural network to play the role of a revenue-maximizing auction, then auditing it against the Myerson reserve-price benchmark. |
| [<img src="game-theory/quantal-response-equilibrium/figures/thumb.png" width="160">](game-theory/quantal-response-equilibrium/figures/qre-path.png) | **[Market Entry with Quantal Response Equilibrium](game-theory/quantal-response-equilibrium/)** | Two firms making noisy entry decisions where each entry probability is the smoothed best response to the other. |

## Time Series

These tutorials cover stochastic processes, macroeconomic data, and forecasting.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="time-series/fred-macro-data/figures/thumb.png" width="160">](time-series/fred-macro-data/figures/time-series.png) | **[Business-Cycle Moments from a FRED-Style Macro Panel](time-series/fred-macro-data/)** | Detrending a small quarterly macro panel recovers business-cycle volatility, comovement, persistence, and an Okun slope. |
| [<img src="time-series/ar-processes/figures/thumb.png" width="160">](time-series/ar-processes/figures/ar1-irfs.png) | **[Fiscal-Shock Persistence and Income Dynamics](time-series/ar-processes/)** | The persistence coefficient of a fiscal shock controls how long the multiplier-accelerator economy keeps moving income above trend. |
| [<img src="time-series/reduced-form-var/figures/thumb.png" width="160">](time-series/reduced-form-var/figures/irf-by-ordering.png) | **[Reduced-Form VARs and Cholesky Impulse Responses](time-series/reduced-form-var/)** | Joint dynamics of output and prices fitted equation by equation, with a recursive ordering deciding which residual is the named shock. |
| [<img src="time-series/stock-watson/figures/thumb.png" width="160">](time-series/stock-watson/figures/factor-comparison.png) | **[Macro Forecasting with Stock-Watson Diffusion Indexes](time-series/stock-watson/)** | One common factor extracted from a wide panel of indicators beats an autoregressive baseline at forecasting industrial production. |
| [<img src="time-series/ridge-lasso-sparsity/figures/thumb.png" width="160">](time-series/ridge-lasso-sparsity/figures/forecast-comparison.png) | **[Policy Forecasting with Ridge, Lasso, and Sparsity](time-series/ridge-lasso-sparsity/)** | Ridge keeps many weak correlated signals while lasso selects a compact subset, and a sparse fit is not the same as a sparse economy. |

## Agent-Based Models

These tutorials simulate local behavior and market institutions, then compare the aggregate outcome with an economic benchmark.

| Preview | Tutorial | Description |
|---|---|---|
| [<img src="agent-based-models/schelling-segregation/figures/thumb.png" width="160">](agent-based-models/schelling-segregation/figures/phase-transition.png) | **[Schelling Segregation on a Checkerboard](agent-based-models/schelling-segregation/)** | How small individual preferences about neighbors can quietly sort a whole city into segregated blocks. |
| [<img src="agent-based-models/zero-intelligence-traders/figures/thumb.png" width="160">](agent-based-models/zero-intelligence-traders/figures/demand-supply-schedule.png) | **[Zero-Intelligence Traders in a Double Auction](agent-based-models/zero-intelligence-traders/)** | Even traders who bid completely at random recover most of the gains from a competitive double auction. |
| [<img src="agent-based-models/cobweb-arifovic-ga-learning/figures/thumb.png" width="160">](agent-based-models/cobweb-arifovic-ga-learning/figures/price-paths.png) | **[Cobweb Markets and Arifovic Genetic-Algorithm Learning](agent-based-models/cobweb-arifovic-ga-learning/)** | A population of boundedly rational farmers learning by genetic operators converges to the rational-expectations price in a cobweb market. |
| [<img src="agent-based-models/brock-hommes-asset-pricing/figures/thumb.png" width="160">](agent-based-models/brock-hommes-asset-pricing/figures/price-paths.png) | **[Brock-Hommes Asset Pricing with Strategy Switching](agent-based-models/brock-hommes-asset-pricing/)** | Asset prices drift from fundamentals when traders switch between mean-reverting and trend-following rules by recent profit. |
| [<img src="agent-based-models/algorithmic-collusion-q-learning/figures/thumb.png" width="160">](agent-based-models/algorithmic-collusion-q-learning/figures/price-paths.png) | **[Algorithmic Collusion by Q-Learning](agent-based-models/algorithmic-collusion-q-learning/)** | Two pricing agents that learn from their own past profits drift above the static Bertrand benchmark without any explicit coordination. |

## Selected External Resources

### Core Computational Economics

- [QuantEcon](https://github.com/QuantEcon)
- [John Stachurski GitHub](https://github.com/jstac)
- [OpenSourceEcon CompMethods](https://github.com/OpenSourceEcon/CompMethods)
- [OpenSourceEconomics](https://github.com/OpenSourceEconomics)
- [CompEcon (Iskhakov)](https://github.com/fediskhakov/CompEcon)
- [Sciences Po CompEcon CoursePack](https://github.com/ScPo-CompEcon/CoursePack)
- [EconRL](https://github.com/SimonHashtag/EconRL)

### Heterogeneous-Agent & HANK Models

- [Sequence-Jacobian](https://github.com/shade-econ/sequence-jacobian)
- [Rognlie ECON 411-3](https://github.com/mrognlie/econ411-3)
- [HARK](https://github.com/econ-ark/HARK)
- [Benjamin Moll Codes](https://benjaminmoll.com/codes/)
- [Quantitative Macro Models](https://github.com/hessjacob/Quantitative-Macro-Models)
- [BASEforHANK](https://github.com/BASEforHANK)

### Empirical IO & Structural Estimation

- [PyBLP](https://github.com/jeffgortmaker/pyblp)
- [Chris Conlon Grad IO](https://github.com/chrisconlon/Grad-IO)
- [Courthoud PhD Industrial Organization](https://github.com/matteocourthoud/Phd-Industrial-Organization)
- [respy](https://github.com/OpenSourceEconomics/respy)
- [Dynamic Structural Econometrics (DSE 2023)](https://github.com/dseconf/DSE2023)]
- [Kenneth Train Software](https://eml.berkeley.edu/~train/software.html)
- [Victor Aguirregabiria Computer Code](https://sites.google.com/view/victoraguirregabiriaswebsite/computer-code?authuser=0)
- [EmpiricalIO](https://github.com/kohei-kawaguchi/EmpiricalIO)
- [Archive of Empirical Dynamic Programming Research](https://github.com/CForg/Archive-of-Empirical-Dynamic-Programming-Research)

### DSGE, Dynamics, and Filtering

- [New York Fed DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)
- [Global DGSE Solver](https://github.com/gdsge/gdsge)
- [DSGE_mod](https://github.com/JohannesPfeifer/DSGE_mod)
- [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl)
- [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
