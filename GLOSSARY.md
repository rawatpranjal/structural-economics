# Internal Glossary

This file is an internal reference for contributors and agents working in the
`After Quant Econ` repository. It records recurring notation, economic terms,
computational terms, and catalog language used across active tutorials.

It is not meant to be exhaustive. Tutorial-specific notation belongs in that
tutorial's generated `README.md`. Use this glossary to keep new tutorials and
catalog rows consistent with the current repo style.

Writing and presentation conventions live in `STYLE_GUIDE.md`; this glossary is
only for terms, notation, and shared catalog language.

## General Notation Conventions

### Time, Counts, and Indices

| Notation | Meaning | Repo convention |
|---|---|---|
| $t$ | Time period or observation index | Use for dynamic paths, time series, and revealed-preference observations. |
| $T$ | Number of periods, markets, observations, or maturity dates | Define locally in Model Setup because its meaning changes by tutorial. |
| $T_{sim}$ | Simulation horizon | Use when distinguishing simulated periods from model or data periods. |
| $N$ | Number of grid points, states, agents, draws, or observations | Qualify in text; avoid leaving bare $N$ ambiguous. |
| $i$ | Agent, consumer, player, or row index | Common for households, simulated consumers, and game rows. |
| $j$ | Product, next-state, alternative, or column index | Common for products in demand and next states in Markov sums. |
| $k$ | Next state, product, or summation index | Use as a local helper index when $i$ and $j$ are already taken. |
| $m$ | Market or moment index | Use when market-level notation needs a separate index. |

### Agents, Products, and Markets

| Notation | Meaning | Repo convention |
|---|---|---|
| $i$ | Household, consumer, firm, player, or simulated type | Define the economic role at first use. |
| $j$ | Product, action, asset grid point, or state grid point | In demand tutorials, reserve $j$ for products when possible. |
| $t$ | Market, date, or observation | In BLP-style demand, $t$ often indexes markets. In time series, $t$ indexes dates. |
| $J$ | Products or actions | In differentiated-products demand, $J$ is products per market. |
| $s_{jt}$ | Market share of product $j$ in market $t$ | Use $s_0$ for the outside-good share in logit and nested-logit settings. |
| $p_{jt}$ | Price of product $j$ in market $t$ | In choice-theory tutorials, $p_t$ can instead be a price vector for observation $t$. |
| $x_{jt}$ | Product characteristic | In revealed preference, $x_t$ usually means a chosen bundle. |

### States, Controls, and Shocks

| Notation | Meaning | Repo convention |
|---|---|---|
| $s$ | Generic state or hidden state | Use for state-space and dynamic models when the state is not specifically assets or capital. |
| $a$ | Assets or current asset state | Use in consumption-savings and heterogeneous-agent tutorials. |
| $a'$ | Next-period assets or savings choice | The prime denotes the chosen next state. |
| $k$ | Capital | Use in growth, RBC, and macro tutorials. |
| $W$ | Resource stock or wealth | Used in cake-eating and simple resource problems. |
| $c$ | Consumption | A common control in dynamic macro models. |
| $z$ | Exogenous shock state or instrument | In shock-discretization and macro, $z_t$ is usually a shock. In demand estimation, $z$ can be an instrument. Define locally. |
| $y$ | Income, output, yield, or observed signal | Define locally because finance, macro, and filtering tutorials use $y$ differently. |
| $\epsilon_t$, $\varepsilon_{ijt}$ | Shock or logit error | Use $\epsilon$ for model shocks and $\varepsilon$ for idiosyncratic choice errors when the distinction matters. |
| $\xi_{jt}$ | Unobserved product quality or demand shock | Standard in differentiated-products demand and BLP-style notation. |
| $\nu_i$ | Simulated consumer taste draw | Common in random-coefficients demand. |

### Parameters and Preferences

| Notation | Meaning | Repo convention |
|---|---|---|
| $\beta$ | Discount factor | Usually in $(0,1)$; common in Bellman equations and Euler equations. |
| $\rho$ | Persistence or continuous-time discount rate | In AR(1) models, $\rho$ is persistence. In continuous time, it may be the discount rate. Define locally. |
| $r$ | Interest rate | Often exogenous in partial-equilibrium household problems and endogenous in general equilibrium. |
| $R$ | Gross return | Usually $R=1+r$. |
| $\sigma$ | Risk aversion, shock scale, or random-coefficient scale | Define locally; $\sigma$ is overloaded across tutorials. |
| $\gamma$ | CRRA risk aversion | Often used when $\sigma$ is needed for shock volatility or random coefficients. |
| $\alpha$ | Price coefficient, production share, or policy parameter | In demand, $\alpha$ is usually price sensitivity and should have the expected sign. |
| $\theta$ | Generic structural parameter vector | Use when several primitive parameters are estimated or calibrated together. |
| $u(c)$ | Flow utility | For CRRA, use $u(c)=c^{1-\sigma}/(1-\sigma)$ or the log special case. |

### Grids, Policies, and Value Functions

| Notation | Meaning | Repo convention |
|---|---|---|
| $V(s)$ | Value function | Expected discounted value starting from state $s$. |
| $V_n(s)$ | Iteration-$n$ value function | Use in VFI convergence descriptions. |
| $g(s)$ or $g_a(a,y)$ | Policy function | Maps states into controls or next states. Name the chosen object in text. |
| $c^*(s)$ | Optimal consumption policy | Use a star for an optimizer when it improves readability. |
| $P$ | Transition matrix, price, covariance, or bond price | Avoid bare $P$ when there is possible conflict; say "transition matrix" or "price" in prose. |
| $P_{ij}$ | Markov transition probability | Probability of moving from current state $i$ to next state $j$. |
| $\pi$ | Stationary or invariant distribution | Usually satisfies $\pi=\pi P$ and sums to one. |
| $\mu$ | Marginal utility, mean, or type distribution | Define locally; avoid using it for several objects in the same tutorial. |
| $\delta_{jt}$ | Mean utility in BLP-style demand | Recovered by Berry inversion or BLP contraction. |

### Naming in Code and Files

| Item | Convention |
|---|---|
| Tutorial folder | `kebab-case`, one economic object per folder. |
| Python functions and variables | `snake_case`. |
| Executable entrypoint | `run.py` in each active tutorial folder. |
| Generated report | `README.md` produced by `lib.output.ModelReport`; visible `Reproduce` sections and figure captions are optional per tutorial. |
| Figures | `figures/*.png`, with `figures/thumb.png` for the root catalog; use useful alt text or nearby prose even when visible captions are omitted. |
| Tables | `tables/*.csv` when a tutorial has tabular outputs. |
| Legacy material | Store under `_legacy/`; active tutorials should not depend on it at runtime. |

## Core Economics Terms

### Dynamic Programming

| Term | Meaning in this repo |
|---|---|
| Bellman equation | Recursive problem that writes today's value as current payoff plus discounted continuation value. |
| State | The information needed today to evaluate feasible choices and future values. |
| Control | The choice variable selected at the current state, such as consumption, savings, or investment. |
| Value function | Function giving the optimized lifetime value at each state. |
| Policy function | Function mapping states into optimal controls or next states. |
| Transition law | Rule or probability matrix describing how states evolve after choices and shocks. |
| Continuation value | Expected future value term inside a Bellman equation. |
| Discount factor | Weight on future utility or payoffs, usually $\beta$. |
| Contraction | Mapping that brings successive guesses closer together and guarantees a unique fixed point. |
| Fixed point | Object unchanged by the update rule, such as a value function, price vector, policy, or equilibrium strategy. |
| Euler equation | First-order condition equating marginal utility today with discounted expected marginal utility tomorrow. |
| Borrowing constraint | Lower bound on assets or debt, often written $a'\geq 0$ or $a'\geq \underline a$. |
| Reservation rule | Threshold policy, such as accepting job offers above a reservation wage. |
| Stationary distribution | Long-run distribution implied by policies and transition probabilities. |

### Macro and Heterogeneous Agents

| Term | Meaning in this repo |
|---|---|
| Representative agent | Single household or planner used to stand in for the aggregate economy. |
| Heterogeneous agents | Models with households or agents that differ by income, wealth, preferences, or constraints. |
| Idiosyncratic risk | Agent-specific risk that does not average out for the individual but may wash out in the aggregate. |
| Aggregate shock | Economy-wide shock, such as productivity in RBC models. |
| Precautionary savings | Extra saving generated by income risk, prudence, and borrowing limits. |
| Buffer stock | Wealth held to self-insure against bad income or productivity draws. |
| Aiyagari equilibrium | General equilibrium with incomplete markets, idiosyncratic income risk, and capital-market clearing. |
| Huggett equilibrium | Incomplete-markets equilibrium often used to study borrowing constraints and interest-rate determination. |
| Capital market clearing | Condition that aggregate household assets equal the capital demand or asset supply in equilibrium. |
| MPC | Marginal propensity to consume out of an income or wealth transfer. |
| Gini coefficient | Scalar measure of inequality, commonly applied to simulated wealth distributions. |
| Steady state | Time-invariant allocation or equilibrium object. |
| Saddle path | Stable path converging to a saddle-point steady state in continuous-time or optimal-control models. |
| HJB equation | Hamilton-Jacobi-Bellman equation for continuous-time dynamic optimization. |
| KFE | Kolmogorov Forward Equation describing the evolution of a distribution over states. |

### Choice and Demand

| Term | Meaning in this repo |
|---|---|
| Bundle | Vector of goods chosen by a consumer in revealed-preference tutorials. |
| Budget set | Bundles affordable at observed prices and expenditure. |
| Revealed preference | Inference from chosen bundles and prices about what the consumer preferred. |
| GARP | Generalized Axiom of Revealed Preference; the main finite-data rationalizability test. |
| Afriat inequalities | Linear inequalities whose feasibility is equivalent to rationalizability by a well-behaved utility function. |
| Rationalizable | Consistent with some utility-maximizing model under the maintained assumptions. |
| Logit demand | Discrete-choice demand model with T1EV errors and closed-form choice probabilities. |
| IIA | Independence of Irrelevant Alternatives; substitution in plain logit is proportional to existing shares. |
| Nested logit | Logit model with grouped alternatives and stronger substitution within nests. |
| Berry inversion | Transformation from observed shares to mean utilities in logit-style demand. |
| BLP | Random-coefficients demand model allowing heterogeneous consumer tastes. |
| Mean utility | Product-level utility component common across consumers, usually $\delta_{jt}$. |
| Outside good | No-purchase option or omitted alternative, usually share $s_0$. |
| Price endogeneity | Correlation between price and unobserved quality or demand shocks. |
| Instrument | Variable that shifts price or another endogenous regressor but is excluded from utility or demand shocks. |

### Industrial Organization

| Term | Meaning in this repo |
|---|---|
| Differentiated products | Products that are imperfect substitutes because characteristics, brands, or qualities differ. |
| Bertrand-Nash pricing | Price equilibrium where each firm chooses prices given rivals' prices. |
| Ownership matrix | Matrix recording which products are controlled by the same firm. |
| First-order condition | Optimality condition used to recover markups, marginal costs, or prices. |
| Markup | Price minus marginal cost, or a ratio based on that wedge. |
| Marginal cost | Cost of producing one more unit, often recovered from demand and pricing first-order conditions. |
| Merger simulation | Counterfactual price or welfare exercise under changed product ownership. |
| HHI | Herfindahl-Hirschman Index, a concentration measure based on market shares. |
| GUPPI | Gross Upward Pricing Pressure Index, a merger-screening statistic. |
| Vertical relationship | Interaction between upstream and downstream firms, such as wholesale and retail pricing. |
| Double marginalization | Pricing distortion when both upstream and downstream firms add markups. |
| Two-part tariff | Contract with a per-unit price and a fixed fee. |
| Three-part tariff | Contract with fixed fee, included allowance, and overage price. |
| Nash-in-Nash | Bilateral bargaining solution applied across multiple linked negotiations. |
| Dynamic game | Game where current actions affect future states and payoffs. |
| MPE | Markov-perfect equilibrium; strategies depend on payoff-relevant states. |
| Entry and exit | Firm decisions to enter or leave a market, often governed by sunk costs and continuation values. |
| CCP | Conditional choice probability; probability of an action conditional on the state in dynamic discrete choice. |

### Game Theory

| Term | Meaning in this repo |
|---|---|
| Normal-form game | Payoff table listing payoffs for every action profile. |
| Action profile | One action for each player. |
| Payoff matrix | Matrix of player payoffs in a finite game. |
| Best response | Action that maximizes a player's payoff given rivals' actions or mixed strategies. |
| Nash equilibrium | Action profile or mixed strategy where every player is best responding. |
| Mixed strategy | Probability distribution over actions. |
| Indifference condition | Equation making a player indifferent among actions used with positive probability. |
| Bayesian Nash equilibrium | Strategy profile optimal given beliefs and private information. |
| QRE | Quantal Response Equilibrium; noisy best-response fixed point. |
| Unilateral-deviation check | Verification that no single player can profitably deviate. |
| Equilibrium residual | Numerical error in a best-response, indifference, or fixed-point condition. |

### Time Series and Data

| Term | Meaning in this repo |
|---|---|
| AR(1) | First-order autoregressive process, often $z_{t+1}=\rho z_t+\sigma_\epsilon\epsilon_{t+1}$. |
| Persistence | Degree to which shocks carry into future periods, often $\rho$. |
| Innovation | New shock or forecast surprise. |
| State-space model | System with hidden state transition equation and observation equation. |
| Kalman filter | Recursive prediction-update algorithm for linear Gaussian state-space models. |
| Posterior covariance | Filtered uncertainty about hidden states after seeing data. |
| Likelihood | Probability density of observed data under a model, often accumulated recursively. |
| HP filter | Hodrick-Prescott filter used to separate trend and cycle in macro data. |
| Factor model | Model where many series are summarized by a small number of latent factors. |
| PCA | Principal components analysis, used to estimate factors from a panel. |
| IRF | Impulse response function, the path of variables after a shock. |
| Forecast benchmark | Simple comparison model, such as no-change or AR(1), used to judge predictive value. |

### Computational Finance

| Term | Meaning in this repo |
|---|---|
| Cash flow | Promised payment at a future date. |
| Present value | Discounted value today of future cash flows. |
| Yield to maturity | Single discount rate that prices a promised cash-flow stream. |
| Coupon bond | Bond with periodic coupon payments and final face-value repayment. |
| Par value | Face value of a bond, often 100 in teaching examples. |
| Term structure | Relationship between yields and maturities. |
| Yield curve | Cross-section of yields by maturity at a date. |
| Forward rate | Rate implied by yields for borrowing or lending over a future interval. |
| Random-walk diagnostic | Test or plot asking whether price or return changes are predictable from past changes. |
| Efficient-market test | Joint test of market efficiency and the expected-return model. |
| Mean-variance frontier | Portfolio frontier tracing the best expected return for each risk level, or least risk for each return. |
| Covariance matrix | Matrix of asset return comovements that drives portfolio risk. |

## Computational Terminology

| Term | Meaning in this repo |
|---|---|
| VFI | Value Function Iteration; repeated application of the Bellman operator until the value function converges. |
| EGP | Endogenous Grid Points; Euler-equation inversion method for savings problems. |
| EEI | Envelope Equation Iteration; iteration on marginal value rather than the value level. |
| STPFI | Simultaneous Transition and Policy Function Iteration for global nonlinear DSGE models. |
| NFXP | Nested Fixed Point; outer parameter search with an inner model solution or inversion. |
| Bellman operator | Mapping from a guessed value function to an updated value function. |
| Convergence tolerance | Numerical threshold for stopping an iteration, often a sup-norm difference. |
| Sup norm | Maximum absolute difference across grid points. |
| Grid | Finite set of state or control points used to approximate a continuous object. |
| Tensor-product grid | Multidimensional grid built from all combinations of one-dimensional grids. |
| Discretization | Approximation of a continuous process or state space by finite points and probabilities. |
| Tauchen method | Interval-based Markov approximation to a Gaussian AR(1). |
| Rouwenhorst method | Markov approximation designed to preserve persistence on coarse grids. |
| Interpolation | Approximation of function values between grid points. |
| Collocation | Projection method that enforces equations at selected grid points. |
| Projection method | Approximation method using basis functions, such as Chebyshev polynomials. |
| Root finding | Numerical search for a zero of an equation. |
| Bisection | Robust one-dimensional root-finding method based on sign changes. |
| Newton method | Derivative-based root or optimization method using local slope or curvature. |
| Fixed-point iteration | Repeatedly applying a map until the object stops changing. |
| Simulation draws | Random or quasi-random draws used to approximate expectations or distributions. |
| Monte Carlo error | Sampling error caused by using finite simulation draws. |
| Stationary simulation | Long simulation used to approximate an invariant distribution. |
| Finite difference | Approximation to derivatives using nearby grid values. |
| Upwind scheme | Finite-difference scheme choosing derivative direction based on state drift. |
| Filtering | Recursive inference about hidden states from observed signals. |
| GMM | Generalized Method of Moments; parameter choice minimizing moment violations. |
| IV/2SLS | Instrumental variables / two-stage least squares; used when regressors are endogenous. |
| MLE | Maximum likelihood estimation; parameter choice maximizing data likelihood. |
| MCMC | Markov Chain Monte Carlo; simulation method for sampling from a target distribution. |
| Particle filter | Sequential Monte Carlo method for nonlinear or non-Gaussian filtering. |

## Report and Catalog Terminology

| Term | Meaning in this repo |
|---|---|
| Tutorial | Self-contained active folder with `run.py`, generated `README.md`, figures, and optional tables. |
| Root catalog | The root `README.md` listing active tutorials by economic subject. |
| Catalog row | One root README table row with thumbnail, title, topic, method, and key insight. |
| Method label | Short description of the main numerical or empirical method in a catalog row. |
| Key insight | One-sentence economic or computational takeaway shown in the root catalog. |
| Generated README | Tutorial report written by `lib.output.ModelReport`, not hand-maintained prose. |
| Overview | Report section explaining the economic question and why it matters. |
| Equations | Report section with the model equations or empirical moments. |
| Model Setup | Report section listing calibrated parameters, data objects, grids, or simulation settings. |
| Solution Method | Report section explaining the algorithm or estimator. |
| Results | Report section with figures, tables, and short interpretation. |
| Takeaway | Report section summarizing what the tutorial teaches. |
| Reproduce | Report section with the local `python run.py` command. |
| References | Report section listing canonical sources used by the tutorial. |
| Thumbnail | `figures/thumb.png`, used by the root catalog. |
| Active folder | Polished tutorial folder outside `_legacy/`. |
| Legacy material | Source notebooks, handouts, old scripts, or PDFs preserved under `_legacy/`. |
