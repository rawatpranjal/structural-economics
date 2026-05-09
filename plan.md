# PyBehavior-Sourced Tutorial Plan

## Summary

Use PyBehavior as a source of behavioral-economics tutorial ideas, but implement the tutorials in this repository's existing self-contained `run.py` style. The goal is not to add behavioral labels for their own sake. Each tutorial should teach a computational tool that is not already covered cleanly by the current catalog.

The three strongest candidates are:

1. Convex time budget estimation: nonlinear structural estimation from continuous allocations.
2. Consideration set estimation: latent choice-set likelihood and hidden-set summation.
3. Probability weighting: nonlinear probability-distortion estimation from risky-choice data.

All three should live under `choice/`, generate their own `README.md`, `figures/`, and `tables/`, add compact rows to the root catalog, and pass `python3 scripts/validate_catalog.py`.

## Tutorial 1: Convex Time Budget Present Bias

Target folder:

```text
choice/convex-time-budget-present-bias/
```

Core lesson:

```text
Recover beta-delta time-preference parameters from continuous experimental
budget allocations using nonlinear structural estimation.
```

This is the strongest first tutorial because it is not another discrete-choice logit. The observed choice is a continuous allocation between sooner and later payments. The computational object is a nonlinear first-order condition or likelihood, estimated across designed budget variation.

Model setup:

```text
c_t + q c_{t+k} = m
U = u(c_t) + beta * delta^k * u(c_{t+k})
u(c) = c^(1-rho) / (1-rho)
```

Implementation requirements:

- Simulate a convex time budget design with front-end delay, delay length, gross interest rate, and budget variation.
- Generate continuous sooner/later allocations from known `beta`, `delta`, and `rho`.
- Estimate `beta`, `delta`, and `rho` by nonlinear least squares or likelihood using `scipy.optimize`.
- Include corner handling or mild interior noise so the estimation problem is realistic but stable.
- Compare a weak design with only today-vs-future choices against a stronger design with both today-vs-future and future-vs-future choices.
- Explain why front-end delay helps separate present bias from long-run discounting.

Expected outputs:

- `figures/budget-lines.png`: CTB budget lines with simulated allocations.
- `figures/allocation-response.png`: sooner allocation by interest rate and delay.
- `figures/identification-surface.png`: objective surface or profile contrast for `beta` and `delta`.
- `tables/parameter-recovery.csv`: true parameters, estimates, and uncertainty or bootstrap intervals.
- `tables/design-comparison.csv`: weak-design versus full-design fit and parameter recovery.

README emphasis:

- Start from the economic question: how do experimental budgets reveal present bias?
- Show the CTB budget equation and beta-delta utility equation.
- Keep the method section focused on nonlinear estimation from continuous choices.
- Make the main takeaway: CTB is a structural estimation problem over designed continuous allocations, not a binary-choice exercise.

## Tutorial 2: Consideration Set Estimation

Target folder:

```text
choice/consideration-set-estimation/
```

Core lesson:

```text
Infer latent choice sets by summing over hidden product subsets in the
observed-choice likelihood.
```

This is the second-best candidate because it teaches a genuinely different computational operation: likelihood evaluation with a latent combinatorial object. The consumer's final product choice is observed, but the set of products considered before choice is not.

Model setup:

```text
Stage 1: product j enters consideration with probability pi_j
Stage 2: conditional on C, choose among products in C by utility

P(choose j) = sum_{C contains j} P(choose j | C) P(C)
```

Use a small product universe, for example `J = 5`, so exact enumeration over all nonempty consideration sets is feasible and transparent.

Implementation requirements:

- Simulate products with price, quality, display status, and availability.
- Let display or prominence affect consideration.
- Let price and quality affect final utility.
- Enumerate all nonempty consideration sets for small `J`.
- Compute the observed-choice likelihood by summing over sets that contain the chosen product.
- Estimate attention and preference parameters by MLE.
- Compare the latent-set model against a full-choice-set multinomial logit.
- Show how a high-utility product can have low sales because it is rarely considered.

Expected outputs:

- `figures/consideration-recovery.png`: true versus estimated consideration probabilities.
- `figures/share-decomposition.png`: observed shares, full-information predictions, and consideration-set predictions.
- `figures/display-counterfactual.png`: demand shift when one product receives more display attention.
- `tables/parameter-recovery.csv`: true and estimated attention/preference parameters.
- `tables/model-comparison.csv`: latent consideration model versus full-choice-set logit.

README emphasis:

- Start from the demand interpretation: low market share can mean low attention rather than low utility.
- Show the hidden-set likelihood clearly.
- Make enumeration explicit in the algorithm block.
- State the computational limitation: exact summation scales as `2^J`, so the tutorial uses small `J` to teach the object.

## Tutorial 3: Probability Weighting and Prospect Theory

Target folder:

```text
choice/probability-weighting-lottery-choice/
```

Core lesson:

```text
Estimate a constrained nonlinear probability-distortion function from
risky choices or certainty equivalents.
```

This is useful, but it should be third. A weak version would only plot prospect-theory curves. The worthwhile version teaches nonlinear transformation estimation, certainty-equivalent inversion, and identification problems between utility curvature, probability weighting, and choice noise.

Model setup:

```text
Expected utility:
EU = p u(x_1) + (1 - p) u(x_0)

Probability weighting:
PT = w(p) v(x_1) + [1 - w(p)] v(x_0)

Prelec weighting:
w(p) = exp(-eta * (-log p)^alpha)
```

Implementation requirements:

- Simulate lottery choices or certainty equivalents across probabilities, prizes, and safe amounts.
- Estimate Prelec weighting parameters by nonlinear least squares or likelihood.
- Compare expected utility, probability weighting only, and full prospect-theory value weighting.
- Include certainty-equivalent inversion by solving for the sure amount that makes the subject indifferent to a lottery.
- Include an identification comparison showing that weak probability variation confounds probability weighting with utility curvature.
- Enforce basic shape restrictions through parameter bounds: positive `eta`, positive `alpha`, and monotone decision weights.

Expected outputs:

- `figures/probability-weights.png`: objective probabilities versus estimated decision weights.
- `figures/certainty-equivalents.png`: certainty equivalents by probability and prize.
- `figures/model-fit.png`: expected utility versus probability-weighting fit.
- `figures/identification-comparison.png`: strong-design versus weak-design parameter recovery.
- `tables/parameter-recovery.csv`: true and estimated `alpha`, `eta`, curvature, and choice-noise parameters.
- `tables/model-comparison.csv`: expected utility, probability weighting, and prospect-theory fit metrics.

README emphasis:

- Avoid making the tutorial just "people overweight small probabilities."
- Put the computational object first: estimate a nonlinear function under shape restrictions.
- Explain why lotteries and insurance can both be attractive under probability distortion.
- State the identification caveat: curvature and probability weighting can mimic each other unless the design has enough probability and payoff variation.

## Catalog Placement

Add all three tutorials under `Choice and Demand`.

Suggested root catalog descriptions:

```text
Estimating Present Bias with Convex Time Budgets
Continuous budget allocations identify present bias, patience, and curvature. Nonlinear estimation recovers beta-delta preferences from designed intertemporal choices.

Latent Consideration Sets in Product Choice
Consumers may not evaluate every product. Enumerating hidden choice sets separates attention from preference in observed demand.

Probability Weighting in Lottery Choice
Risky choices can reveal distorted probability weights. Nonlinear estimation recovers a Prelec weighting curve and compares it with expected utility.
```

## Implementation Order

1. Build `choice/convex-time-budget-present-bias/`.
2. Build `choice/consideration-set-estimation/`.
3. Build `choice/probability-weighting-lottery-choice/`.

This order prioritizes the most distinct computational contribution first. CTB adds continuous-choice nonlinear structural estimation. Consideration sets add latent discrete-set likelihoods. Probability weighting adds nonlinear transformation estimation and identification discipline.

## Test Plan

For each tutorial:

- Run `python run.py` from the tutorial folder.
- Confirm `README.md`, `figures/`, and `tables/` are regenerated.
- Inspect the generated README directly for readable equations and working image links.
- Run `python3 scripts/validate_catalog.py` from the repo root.
- Keep dependencies to the current stack: `numpy`, `scipy`, `pandas`, and `matplotlib`.
- Do not add PyBehavior or PyTorch as runtime dependencies.

## Assumptions

- PyBehavior is an idea source, not a dependency.
- The examples should be synthetic, transparent, and pedagogical.
- Each tutorial should teach a reusable computational tool.
- Root README descriptions should stay short and plain.
- Generated tutorial READMEs should be inspected directly before considering the work done.

## Addendum: Interactive Pricing Theory Source

The strongest computational method in `rawatpranjal/interactive-pricing-theory` is not a static pricing formula. It is dynamic online pricing with partial identification: a bandit algorithm that uses revealed-preference restrictions to eliminate dominated prices while learning demand.

Recommended target folder:

```text
industrial-organization/online-pricing-partial-identification/
```

Core lesson:

```text
Use economic structure to turn purchase/no-purchase observations into bounds
on demand, then combine those bounds with bandit exploration.
```

Why this should outrank the PyBehavior candidates if the goal is a hardcore computational method:

- It is an actual learning algorithm, not just nonlinear parameter estimation.
- It teaches regret, exploration, UCB, Thompson sampling, and dominance elimination.
- The economic content is essential: WARP-style monotonicity lets a buy at price `p` imply willingness to buy below `p`, while a no-buy implies an upper valuation bound.
- The tutorial can show the computational value of structure by comparing regret rates across epsilon-greedy, learn-then-earn, UCB1, Thompson sampling, and UCB with partial identification.

Model setup:

```text
There are K candidate prices and S customer segments.
Each segment s has an unknown valuation interval.
At round t, the seller posts price p_t and observes purchase/no purchase.

If segment s buys at p_t:
    v_s >= p_t, so the lower valuation bound rises.
If segment s does not buy at p_t:
    v_s < p_t, so the upper valuation bound falls.

The bounds imply lower and upper demand at each price.
Profit lower bound:  p * D_L(p)
Profit upper bound:  p * D_U(p)

Eliminate price p if:
    upper_profit(p) <= max_q lower_profit(q)
```

Implementation requirements:

- Simulate segmented demand with a discrete price grid.
- Implement benchmark algorithms: epsilon-greedy, learn-then-earn, UCB1, and Thompson sampling.
- Implement UCB-PI: maintain segment valuation bounds, compute profit bounds, eliminate dominated prices, and apply an optimistic index over active prices.
- Track cumulative regret against the oracle best fixed price.
- Show how the active price set shrinks as the algorithm learns.
- Keep the example self-contained in NumPy/SciPy/Matplotlib; do not depend on the pricing-theory TypeScript app.

Expected outputs:

- `figures/regret-comparison.png`: cumulative regret for all algorithms on log-log axes.
- `figures/active-prices.png`: active price count over time under partial identification.
- `figures/profit-bounds.png`: lower and upper profit bounds across the price grid at selected rounds.
- `figures/valuation-intervals.png`: segment-level valuation intervals tightening with observations.
- `tables/final-regret.csv`: final regret, average regret, oracle price, learned best price, and active price count.
- `tables/elimination-diagnostics.csv`: number of dominated prices eliminated at checkpoints.

README emphasis:

- Do not present this as "another bandit tutorial."
- Put the new computational tool first: revealed-preference bounds inside online learning.
- Make the comparison against ordinary UCB and Thompson sampling central.
- Explain why economic restrictions can buy faster learning: fewer plausible prices remain as observations accumulate.
- Be careful with claims about formal rates; report empirical regret slopes from the simulation unless a theorem is stated exactly.

Runner-up methods from the same source:

1. `dynamic-pricing-sawtooth`: Gallego-van Ryzin finite-inventory dynamic pricing by backward induction. This is clean and computational, but the repo already has many dynamic programming tutorials.
2. `network-rm`: deterministic linear programming for network revenue management and bid-price controls. This is useful if adding a revenue-management section.
3. `markdown-management`: discrete price-ladder dynamic programming with a markdown-only constraint and strategic waiting. This is strong, but less novel than UCB-PI once the repo already has dynamic programming examples.
4. `dynamic-durable-games`: Markov-perfect dynamic oligopoly. Skip for now because the repo already has dynamic games and dynamic-games estimation tutorials.
