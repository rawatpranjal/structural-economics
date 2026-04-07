# Simulation

### 1. Merger Simulation (The Industry Standard)
This is the most common application of BLP in antitrust and consulting.
**Question:** "If Firm A acquires Firm B, by how much will prices rise?"

**The Logic (Unilateral Effects):**
*   Before the merger, Firm A sets prices to maximize its own profit. It ignores that raising the price of Product A sends customers to Product B.
*   After the merger, the joint firm *internalizes* this. Raising the price of Product A is less painful because the lost customers switch to Product B, which the firm now owns. This incentive drives prices up.

**The Algorithm:**
1.  **Retrieve Marginal Costs ($c$):** We calculated these during the Estimation phase. We assume these **do not change** (unless we explicitly model "synergies").
2.  **Update the Ownership Matrix ($\Omega \to \Omega^{post}$):**
    *   In the pre-merger $\Omega$, $\Omega_{jk} = 1$ only if $j$ and $k$ were owned by the same firm.
    *   In $\Omega^{post}$, we switch the zeros to ones for the merging products (e.g., Chevy and Ford now treat each other as "owned").
3.  **Solve for New Prices ($p^*$):**
    We must find the new price vector that satisfies the First Order Condition with the *new* ownership matrix:
    $$p^* = c + [\Omega^{post}(p^*)]^{-1} s(p^*)$$
    *   *Note:* This is hard because shares $s(p^*)$ and derivatives inside $\Omega(p^*)$ depend on the price $p^*$.
    *   *Solution:* We use a numerical fixed-point iteration solver to find the new equilibrium vector $p^*$.

---

### 2. Welfare Analysis (Consumer Surplus)
**Question:** "How much value did consumers gain from the introduction of the iPhone?" or "How much did the merger harm consumers?"

In Logit models, Consumer Surplus (CS) has a closed-form solution (the "Log-Sum" formula).

**For a single consumer $i$:**
$$CS_{it} = \frac{1}{\alpha_i} \ln \left( \sum_{j=0}^{J} \exp(V_{ijt}) \right) + C$$
*   $\alpha_i$: The consumer's price sensitivity (marginal utility of income).
*   $\sum \exp(V)$: The denominator of the Logit equation (inclusive value).

**Total Consumer Welfare:**
We integrate (average) this over the population distribution $P(v, d)$:
$$W = M \int CS_{it} dP(v, d)$$

**Calculating the Change ($\Delta W$):**
To measure the impact of a policy (e.g., a tariff removing a product or raising its price):
$$\Delta W = \int \frac{1}{\alpha_i} \left[ \ln \left( \sum_{j \in \text{Post}} e^{V_{ij}} \right) - \ln \left( \sum_{j \in \text{Pre}} e^{V_{ij}} \right) \right] dP(v, d)$$

---

### 3. Elasticities & Diversion Ratios (Diagnostics)
Before running a merger simulation, practitioners check the **Diversion Ratios**. This tells us *where* volume goes when prices rise.

**Formula:**
$$D_{jk} = \frac{\partial s_k / \partial p_j}{|\partial s_j / \partial p_j|}$$
*Translation:* "If Product $j$ loses 100 units sales due to a price hike, how many of those units go to Product $k$?"

*   **Why BLP wins here:** In Simple Logit, diversion is purely proportional to market share ($D_{jk} \propto s_k$). In BLP, if Product $j$ and Product $k$ are similar (e.g., both are expensive luxury cars), the correlation in random coefficients will create a **high diversion ratio**, even if their market shares are small.

---

### 4. Key Assumptions & Gotchas in Simulation
In an interview, listing these shows you understand the limitations of the model.

1.  **$\xi$ is Invariant:** We assume the Unobserved Quality ($\xi_{jt}$) stays the same during the simulation.
    *   *Critique:* If a merger leads to a re-branding or quality degradation, BLP misses it.
2.  **Marginal Costs are Constant:** Unless you manually adjust $c$ to simulate "efficiencies" (e.g., the merger reduces costs by 5%), the model assumes costs are static.
3.  **Equilibrium Existence:** When solving for post-merger prices $p^*$, there is no mathematical guarantee that a unique equilibrium exists, or that the solver will find it. (Though in practice, with Logit demand, it usually works).
4.  **The Lucas Critique:** We assume consumer preference parameters ($\beta, \sigma$) are structural and do not change just because the market structure changed.

---

### Summary of the Full BLP Workflow
1.  **Data:** Shares, Prices, Characteristics, Instruments.
2.  **Model:** $U = \delta + \mu + \epsilon$. Heterogeneity driven by demographics and random shocks.
3.  **Estimation (NFXP):**
    *   Inner Loop: Invert shares to find $\delta$.
    *   Outer Loop: Optimize $\sigma$ to minimize GMM objective ($\xi \perp Z$).
4.  **Simulation:**
    *   Change ownership $\Omega$.
    *   Solve $p = c + \Omega^{-1}s$ for new prices.
    *   Calculate change in Consumer Surplus.