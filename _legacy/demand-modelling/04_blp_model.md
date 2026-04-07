# BLP Model

### 1. The Utility Function (Decomposed)
The core innovation of BLP is splitting utility into a **Linear** part (common to everyone) and a **Non-Linear** part (specific to the individual).

$$U_{ijt} = \underbrace{\delta_{jt}}_{\text{Mean Utility}} + \underbrace{\mu_{ijt}}_{\text{Heterogeneity}} + \epsilon_{ijt}$$

#### A. Mean Utility ($\delta_{jt}$)
This is the standard Logit part. It captures the average valuation of the product.
$$\delta_{jt} = x_{jt}\beta - \alpha p_{jt} + \xi_{jt}$$
*   **$\beta, \alpha$:** The *average* tastes in the population.
*   **$\xi_{jt}$:** Unobserved Quality (The structural error term).

#### B. Heterogeneity ($\mu_{ijt}$)
This captures how individual $i$ deviates from the average.
$$\mu_{ijt} = \sum_{k} x_{jt}^k (\sigma_k v_{ik} + \pi_{k} d_{ik})$$

This deviation is driven by two factors:
1.  **Random Shocks ($v_{ik}$):** Unobserved preferences. (e.g., Consumer $i$ just happens to love sugar). We estimate the standard deviation ($\sigma$) of these shocks.
2.  **Demographics ($d_{ik}$):** Observed data. (e.g., High-income consumers are less price-sensitive). We estimate the interaction parameter ($\pi$).

**Example (Car Market):**
If $x$ is "Horsepower" and $d$ is "Age":
*   $\beta$: Average preference for horsepower.
*   $\sigma$: How much preferences for horsepower vary randomly.
*   $\pi$: How much *Age* explains preference for horsepower (e.g., young people like fast cars).

---

### 2. Market Shares (The Integral)
Because every consumer has different parameters ($\beta_i, \alpha_i$), we cannot write a simple fraction for market share like we did in Simple/Nested Logit.

Instead, the market share of product $j$ is the **sum** (integral) of individual choices across the population distribution ($P$):

$$s_{jt}(\delta, \theta) = \int \underbrace{\frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k} \exp(\delta_{kt} + \mu_{ikt})}}_{\text{Individual Prob (Logit)}} dP(v, d)$$

**Practitioner Translation:**
We cannot solve this integral analytically. We must **Simulate** it.
1.  We draw $ns$ (e.g., 1000) "fake individuals" from the demographic distribution.
2.  We calculate the probability that *each* fake person buys product $j$.
3.  We average those probabilities to get the aggregate market share.

---

### 3. The Supply Side (Multi-Product Firms)
We assume firms compete on price (Bertrand-Nash) and manage a portfolio of products $\mathcal{F}_f$.

**Profit Function:**
$$\Pi_f = \sum_{j \in \mathcal{F}_f} (p_j - c_j) M s_j(p)$$

**First Order Condition (The Pricing Equation):**
$$s_{jt} + \sum_{k \in \mathcal{F}_f} (p_{kt} - c_{kt}) \frac{\partial s_{kt}}{\partial p_{jt}} = 0$$

In vector notation, solving for Marginal Costs ($c$):
$$c = p - \Omega^{-1} s(p)$$

*   **$p$:** Observed Prices vector.
*   **$s$:** Observed Shares vector.
*   **$\Omega$:** The Ownership-Derivative matrix. (Crucially, the derivatives inside $\Omega$ now come from the complex random coefficients integral, not the simple Logit formula).

---

### Summary of Parameters to Estimate ($\theta$)
Unlike Simple Logit where we just ran a regression for $\beta$, here we split parameters into two groups:

1.  **Linear Parameters ($\theta_1$):** $\alpha, \beta, \gamma$.
    *   These enter the Mean Utility ($\delta$) linearly.
    *   Estimated via IV Regression (GMM) in the outer loop.
2.  **Non-Linear Parameters ($\theta_2$):** $\sigma$ (Random Coefficients), $\pi$ (Demographic Interactions).
    *   These enter the Heterogeneity ($\mu$) non-linearly.
    *   Estimated via Non-Linear Optimization (e.g., BFGS) by minimizing the GMM objective function.