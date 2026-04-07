# Logit

### 1. The Setup: Consumer Utility
We start with the utility consumer $i$ gets from product $j$ in market $t$. In the **Simple Logit**, we assume **no random coefficients** (no $\sigma$), meaning everyone has the same taste parameters.

$$U_{ijt} = \underbrace{x_{jt}\beta - \alpha p_{jt} + \xi_{jt}}_{\delta_{jt} \text{ (Mean Utility)}} + \epsilon_{ijt}$$

*   $\delta_{jt}$: The **Mean Utility**. This is the value everyone agrees on.
*   $\epsilon_{ijt}$: The **Idiosyncratic Error**. Distributed **i.i.d. Type I Extreme Value (Gumbel)**. This specific distribution assumption is what allows us to derive the closed-form "Logit" algebra below.

---

### 2. Deriving Market Shares ($s_{jt}$)
A consumer chooses product $j$ if $U_{ijt} > U_{ikt}$ for all other products $k$. Because of the Gumbel distribution of $\epsilon$, the integral over the population collapses into this famous closed-form equation:

$$s_{jt} = \frac{\exp(\delta_{jt})}{1 + \sum_{k=1}^{J} \exp(\delta_{kt})}$$

*   *Note:* The "1" in the denominator represents the **Outside Good** ($j=0$), where we normalize $\delta_{0t} = 0$, so $\exp(0) = 1$.

---

### 3. The Inversion (The "Regression" Equation)
This is the most critical step for estimation. We cannot estimate the equation in Step 2 directly because $\xi$ (unobserved quality) is inside the non-linear $\exp(\cdot)$ function. We must "linearize" it to run a regression.

1.  Take the log of the share of product $j$:
    $$\ln(s_{jt}) = \delta_{jt} - \ln\left(1 + \sum \exp(\delta_{kt})\right)$$
2.  Take the log of the share of the outside good ($j=0$, where $\delta_{0t}=0$):
    $$\ln(s_{0t}) = 0 - \ln\left(1 + \sum \exp(\delta_{kt})\right)$$
3.  Subtract the two equations:
    $$\ln(s_{jt}) - \ln(s_{0t}) = \delta_{jt}$$

**The Estimation Equation:**
Substituting the definition of $\delta_{jt}$ back in, we get a linear equation we can estimate using IV Regression (2SLS):

$$\ln(s_{jt}) - \ln(s_{0t}) = x_{jt}\beta - \alpha p_{jt} + \xi_{jt}$$

*   **Y-variable:** $\ln(s_{jt}) - \ln(s_{0t})$ (Data)
*   **X-variables:** Characteristics $x_{jt}$ and Price $p_{jt}$ (Data)
*   **Error term:** $\xi_{jt}$ (Unobserved Quality) -> **This is why we need Instruments for Price.**

---

### 4. Deriving Elasticities (The "IIA" Problem)
Practitioners care about elasticities ($\eta$). Let's derive the own-price and cross-price elasticities from the share equation in Step 2.

The derivative of share w.r.t. price is:
*   $\frac{\partial s_{jt}}{\partial p_{jt}} = -\alpha s_{jt}(1 - s_{jt})$
*   $\frac{\partial s_{jt}}{\partial p_{kt}} = \alpha s_{jt} s_{kt}$

**Elasticity Formulas:**
$$\eta_{jkt} = \frac{\partial s_{jt}}{\partial p_{kt}} \frac{p_{kt}}{s_{jt}}$$

1.  **Own-Price Elasticity:**
    $$\eta_{jj} = -\alpha p_{jt} (1 - s_{jt})$$
    *Implication:* Elasticity is driven mostly by price ($p_{jt}$). If shares are small, $(1-s_{jt}) \approx 1$, so elasticity is proportional to price.

2.  **Cross-Price Elasticity:**
    $$\eta_{jk} = \alpha p_{kt} s_{kt}$$
    *The "Gotcha" (IIA):* Notice that the cross-elasticity of $j$ with respect to $k$ depends **only on the market share of $k$**.
    *   *Translation:* If the price of a BMW goes up, customers switch to a Yugo and a Mercedes in exact proportion to their market shares. The model doesn't know BMW and Mercedes are "similar." **This is the flaw BLP fixes.**

---

### 5. Deriving Supply Side (Markups)
Firms maximize profit. We assume single-product firms for simplicity here (Bertrand-Nash competition).

$$\pi_j = (p_j - c_j) M s_j$$

Take the First Order Condition (FOC) with respect to price $p_j$:
$$s_j + (p_j - c_j)\frac{\partial s_j}{\partial p_j} = 0$$

Rearrange to solve for the **Markup** ($p-c$):
$$p_j - c_j = -\frac{s_j}{\partial s_j / \partial p_j}$$

Substitute the derivative we found in Step 4 ($-\alpha s_j (1-s_j)$):
$$p_j - c_j = -\frac{s_j}{-\alpha s_j (1-s_j)} = \frac{1}{\alpha(1-s_j)}$$

**Practitioner Insight:**
*   We observe Prices ($p$).
*   We estimate $\alpha$ (Price Sensitivity).
*   We observe Shares ($s$).
*   Therefore, we can calculate **Marginal Cost ($c$)** without ever seeing the firm's accounting books.