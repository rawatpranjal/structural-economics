# Nested Logit

### 1. The Setup: Utility with Nesting
We assume products are divided into mutually exclusive groups (nests) $g = 0, 1, \dots, G$. (Nest 0 is the outside good).

The utility of consumer $i$ for product $j$ in nest $g$ is:

$$U_{ijt} = \delta_{jt} + \zeta_{igt} + (1-\sigma)\epsilon_{ijt}$$

*   **$\delta_{jt}$ (Mean Utility):** Same as before ($x\beta - \alpha p + \xi$).
*   **$\zeta_{igt}$ (Nest-Specific Shock):** A random shock common to *all* products in nest $g$. (If I like trucks, I like *all* trucks).
*   **$\epsilon_{ijt}$:** Idiosyncratic shock (Standard Gumbel).
*   **$\sigma$ (Nesting Parameter):** Measures correlation within a nest.
    *   $0 \le \sigma < 1$.
    *   If $\sigma = 0$: No correlation (collapses to Simple Logit).
    *   If $\sigma \to 1$: Products in the nest are perfect substitutes.

---

### 2. Deriving Market Shares (The Two-Step Logic)
It is mathematically easier to define the market share of product $j$ as the product of two probabilities:
$$s_{jt} = s_{j|g,t} \times s_{gt}$$

**A. The Within-Nest Share ($s_{j|g}$):**
The probability of choosing $j$ *given* that you have already decided to buy from nest $g$. This looks like a standard logit, but the utility is scaled by $(1-\sigma)$.

$$s_{j|g,t} = \frac{\exp\left( \frac{\delta_{jt}}{1-\sigma} \right)}{D_{gt}}$$
*Where $D_{gt}$ is the "Inclusive Value" (sum of utilities in the nest):*
$$D_{gt} = \sum_{k \in g} \exp\left( \frac{\delta_{kt}}{1-\sigma} \right)$$

**B. The Nest Share ($s_{g}$):**
The probability of choosing nest $g$ over other nests.
$$s_{gt} = \frac{D_{gt}^{1-\sigma}}{\sum_{h} D_{ht}^{1-\sigma}}$$

**C. Total Share Equation:**
$$s_{jt} = \frac{\exp\left( \frac{\delta_{jt}}{1-\sigma} \right)}{D_{gt}} \times \frac{D_{gt}^{1-\sigma}}{\sum_{h} D_{ht}^{1-\sigma}}$$

---

### 3. The Inversion (Berry 1994)
How do we get from that messy share equation to a linear regression? This is the famous Berry (1994) inversion.

**Step 1:** Take the log of the share equation for product $j$:
$$\ln(s_{jt}) = \frac{\delta_{jt}}{1-\sigma} - \ln(D_{gt}) + (1-\sigma)\ln(D_{gt}) - \ln\left(\sum D^{1-\sigma}\right)$$
Simplify the $D_{gt}$ terms ($-\ln(D) + (1-\sigma)\ln(D) = -\sigma\ln(D)$):
$$\ln(s_{jt}) = \frac{\delta_{jt}}{1-\sigma} - \sigma\ln(D_{gt}) - \ln\left(\sum D\right)$$

**Step 2:** Take the log of the Outside Good ($j=0$):
$$\ln(s_{0t}) = 0 - 0 - \ln\left(\sum D\right)$$

**Step 3:** Subtract them:
$$\ln(s_{jt}) - \ln(s_{0t}) = \frac{\delta_{jt}}{1-\sigma} - \sigma\ln(D_{gt})$$

**Step 4:** The "Magic" Substitution.
Recall the formula for the within-nest share $s_{j|g}$ (from Section 2A). Taking the log of that equation gives us:
$$\ln(s_{j|g}) = \frac{\delta_{jt}}{1-\sigma} - \ln(D_{gt}) \implies \ln(D_{gt}) = \frac{\delta_{jt}}{1-\sigma} - \ln(s_{j|g})$$

Substitute this back into Step 3. The math cancels out beautifully to leave:
$$\ln(s_{jt}) - \ln(s_{0t}) = \delta_{jt} + \sigma \ln(s_{j|g,t})$$

**The Final Estimation Equation:**
Expand $\delta_{jt}$ to get the linear regression:

$$\ln(s_{jt}) - \ln(s_{0t}) = x_{jt}\beta - \alpha p_{jt} + \sigma \ln(s_{j|g,t}) + \xi_{jt}$$

---

### 4. Identification (The "New" Endogeneity)
In Simple Logit, we only had to instrument for Price ($p_{jt}$).
In Nested Logit, we have a new endogenous variable on the Right-Hand Side: **$\ln(s_{j|g,t})$**.

*   **Why is it endogenous?** Because the within-group share contains $\xi_{jt}$. A product with high unobserved quality will have a high within-group share.
*   **The Solution (New Instruments):** You need instruments correlated with *within-group* share but not $\xi_{jt}$.
    *   *Standard IV:* The **Number of products in the nest**.
    *   *BLP IV:* The sum of characteristics of **other** products in the **same** nest. (If the nest is crowded with high-quality rivals, my within-group share goes down).

---

### 5. Elasticities (Solving the IIA Problem)
Nested Logit gives us two different cross-price elasticities, which breaks the strict IIA assumption.

1.  **Own-Price Elasticity:**
    $$\eta_{jj} = -\alpha p_{jt} \left[ \frac{1}{1-\sigma} - \left( \frac{1}{1-\sigma} - 1 \right)s_{j|g} - s_{jt} \right]$$
    *Intuition:* As $\sigma \to 1$, the $1/(1-\sigma)$ term explodes. This means if products in a nest are perfect substitutes, a tiny price increase causes you to lose *all* sales (elasticity $\to \infty$).

2.  **Cross-Price (Same Nest):**
    $$\eta_{jk} = \alpha p_{kt} \left[ \left( \frac{1}{1-\sigma} - 1 \right)s_{k|g} + s_{kt} \right]$$
    *Intuition:* High substitution. If Ford raises price, Chevy gains a lot of share.

3.  **Cross-Price (Different Nest):**
    $$\eta_{jk} = \alpha p_{kt} s_{kt}$$
    *Intuition:* Low substitution. If Ford raises price, the Yugo (different nest) only gains share proportional to its total market size.

### Summary Checklist
*   **Assumptions:** Consumers substitute more within groups than across groups.
*   **Regression:** We add $\ln(s_{j|g})$ (Within-Group Share) as a regressor.
*   **Parameters:** We estimate $\beta, \alpha$, and the nesting parameter $\sigma$.
*   **Identification:** Requires instruments for both Price AND Within-Group Share.