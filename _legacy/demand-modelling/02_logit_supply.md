# Logit with Multi-Product Supply Side

### 1. Utility & Shares (Verified)
**Setup:** $U_{ijt} = \delta_{jt} + \epsilon_{ijt}$ where $\delta_{jt} = x_{jt}\beta - \alpha p_{jt} + \xi_{jt}$.
**Shares:** $s_{jt} = \frac{\exp(\delta_{jt})}{1 + \sum_{k=1}^{J} \exp(\delta_{kt})}$.

### 2. The Inversion (Verified)
**Logic:** We linearize the non-linear share equation to run a regression.
**Result:** $\ln(s_{jt}) - \ln(s_{0t}) = x_{jt}\beta - \alpha p_{jt} + \xi_{jt}$.
*   *Note:* This equation allows us to estimate $\alpha$ and $\beta$ using IV regression (2SLS).

### 3. Elasticities (Verified)
**Assumption:** Utility decreases with price ($-\alpha p$).
**Own-Price:** $\eta_{jj} = -\alpha p_{jt} (1 - s_{jt})$.
**Cross-Price:** $\eta_{jk} = \alpha p_{kt} s_{kt}$.
*   *Insight:* Cross-elasticity depends only on the rival's market share ($s_{kt}$) and price, not on how similar product $j$ is to product $k$.

---

### 4. The Supply Side (Expanded for Multi-Product Firms)
*This is the advanced version you need for the interview.*

In the previous response, we assumed firms sell only one product. In reality, firms sell a portfolio of products $\mathcal{F}_f$.

**The Profit Function:**
Firm $f$ maximizes the sum of profits from **all** its products:
$$\Pi_f = \sum_{j \in \mathcal{F}_f} (p_j - c_j) M s_j(p)$$

**The First Order Condition (FOC):**
When firm $f$ sets the price $p_j$, it considers how that price affects the sales of product $j$ **AND** how it cannibalizes or helps its other products $k \in \mathcal{F}_f$.

Differentiating with respect to $p_j$:
$$s_j + \sum_{k \in \mathcal{F}_f} (p_k - c_k) \frac{\partial s_k}{\partial p_j} = 0$$

**The Matrix Notation (The "BLP Formula"):**
We can't solve this algebraically like the single-product case. We stack these equations into matrices.

Define $\Omega$ (Omega) as the **Ownership-Derivative Matrix**:
$$ \Omega_{jk} = \begin{cases} -\frac{\partial s_k}{\partial p_j}, & \text{if products } j \text{ and } k \text{ are made by the same firm} \\ 0, & \text{otherwise} \end{cases} $$

The FOC in vector form becomes:
$$s - \Omega(p - c) = 0$$

**Solving for Markups:**
We can now solve for the markup vector $(p-c)$ by inverting the $\Omega$ matrix:
$$p - c = \Omega^{-1} s$$

And finally, recover Marginal Costs:
$$c = p - \Omega^{-1} s$$

### Summary of Corrections/Additions
1.  **Single vs. Multi-Product:** The previous markup derivation ($p-c = \frac{1}{\alpha(1-s)}$) is correct **only** if the firm sells one product.
2.  **General Case:** The matrix formula $c = p - \Omega^{-1} s$ is the general practitioner's answer used in BLP code (like PyBLP).

The rest of the math (Inversion, Elasticity formulas) is error-free.