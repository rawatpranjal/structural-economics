# BLP Case Study

### Phase 1: The Fake Data (Input)
Imagine a market with 3 products and an "Outside Good" (people who eat toast). We observe shares, prices, and one characteristic: **Sugar**.

**The Market Snapshot**
| Product | Brand | Price ($) | Sugar (1-10) | Market Share ($s_j$) |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Choco-Bombs** | $3.00 | 10 (High) | **40%** |
| **B** | **Fiber-Bran** | $5.00 | 1 (Low) | **20%** |
| **C** | **Store-Frosted** | $2.00 | 8 (High) | **10%** |
| **0** | **Outside Good** | -- | -- | **30%** |

**The Hidden Truth (Consumer Heterogeneity)**
The researcher doesn't see this, but the population is composed of two types:
1.  **Kids (60%):** Love sugar, don't care about price.
2.  **Parents (40%):** Hate sugar, very price sensitive.

---

### Phase 2: The Estimation (The Pipeline)

We run two models.
1.  **Logit:** Assumes the "Average Consumer."
2.  **BLP:** Tries to find the "Variance" ($\sigma$) in tastes.

#### Step 1: Inverting Shares to find Mean Utility ($\delta$)
The algorithm calculates the "Mean Utility" required to match the observed shares.

| Product | Share ($s_j$) | Outside ($s_0$) | Logit $\delta$ ($\ln s_j - \ln s_0$) | BLP $\delta$ (Adjusted for $\sigma$) |
| :--- | :--- | :--- | :--- | :--- |
| **A** | 0.40 | 0.30 | **0.28** | **0.20** |
| **B** | 0.20 | 0.30 | **-0.40** | **-0.55** |
| **C** | 0.10 | 0.30 | **-1.10** | **-1.15** |

*Note: BLP finds different mean utilities because it accounts for the fact that Kids *really* love Product A, pushing its share up.*

#### Step 2: Recovering Parameters ($\beta, \alpha, \sigma$)
We regress $\delta$ on Characteristics (Price, Sugar) using Instruments.

**The Estimated Parameters:**
| Parameter | Meaning | **Logit Estimate** (The Wrong Avg) | **BLP Estimate** (The Truth) |
| :--- | :--- | :--- | :--- |
| $\alpha$ | Price Sensitivity | **-1.5** (Moderate) | **-3.0** (Mean), $\sigma$=**2.0** (High Variance) |
| $\beta_{sugar}$ | Taste for Sugar | **0.5** (Mild Like) | **-1.0** (Mean), $\sigma$=**4.0** (Polarizing) |

*   **Logit says:** "People are moderately sensitive to price and kinda like sugar."
*   **BLP says:** "The average person hates sugar and price, BUT there is a huge variance ($\sigma$). Some people (Kids) love sugar and don't care about price."

---

### Phase 3: Derived Elasticities (The Diagnostics)
Now we calculate how consumers switch if **Choco-Bombs (A)** raises its price by 10%.

**Diversion Ratios: Where do the customers go?**

| Rival Product | **Logit Prediction** (IIA) | **BLP Prediction** (Realistic) |
| :--- | :--- | :--- |
| **To Fiber-Bran (B)** | **67%** | **5%** |
| **To Store-Frosted (C)** | **33%** | **95%** |

*   **The Logit Fail:** Logit says most people switch to Fiber-Bran just because it has a higher market share (20% vs 10%). This is absurd. People buying Choco-Bombs don't want Fiber-Bran.
*   **The BLP Win:** BLP knows that the people buying Choco-Bombs are "Sugar-Lovers / Price-Insensitive." If they leave A, they switch to the other sugary option (C), even though C is small.

---

### Phase 4: The Simulation (Merger Counterfactual)
**Scenario:** The maker of **Choco-Bombs (A)** acquires **Store-Frosted (C)**.
We solve for the new equilibrium prices ($p^*$).

**Cost Recovery (Supply Side)**
First, we back out marginal costs using the pre-merger margins.
*   Product A Price: $3.00.
*   BLP estimated Marginal Cost: **$1.80** (Markup $1.20).

**Post-Merger Price Simulation**
| Product | Pre-Merger Price | **Logit Predicted Price** | **BLP Predicted Price** |
| :--- | :--- | :--- | :--- |
| **Choco-Bombs (A)** | $3.00 | **$3.05** (+1.6%) | **$3.25** (+8.3%) |

**Why the difference?**
*   **Logit** thinks A and C are not close competitors (because C is small). It predicts a tiny price hike.
*   **BLP** identifies A and C as **close substitutes** in the characteristic space (Sugar). It knows that if A raises prices, it usually loses customers to C. Now that A owns C, it captures those lost customers. This gives the firm a massive incentive to raise the price of A.

### Summary of Case Study Results

| Metric | Simple Logit | **BLP** | Conclusion |
| :--- | :--- | :--- | :--- |
| **Who likes Sugar?** | Everyone a little bit. | Kids love it, Parents hate it. | BLP captures reality. |
| **Substitution** | Proportional to size. | Based on characteristics. | BLP fixes IIA. |
| **Merger Effect** | Harmless (+1%). | **Anti-Competitive (+8%)**. | **BLP blocked the merger.** |