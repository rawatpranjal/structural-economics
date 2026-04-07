# Estimation

### The High-Level Strategy (GMM)
We use **Generalized Method of Moments (GMM)**.
**The Goal:** Find the parameters $(\theta_1, \theta_2)$ that make the model's error terms ($\xi$) uncorrelated with the instrumental variables ($Z$).

**The Moment Condition:**
$$E[\xi_{jt}(\theta) \cdot Z_{jt}] = 0$$

Since we have linear parameters ($\alpha, \beta$) and non-linear parameters ($\sigma$), we split the estimation into two loops.

---

### 1. The Outer Loop (Search for $\sigma$)
The computer uses an optimization algorithm (like BFGS or Newton-Raphson) to search **only** for the non-linear parameters ($\sigma$, the standard deviations of taste).

1.  **Guess:** The optimizer picks a starting value for $\sigma$ (e.g., "Let's guess the standard deviation of price sensitivity is 2").
2.  **Pass:** Send this guess to the Inner Loop.

---

### 2. The Inner Loop (The Contraction Mapping)
*This is the most famous part of the BLP paper.*

For the specific guess of $\sigma$ (from the Outer Loop), we must find the vector of Mean Utilities ($\delta$) that makes the **Predicted Market Shares** exactly match the **Observed Market Shares**.

Since we cannot solve this algebraically, we use an iterative algorithm called a **Contraction Mapping**:

$$\delta^{h+1}_{jt} = \delta^{h}_{jt} + \ln(S_{jt}^{\text{observed}}) - \ln(s_{jt}(\delta^h, \sigma))$$

1.  **Simulate Shares:** Using the guess for $\sigma$ and the current $\delta$, we simulate the choices of 1,000 fake consumers and calculate predicted shares $s(\delta)$.
2.  **Update:** If predicted shares < observed shares, we bump up $\delta$.
3.  **Iterate:** Repeat until the shares match perfectly (convergence).

**Result:** We now have the unique vector $\delta_{jt}(\sigma)$ that corresponds to our guess of $\sigma$.

---

### 3. The Linear Step (Concentrating Out Parameters)
Now that we have $\delta_{jt}$ from the Inner Loop, we can recover the linear parameters ($\beta, \alpha$) using standard IV regression logic.

Recall the definition of Mean Utility:
$$\delta_{jt} = x_{jt}\beta - \alpha p_{jt} + \xi_{jt}$$

Since we know $\delta$, $x$, and $p$, this is just a linear equation. We calculate the error term $\xi$ for any candidate $\beta, \alpha$:
$$\xi_{jt}(\theta) = \delta_{jt}(\sigma) - (x_{jt}\beta - \alpha p_{jt})$$

*Note: In practice, we "concentrate out" $\beta$ and $\alpha$ by projecting $\delta$ onto the instruments $Z$, so the optimizer doesn't have to search for them manually. This saves massive amounts of computation time.*

---

### 4. The Objective Function (Minimizing the GMM)
We now have the structural error term $\xi(\theta)$. We calculate the **GMM Objective Function**:

$$J(\theta) = \xi(\theta)' Z W Z' \xi(\theta)$$

*   **Z:** Instrumental Variables.
*   **W:** Weighting Matrix.

The optimizer (Outer Loop) checks this value.
*   If it is close to 0, we are done.
*   If it is large (meaning $\xi$ is still correlated with $Z$), the optimizer picks a new $\sigma$ and restarts the process at Step 1.

---

### 5. Adding the Supply Side (Optional but Recommended)
If we are jointly estimating Supply, we add a second equation to the GMM objective.

1.  Using the demand estimates ($\alpha, \sigma, \delta$), calculate the derivatives and the matrix $\Omega$.
2.  Back out Marginal Costs: $c_{jt} = p_{jt} - \Omega^{-1}s_{jt}$.
3.  Specify a cost function: $\ln(c_{jt}) = w_{jt}\gamma + \omega_{jt}$.
4.  Calculate the cost-side error: $\omega_{jt} = \ln(c_{jt}) - w_{jt}\gamma$.

The GMM objective now minimizes the correlation of **both** errors ($\xi$ and $\omega$) with the instruments.

---

### Summary of the Algorithm Flow
1.  **Outer Loop:** Guess $\sigma$ (Non-linear taste variation).
2.  **Inner Loop:** Invert shares to find $\delta$ (Mean Utility) using Contraction Mapping.
3.  **Linear Step:** Regress $\delta$ on $X$ and $P$ (using IVs) to get $\beta, \alpha$ and the error $\xi$.
4.  **Supply Step:** Back out costs $c$, regress on cost shifters $w$ to get $\gamma$ and error $\omega$.
5.  **GMM Check:** Are errors $\xi, \omega$ orthogonal to Instruments $Z$?
    *   No? Update $\sigma$ and repeat.
    *   Yes? Stop. You have your parameters.