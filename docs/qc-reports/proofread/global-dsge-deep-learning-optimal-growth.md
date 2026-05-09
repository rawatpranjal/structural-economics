# Proofread: global-dsge/deep-learning-optimal-growth/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T07:15:00Z._

## Paper / Source Verification

### Brock, W. A., and Mirman, L. J. (1972). *Optimal Economic Growth and Uncertainty: The Discounted Case*. Journal of Economic Theory.

- **Located:** https://ideas.repec.org/a/eee/jetheo/v4y1972i3p479-513.html
- **Tutorial claims:** The Brock-Mirman (1972) paper provides the canonical stochastic optimal growth model with the closed-form log-linear policy used to audit the neural solution.
- **Source says:** The published title is "Optimal economic growth and uncertainty: The discounted case" (sentence case); authors William A. Brock and Leonard J. Mirman; Journal of Economic Theory, vol. 4, 1972. The paper establishes conditions for a stable optimal path under discounted uncertainty.
- **Verdict:** OK
- **Note:** Title-case rendering in the tutorial differs cosmetically from the published sentence case; both forms appear throughout the citations literature.

---

### Maliar, L., Maliar, S., and Winant, P. (2022). *Deep Learning for Solving Dynamic Economic Models*. Lecture slides.

- **Located:** https://ideas.repec.org/a/eee/moneco/v122y2021icp76-101.html
- **Tutorial claims:** A 2022 work presented as lecture slides that covers deep learning methods for dynamic economic models.
- **Source says:** The paper "Deep Learning for Solving Dynamic Economic Models" by Lilia Maliar, Serguei Maliar, and Pablo Winant was published in the *Journal of Monetary Economics*, vol. 122, pp. 76-101, **2021** (DOI: 10.1016/j.jmoneco.2021.07.004). It is a peer-reviewed journal article, not lecture slides. The paper casts Bellman equations, Euler equations, and lifetime reward objectives as nonlinear regression problems solved with deep neural networks and stochastic gradient descent.
- **Verdict:** MAJOR
- **Note:** The correct citation is: Maliar, L., Maliar, S., and Winant, P. (2021). *Deep Learning for Solving Dynamic Economic Models*. Journal of Monetary Economics, 122, 76-101. The year should be 2021 and the venue should be Journal of Monetary Economics, not lecture slides.

---

## Main Message Audit

> Deep learning is not needed to solve this Brock-Mirman model. The example is useful because the answer is known. The same residual-minimization machinery appears in larger nonlinear macro models. Feasibility comes from the policy parameterization. Training comes from Euler residuals on simulated states. Credibility comes from the out-of-sample closed-form audit.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Deep learning is not needed to solve this Brock-Mirman model | Equations (closed-form rule given); Results (exact audit shown) | OK |
| The example is useful because the answer is known | Results (policy error table and figures compare neural to exact) | OK |
| The same residual-minimization machinery appears in larger nonlinear macro models | Not demonstrated - only Brock-Mirman is solved | OVERREACH |
| Feasibility comes from the policy parameterization | Equations (saving share bounded; $k'$ and $c$ defined by construction) | OK |
| Training comes from Euler residuals on simulated states | Solution Method (algorithm draws uniform states and minimizes $\Xi_n(\theta)$) | OK |
| Credibility comes from the out-of-sample closed-form audit | Results (holdout grid, policy error table, Euler residual plot) | OK |

Issues:
- "The same residual-minimization machinery appears in larger nonlinear macro models" is stated in the Takeaway but the tutorial presents only the Brock-Mirman case; no larger model is solved or cited to demonstrate portability of the machinery.

---

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k_t$ | Equations: "With state $k_t$, output is" | Yes, at first use | State variable; OK |
| $y_t$ | Equations: $y_t = A k_t^{\alpha}$ | Yes, at first use | Output; OK |
| $A$ | Equations: $y_t = A k_t^{\alpha}$ | Partial - named in Model Setup table within 50 lines | Acceptable |
| $\alpha$ | Equations: $y_t = A k_t^{\alpha}$, $0<\alpha<1$ | Yes, range given at first use | OK |
| $c_t$ | Equations: feasibility constraint | Yes, implicitly as consumption | OK |
| $k_{t+1}$ | Equations: feasibility constraint | Yes, implicitly as next-period capital | OK |
| $\beta$ | Equations: $\sum \beta^t \log c_t$, $0<\beta<1$ | Yes, range given at first use | OK |
| $c_{t+1}$ | Equations: Euler equation denominator | Yes, by analogy with $c_t$ at time $t+1$ | OK |
| $\theta$ | Equations: $r(k;\theta)$ | Implicit - associated with neural network parameters via context | Acceptable; formally tied to $N_\theta$ in saving share formula |
| $r(k;\theta)$ | Equations: "the code evaluates the Euler equation as this log residual" | Yes, at first use | OK |
| $k'(k;\theta)$ | Equations: inside $r(k;\theta)$ formula | Late - defined after saving share as $k'(k;\theta)=s(k;\theta)A k^{\alpha}$ | Flagged - late definition |
| $c(k;\theta)$ | Equations: inside $r(k;\theta)$ formula | Late - defined after saving share as $c(k;\theta)=(1-s(k;\theta))A k^{\alpha}$ | Flagged - late definition |
| $\Xi(\theta)$ | Equations: "The population risk is" | Yes, at first use | OK |
| $k_i, \ldots, k_n$ | Equations: "With draws $k_1,\ldots,k_n$" | Yes, at first use | OK |
| $n$ | Equations: $\Xi_n(\theta) = \frac{1}{n}\sum_{i=1}^{n}$ | Yes, at first use | OK |
| $\Xi_n(\theta)$ | Equations: empirical problem | Yes, at first use | OK |
| $\hat{\theta}$ | Equations: $\hat{\theta} = \arg\min_\theta \Xi_n(\theta)$ | Yes, at first use | OK |
| $s(k;\theta)$ | Equations: "The neural policy first chooses a saving share" | Yes, at first use | OK |
| $s_{\min}, s_{\max}$ | Equations: saving share formula | Yes, at first use; also in Model Setup | OK |
| $\sigma$ | Equations: saving share formula $\sigma(N_\theta(\cdot))$ | Late - defined only in Solution Method pseudocode as sigmoid | Flagged - late definition |
| $N_\theta$ | Equations: saving share formula $\sigma(N_\theta(\log(k/k_{ss})))$ | Absent in prose - pseudocode uses $N\_\text{theta}$; Model Setup describes a "1-16-16-1 tanh MLP" but does not name it $N_\theta$ | Flagged - no prose definition |
| $k_{ss}$ | Equations: saving share formula $\log(k/k_{ss})$ | Late - formally defined after the exact policy as $k_{ss}=(\alpha\beta A)^{1/(1-\alpha)}$ | Flagged - late definition |
| $k'(k)$ | Equations: exact policy $k'(k)=\alpha\beta A k^\alpha$ | Yes, at first use (exact version distinguished from neural $k'(k;\theta)$) | OK |
| $c(k)$ | Equations: exact policy $c(k)=(1-\alpha\beta)A k^\alpha$ | Yes, at first use (exact version) | OK |
| $c_{ss}$ | Model Setup table | Partial - appears only in table row "Closed-form steady-state consumption" | Acceptable |

Flagged issues:
- $k'(k;\theta)$ and $c(k;\theta)$ are both used inside the log residual $r(k;\theta)$ formula before their definitions appear several lines later in the feasibility-by-construction equations.
- $k_{ss}$ appears in the saving share formula before being formally defined in the steady state equation at the end of the Equations section.
- $N_\theta$ (the neural network function) is used in the saving share formula without any prose definition; the Model Setup table describes the architecture but never introduces the symbol $N_\theta$ by name.
- $\sigma$ is used in the saving share formula without definition; the definition appears later in the Solution Method pseudocode.

## Summary

The tutorial is clear and internally consistent. There is 1 MAJOR reference error: Maliar, Maliar, and Winant is cited as 2022 lecture slides when it is a 2021 Journal of Monetary Economics article (vol. 122, pp. 76-101). There are 4 notation issues where symbols are used before they are defined ($k'(k;\theta)$, $c(k;\theta)$, $k_{ss}$, $\sigma$) and 1 symbol ($N_\theta$) that has no prose definition anywhere in the README. There is 1 OVERREACH in the Takeaway where portability to larger macro models is asserted but not demonstrated. The single most important fix is correcting the Maliar et al. reference year to 2021 and venue to Journal of Monetary Economics.
