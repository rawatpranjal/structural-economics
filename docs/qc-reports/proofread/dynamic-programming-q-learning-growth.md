# Proofread: dynamic-programming/q-learning-growth/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T23:05:00Z._

## Paper / Source Verification

### Brock, W. A. and Mirman, L. J. (1972). Optimal Economic Growth and Uncertainty: The Discounted Case. *Journal of Economic Theory*, 4(3), 479-513.

- **Located:** https://doi.org/10.1016/0022-0531(72)90135-4
- **Tutorial claims:** Foundational stochastic growth model with log utility, Cobb-Douglas production, and full depreciation that yields the closed-form saving rule $k'(k,z) = \alpha\beta z A k^\alpha$.
- **Source says:** Paper characterizes the optimal policy for one-sector stochastic growth with discounted utility; the Brock-Mirman model is the canonical source for this closed-form result.
- **Verdict:** OK
- **Note:** Authors, journal, volume, issue, and page range all confirmed via IDEAS/RePEC and Elsevier.

### Watkins, C. J. C. H. and Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3), 279-292.

- **Located:** https://doi.org/10.1007/BF00992698
- **Tutorial claims:** Source of the Q-learning TD update rule $Q(s,a) \leftarrow Q(s,a) + \alpha_t[r + \beta\max_{a'}Q(s',a') - Q(s,a)]$.
- **Source says:** Paper introduces Q-learning and proves convergence; the update rule is confirmed; the issue is a combined 3–4 volume.
- **Verdict:** MINOR
- **Note:** The journal issue is 3–4 (combined); the README cites only issue 3.

### Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.

- **Located:** http://incompleteideas.net/book/the-book-2nd.html
- **Tutorial claims:** Standard RL reference for Q-learning and temporal-difference methods.
- **Source says:** 2nd edition confirmed, MIT Press, 2018; authors confirmed.
- **Verdict:** OK
- **Note:** All bibliographic details are correct.

### Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-Level Control through Deep Reinforcement Learning. *Nature*, 518, 529-533.

- **Located:** https://doi.org/10.1038/nature14236
- **Tutorial claims:** Introduces DQN with experience replay buffer and slow-moving target network; loss is a Huber penalty.
- **Source says:** Paper confirms DQN with replay buffer, target network, and Huber (clipped error) loss; pages 529–533 confirmed.
- **Verdict:** OK
- **Note:** Author list, volume, page range, and key architectural claims all verified.

## Main Message Audit

> "Q-learning recovers the planner's saving rule from sampled transitions and matches the closed-form Brock-Mirman policy. When the transition is unknown, the planner can still recover the saving rule. Sampled transitions are enough. Q-learning trades a model for data. The closed-form Brock-Mirman policy keeps both the model-based and the model-free solvers honest."

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Q-learning recovers the planner's saving rule from sampled transitions | Results (policy MAE 0.0154 vs closed form; policy-comparison figure) | OK |
| matches the closed-form Brock-Mirman policy | Results (policy-comparison figure; algorithm table) | OK |
| When the transition is unknown, the planner can still recover the saving rule | Solution Method (no transition matrix used by Q-learning); Results | OK |
| Sampled transitions are enough | Results (policy MAE 0.0154 at 6,000,000 steps) | OK |
| Q-learning trades a model for data | Solution Method (explicit contrast: VFI uses matrix, Q-learning uses transitions) | OK |
| The closed-form Brock-Mirman policy keeps both solvers honest | Results (both solvers benchmarked against closed form in table and figures) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k_t$ | Equations | Yes — "capital" | ✓ |
| $z_t$ | Equations | Yes — "productivity shock" | ✓ |
| $y_t$ | Equations | Yes — "Output is $y_t = z_t A k_t^\alpha$" | ✓ |
| $A$ | Equations | **No** | Never named or defined anywhere in the README; absent from the Model Setup table |
| $c_t$ | Equations | Yes — implicit from resource constraint $c_t + k_{t+1} = y_t$ | ✓ |
| $k_{t+1}$ | Equations | Yes — next-period capital, from constraint | ✓ |
| $\alpha$ | Equations | Acceptable — Model Setup table within 50 lines: "Capital share α \| 0.36" | ✓ |
| $\rho$ | Equations | Acceptable — Model Setup table within 50 lines: "Productivity persistence ρ \| 0.70" | ✓ |
| $\sigma$ | Equations | Acceptable — Model Setup table within 50 lines: "Innovation std σ \| 0.10" | ✓ |
| $\varepsilon_{t+1}$ | Equations | Yes — "$\varepsilon_{t+1} \sim N(0,1)$" immediately | ✓ |
| $V(k,z)$ | Equations | Yes — "value function" from surrounding prose | ✓ |
| $k'$ | Equations | Yes — from $\max_{k'\in[0,y]}$ context | ✓ |
| $z'$ | Equations | Implicit — next-period productivity; standard prime notation | ✓ |
| $\beta$ | Equations | Acceptable — Model Setup table within 50 lines: "Discount β \| 0.95" | ✓ |
| $\mathbb{E}$ | Equations | Yes — expectation, standard | ✓ |
| $Q(s,a)$ | Equations | Yes — "action-value $Q(s,a)$ for each state-action pair" | ✓ |
| $s$, $a$ | Equations | Yes — "state-action pair" in same sentence | ✓ |
| $\alpha_t$ | Equations (Q-learning update) | **No** | Step size/learning rate; never labeled as such in prose; overloads $\alpha$ (capital share = 0.36) |
| $r$ | Equations (Q-learning update) | Acceptable — pseudocode within ~30 lines: "receive reward $r_t = \log(\ldots)$" | ✓ |
| $s'$, $a'$ | Equations (Q-learning update) | Yes — next-state, next-action; standard prime convention | ✓ |
| $a^{\ast}(s)$ | Equations | Yes — "greedy policy … $a^{\ast}(s) = \arg\max_a Q(s,a)$" | ✓ |
| $n_{s,a}$ | Solution Method | Partial — "decays with visit counts" in the same sentence | ✓ |
| $Q_\theta$ | Solution Method | Yes — "two-layer MLP $Q_\theta(k,z,\cdot)$"; $\theta$ as network parameters confirmed in DQN pseudocode | ✓ |
| $k_{ss}$ | Model Setup | Yes — "Steady-state capital $k_{ss}$" in table | ✓ |

Flagged issues:
- **$A$ is undefined.** The TFP parameter $A$ appears in the Equations section ($y_t = z_t A k_t^\alpha$) and in the benchmark formula ($k'(k,z) = \alpha\beta z A k^\alpha$), but the Model Setup table contains no row for it. Every other parameter (α, β, ρ, σ) has a table entry; $A$ does not.
- **$\alpha_t$ is undefined and overloads $\alpha$.** The Q-learning update uses $\alpha_t$ as the step size, but the README never labels it "learning rate" or "step size" in the Equations section. Because $\alpha$ is already established as the capital share (= 0.36), a reader encountering $\alpha_t$ in the same section can reasonably misread it as a time-varying capital share. The Solution Method section refers to "A Robbins-Monro step size $1/n_{s,a}^{0.6}$" but does not connect that expression to $\alpha_t$.

## Summary

The tutorial is well-supported and internally consistent. All four cited references are locatable; bibliographic details are correct or differ only cosmetically (Watkins & Dayan 1992 is a combined issue 3–4, not solely issue 3). The main message is fully supported by the Equations, Solution Method, and Results sections with no overreach. There are two notation issues: (1) the TFP parameter $A$ is used throughout the Equations section but has no definition or table entry anywhere in the README; (2) $\alpha_t$ is used as the Q-learning step size without being labeled as such and overloads the already-defined capital-share symbol $\alpha$. The most important fix is adding a definition of $\alpha_t$ (e.g., as "step size" or "learning rate") and adding $A$ to the Model Setup table.
