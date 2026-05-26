# Finite-Resource Cake Eating

## Overview

Stokey, Lucas, and Prescott (1989) developed the recursive-methods framework to give macroeconomists a rigorous operator-theory foundation for dynamic programming, and the cake-eating problem is the opening worked example that demonstrates the Bellman operator, its contraction property, and benchmarkability against a closed form. A household owns a fixed cake and chooses consumption each period. The cake does not grow. There is no income or uncertainty. Consuming more today leaves less cake for every future period.

The state is the cake stock remaining at the start of the period. The control is current consumption, bounded above by the remaining stock. A policy rule maps each stock into a consumption choice, and the *value function* assigns to each stock the discounted utility from following the optimal policy.

The Bellman equation is solved three ways on the same wealth grid. Method 1 is value function iteration. Method 2 is modified policy iteration, also called Howard acceleration. Method 3 is exact Howard policy iteration. Log utility gives a closed-form value and policy that benchmark all three numerical solutions.

## Read before

- [Optimal growth model](../optimal-growth/README.md)
- [Consumption-savings under income risk](../consumption-savings/README.md)

## Equations

Let $`W_t`$ be remaining cake at the start of period $`t`$.
The household chooses $`c_t \in [0, W_t]`$ and leaves next-period cake:

```math
W_{t+1} = W_t - c_t, \qquad W_0 \text{ given}.
```

Here $`W_0`$ is the initial cake endowment.

Preferences use discount factor $`\beta \in (0,1)`$ and CRRA flow utility:

```math
\sum_{t=0}^{\infty} \beta^t u(c_t),
\qquad u(c)=\frac{c^{1-\sigma}}{1-\sigma},
\qquad u(c)=\log c \text{ when } \sigma=1.
```

The value function solves a one-state *Bellman equation*:

```math
V(W) = \max_{0 \le c \le W} \big\lbrace \underbrace{u(c)}_{\text{flow utility today}} + \underbrace{\beta V(W-c)}_{\text{discounted continuation value}} \big\rbrace.
```

The flow / continuation split is the entire economic content of the Bellman equation.
Eating one more unit today raises $`u(c)`$ but shrinks the stock left for tomorrow, which lowers $`V(W-c)`$.
The optimum balances those two forces.

The first-order condition and envelope condition give the Euler equation:

```math
u'(c_t) = \beta u'(c_{t+1}).
```

This says marginal utility rises as the cake stock falls.
In the log case, consumption falls at rate $`\beta`$.

A policy is a function $`c^{\ast}: W \mapsto c`$ that prescribes a feasible consumption choice at every stock.
Guessing a constant consumption share and verifying the Euler equation gives the closed-form optimal policy:

```math
c^{\ast}(W) = (1-\beta)  W,
\qquad g(W) = W - c^{\ast}(W) = \beta W.
```

Here $`g(W)`$ is the law of motion for the cake under the optimal policy.
The matching value function is:

```math
V(W) = \frac{\ln((1-\beta) W)}{1-\beta} +
\frac{\beta \ln \beta}{(1-\beta)^2},
\qquad V'(W) = \frac{1}{(1-\beta) W}.
```

This closed form is the target for the numerical check.

### Method 1: Value Function Iteration

Let $`T`$ be the Bellman operator.
It maps any candidate value function $`V`$ to a new function $`TV`$ defined pointwise by:

```math
(TV)(W) = \max_{0 \le c \le W} \lbrace u(c) + \beta V(W-c) \rbrace.
```

The operator $`T`$ is a contraction with modulus $`\beta`$ in the sup norm $`\| \cdot \|_\infty`$.
By the Banach fixed-point theorem it has a unique fixed point $`V^{\ast}`$.
The iteration

```math
V_{n+1} = T V_n
```

starts from any guess $`V_0`$ and the sup-norm distance to $`V^{\ast}`$ shrinks by a factor of $`\beta`$ at each step.

### Method 2: Modified Policy Iteration

A policy is a function $`\pi: W \mapsto c`$ that prescribes a consumption choice at every stock $`W`$.
Define the policy operator $`T_\pi`$ that performs one Bellman step with $`\pi`$ held fixed:

```math
(T_{\pi} V)(W) = u(\pi(W)) + \beta V(W - \pi(W)).
```

The operator $`T_\pi`$ is also a $`\beta`$-contraction in the sup norm.
Its unique fixed point is denoted $`V_\pi`$.
$`V_\pi`$ is the expected discounted utility of always playing $`\pi`$.

Let $`T_\pi^{k}`$ denote the $`k`$-fold composition $`T_\pi \circ \cdots \circ T_\pi`$ with $`k`$ copies.
Applying $`T_\pi^{k}`$ to any starting $`V`$ moves it $`k`$ steps closer to $`V_\pi`$.
This tutorial uses the variant of modified policy iteration whose evaluation phase starts from the improved iterate $`T V_n`$, the same Bellman update VFI computes, rather than from $`V_n`$.
One improvement step is followed by $`k`$ such evaluation sweeps, so the policy contraction $`T_\pi`$ is applied a total of $`k+1`$ times per outer step:

```math
\pi_{n+1}(W) \in \arg\max_{c} \lbrace u(c) + \beta V_n(W-c) \rbrace,
\qquad V_{n+1} = T_{\pi_{n+1}}^{k}  (T V_n).
```

The integer $`k`$ is the inner-sweep count and is set by the user.
Choosing $`k=0`$ does no evaluation sweep, so the outer step reduces to $`V_{n+1} = T V_n`$ and recovers value function iteration exactly.
Letting $`k \to \infty`$ recovers exact policy iteration.

### Method 3: Exact Howard Policy Iteration

On the finite grid the policy operator becomes an affine map on $`\mathbb{R}^{N_W}`$.
Let $`P_\pi`$ be the $`N_W \times N_W`$ matrix whose row $`i`$ holds the linear-interpolation weights at the point $`W_i - \pi(W_i)`$.
Row $`i`$ of $`P_\pi`$ has at most two nonzero entries, one for each end of the bracketing interval.
Stacking grid values into a vector, the policy operator becomes:

```math
T_{\pi} V = u(\pi) + \beta P_{\pi}  V.
```

The fixed point $`V_\pi`$ satisfies $`V_\pi = u(\pi) + \beta P_\pi V_\pi`$.
Rearranging gives a linear system in $`V_\pi`$:

```math
\underbrace{(I - \beta P_{\pi})}_{\text{discounted resolvent}}  V_{\pi} = \underbrace{u(\pi)}_{\text{flow utility under } \pi}.
```

The resolvent $`(I - \beta P_\pi)^{-1}`$ is the discrete analogue of the geometric series $`\sum_{k=0}^{\infty} (\beta P_\pi)^k`$, which is exactly the discounted sum of flow utilities along the Markov chain induced by the policy.
That is why the linear system is the exact policy evaluation: it computes the infinite expected discounted utility in one solve rather than approximating it by repeated application of $`T_\pi`$.
The matrix $`I - \beta P_\pi`$ is invertible because $`\beta P_\pi`$ has spectral radius at most $`\beta < 1`$.
Exact policy iteration alternates policy improvement with this exact solve:

```math
\pi_{n+1} \in \arg\max_{c} \lbrace u(c) + \beta V_n(W-c) \rbrace,
\qquad V_{n+1} = \underbrace{(I - \beta P_{\pi_{n+1}})^{-1}  u(\pi_{n+1})}_{\text{exact value of always playing } \pi_{n+1}}.
```

The iteration can be read as Newton's method applied to the fixed-point equation $`V = T V`$.
Near an optimal policy the improvement step makes only second-order changes in $`V`$.
Convergence is therefore super-linear once the policy is close to the optimum, which is why Howard typically finishes in three or four iterations while VFI needs hundreds.

## Worked Numerical Example

Set $`\beta = 0.9`$ and $`W_0 = 1`$ (the calibration in Model Setup) and trace the optimal path for three periods using the closed-form policy $`c^{\ast}(W) = (1-\beta)W`$.

At $`t = 0`$, the household faces stock $`W_0 = 1`$:

```math
c_0 = (1 - 0.9)(1) = 0.1, \qquad W_1 = W_0 - c_0 = 1 - 0.1 = 0.9.
```

At $`t = 1`$, the remaining stock is $`W_1 = 0.9`$:

```math
c_1 = (1 - 0.9)(0.9) = 0.09, \qquad W_2 = 0.9 - 0.09 = 0.81.
```

At $`t = 2`$, the remaining stock is $`W_2 = 0.81`$:

```math
c_2 = (1 - 0.9)(0.81) = 0.081, \qquad W_3 = 0.81 - 0.081 = 0.729.
```

The three-period path follows the general formula $`c_t = (1-\beta)\beta^t W_0`$ and $`W_t = \beta^t W_0`$:

```math
\boxed{(c_0,\, c_1,\, c_2) = (0.1,\; 0.09,\; 0.081)}.
```

Consumption falls by factor $`\beta = 0.9`$ each period. The household never exhausts the cake in finite time because each period it saves the fraction $`\beta`$ of whatever stock remains.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Discount factor $`\beta`$ | 0.9 | Sup-norm tolerance $`\varepsilon`$ | 1e-06 |
| CRRA curvature $`\sigma`$ | 1.0 | Initial cake endowment $`W_0`$ | 1.0 |
| Wealth grid $`N_W`$ | 500 pts | Wealth grid range $`[w_{\min}, w_{\max}]`$ | $`[0.01, 1.0]`$ |
| Consumption grid $`N_c`$ | 300 pts | MPI inner sweeps $`k`$ | 5 |
| Simulation periods $`T_{\mathrm{sim}}`$ | 30 | | |

## Solution Method

All three solvers run on the same wealth grid with the same initial guess $`V_0(W_i) = u(W_i)`$ and the same off-grid continuation rule that interpolates $`V`$ on $`W' = W - c`$. What differs between methods is how the value function is updated between successive policy improvements. VFI applies the Bellman operator once per outer step. MPI applies the policy operator $`k+1`$ times per step to extract more information from each improved policy. Exact PI solves the policy evaluation exactly by linear algebra in one step.

```
            W-grid, u(·), β, ε
                     |
                     v
              initial guess V_0
                     |
    +------ outer loop -------------------------+
    |                                           |
    |   +---- policy improvement -----------+   |
    |   |   Bellman maximization --> π_new  |   |
    |   +------------------------------------+   |
    |                    |                      |
    |   +---- value update (by method) ------+  |
    |   |   VFI:      apply T once           |  |
    |   |   MPI:      apply T_π  k+1 times   |  |
    |   |   Exact PI: resolvent linear solve  |  |
    |   +-------------------------------------+  |
    |                    |                      |
    +------ err >= ε: repeat -------------------+
                     |
                 err < ε
                     v
              V*(W_i), c*(W_i)
```

```python
# All three methods share the policy improvement step (Bellman maximization).
# Only the value update differs. Shown here for VFI; see run.py for MPI and Exact PI.

def bellman_step(v):
    # (TV)(W_i) = max_c { u(c) + β V(W_i - c) }
    v_new = np.zeros(n_grid)
    policy_c = np.zeros(n_grid)
    for ia in range(n_grid):
        cake = w_grid[ia]
        c_grid = np.linspace(1e-8, cake * 0.9999, n_cons)
        wprime = cake - c_grid
        values = u_vec(c_grid) + beta * v_interp(wprime, v)
        best = np.argmax(values)
        v_new[ia] = values[best]
        policy_c[ia] = c_grid[best]
    return v_new, policy_c

# VFI outer loop: err shrinks by β each step, so ~log(ε)/log(β) iterations needed.
v = u_vec(w_grid)
while True:
    v_new, policy = bellman_step(v)
    err = np.max(np.abs(v_new - v))
    v = v_new
    if err < tol:
        break
```

VFI converges in 68 iterations. MPI with $`k=5`$ converges in 13 outer iterations. Exact PI converges in 11 outer iterations with super-linear rate. The wall times on this calibration are recorded in the diagnostics table below.

## Results

The value function $`V(W)`$ is concave in the stock $`W`$ and matches the closed form away from the lower boundary. Outside the bottom decile of the wealth grid, the largest sup-norm gap to the closed form is 2.52e-02. The gap near $`W = 0`$ is driven by the log singularity in $`u`$, which linear interpolation cannot capture.

![Value function and convergence across solvers](figures/value-convergence.png)

Under log utility the household consumes a constant share $`1 - \beta = `$10% of the remaining stock. The numerical consumption policy $`c^{\ast}(W)`$ traces the closed-form line through the origin. The dotted $`45^{\circ}`$ line marks immediate exhaustion of the cake. Above the bottom decile of the wealth grid, the largest sup-norm gap is 3.24e-04.

![Consumption policy and depletion simulation](figures/policy-simulation.png)

The convergence plot shows three different rates on the same problem. VFI traces a straight line on the log scale with slope $`\log_{10} \beta`$. This is the contraction rate of the operator $`T`$ in the sup norm. MPI with $`k = 5`$ inner sweeps drops faster because each outer step composes the policy contraction $`T_\pi`$ a total of $`k+1`$ times. Exact PI reaches tolerance in a handful of outer iterations and shows the super-linear shape characteristic of Newton's method.

The pointwise table reports the value function at eight selected wealth states. The numerical columns agree to within tolerance at every row.

Value and policy from VFI, MPI, and exact PI against the closed form

|     W |    V VFI |    V MPI |     V PI |   V closed form |   c VFI |   c closed form |
|------:|---------:|---------:|---------:|----------------:|--------:|----------------:|
| 0.109 | -54.6794 | -54.6794 | -54.6794 |        -54.6542 |  0.011  |          0.0109 |
| 0.236 | -46.9527 | -46.9527 | -46.9527 |        -46.9402 |  0.0237 |          0.0236 |
| 0.363 | -42.646  | -42.646  | -42.646  |        -42.6378 |  0.0364 |          0.0363 |
| 0.49  | -39.6455 | -39.6455 | -39.6455 |        -39.6393 |  0.0492 |          0.049  |
| 0.617 | -37.3406 | -37.3406 | -37.3406 |        -37.3356 |  0.0619 |          0.0617 |
| 0.744 | -35.4687 | -35.4687 | -35.4687 |        -35.4645 |  0.0746 |          0.0744 |
| 0.871 | -33.8925 | -33.8925 | -33.8925 |        -33.8889 |  0.0874 |          0.0871 |
| 1     | -32.5114 | -32.5114 | -32.5114 |        -32.5083 |  0.1003 |          0.1    |

The method table summarises the trade-off across the three solvers. VFI takes the most outer iterations but the cheapest per-iteration work. Exact PI takes the fewest outer iterations but pays for an $`O(N_W^{3})`$ linear solve at each step. MPI sits between the two extremes and is the workhorse for larger state spaces.

### Solver diagnostics

| Method | Outer iterations | Method | Wall time (s) |
|:---|---:|:---|---:|
| Value function iteration | 68 | Value function iteration | 0.54 |
| Modified policy iteration | 13 | Modified policy iteration | 0.11 |
| Exact policy iteration | 11 | Exact policy iteration | 0.15 |

| Method | Final update | Method | Sup-norm vs closed form |
|:---|---:|:---|---:|
| Value function iteration | 4.23e-07 | Value function iteration | 0.0252 |
| Modified policy iteration | 1.88e-12 | Modified policy iteration | 0.0252 |
| Exact policy iteration | 0 | Exact policy iteration | 0.0252 |

## Takeaway

Cake eating isolates Bellman logic in its simplest form: a one-state deterministic resource problem with no production, no uncertainty, and no income. The optimal policy consumes a constant fraction of the remaining stock each period, and under log utility that fraction has an exact closed form. This closed form is what Stokey, Lucas, and Prescott used to demonstrate their operator-theory apparatus delivers verifiable answers. The framework went on to anchor the modern recursive-methods toolkit, from stochastic consumption-savings to heterogeneous-agent general equilibrium.

VFI converges slowly because each step peels off only one geometric factor of the discount rate. Modified policy iteration extracts more information from each improved policy by repeating the evaluation step, shrinking the iteration count from dozens to a handful. Exact policy iteration inverts the policy evaluation in one linear solve and converges super-linearly. All three converge to the same grid approximation. The gap that remains traces to the finite grids, not to the choice of solver.

## See also

- [Consumption-savings under income risk](../consumption-savings/README.md)
- [Aiyagari saving and capital-market clearing](../aiyagari/README.md)

## References

- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 4-5.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.
- Howard, R. (1960). *Dynamic Programming and Markov Processes*. MIT Press.
- Puterman, M. and Brumelle, S. (1979). On the convergence of policy iteration in stationary dynamic programming. *Mathematics of Operations Research*, 4(1), 60-69.
