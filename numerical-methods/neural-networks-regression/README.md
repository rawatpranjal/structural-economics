# Feedforward Neural Networks for Regression and Density Approximation

## Overview

Several tutorials in this catalog learn a function from data using a small feedforward neural network. The random utility model in `rum-choice-networks/` bends the utility surface with a tanh layer before the softmax. The adversarial-estimation tutorial trains a shallow tanh discriminator to tell real observations from simulated ones. The deep-optimal-auctions tutorial parameterises an entire mechanism as a differentiable network and trains it by autodiff. All three start from the same primitive: a one-hidden-layer tanh network with a small number of units, a squared-error or cross-entropy loss, weight decay, and an Adam optimiser.

The *universal approximation theorem* (Hornik et al., 1989) says a one-hidden-layer network with enough units can represent any continuous function on a compact domain. Before that result, economists and statisticians had no principled reason to expect a shallow network to outperform a well-chosen polynomial on smooth economic surfaces.

This prelim presents that primitive in isolation. It fits a one-hidden-layer tanh network with sixteen units to a Cobb-Douglas surface (production output as a function of capital and labour with constant returns to scale) observed with Gaussian noise. It compares the fit against a linear baseline and a degree-three polynomial baseline. It sweeps the weight-decay strength to show the bias-variance trade-off. It overlays training curves for Adam against plain stochastic gradient descent to motivate adaptive learning rates.

The four moving parts are the forward pass, the squared-error loss with L2 weight decay (penalty on the sum of squared weights, called L2 because it is the squared L2-norm of the weight vector), the gradient of the loss computed by JAX automatic differentiation (autodiff, which builds the chain rule programmatically), and the Adam parameter update. The same four parts show up in every downstream neural tutorial. Only the loss and the architecture change.

## Read before

- [Root-finding for equilibrium rates](../root-finding/README.md)
- [Fixed-point acceleration](../fixed-point-acceleration/README.md)

## Equations

We need notation for the network's input, output, weights, and biases. Let $`x \in \mathbb{R}^{d_{\mathrm{in}}}`$ be the input vector, let $`H`$ be the number of hidden units, and let $`W \in \mathbb{R}^{H \times d_{\mathrm{in}}}`$, $`b \in \mathbb{R}^{H}`$ be the input-to-hidden weights and biases, with $`\widetilde W \in \mathbb{R}^{d_{\mathrm{out}} \times H}`$, $`\widetilde b \in \mathbb{R}^{d_{\mathrm{out}}}`$ for the hidden-to-output layer. A one-hidden-layer feedforward network with tanh activation maps $`x`$ to a prediction $`\widehat y`$ in two steps:

```math
h = \tanh(W x + b),
\qquad
\widehat y = \widetilde W h + \widetilde b.
```

The hidden activation $`h \in \mathbb{R}^{H}`$ is a smooth, bounded transformation of the input. Tanh is bounded in $`(-1, 1)`$ and is symmetric around zero, which keeps the output linear-in-the-tail and stable to train. The output layer is linear in $`h`$. For regression problems it returns a real number. For binary classification it is composed with a sigmoid to return a probability.

We need a loss that combines fit and a complexity penalty. The squared-error loss with L2 weight decay sums a mean-squared error over $`n`$ training points and a quadratic penalty on the weight matrices:

```math
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \widehat y_i\right)^2 + \lambda  \left(\|W\|_F^2 + \|\widetilde W\|_F^2\right),
\qquad
\theta = (W, b, \widetilde W, \widetilde b).
```

The Frobenius norm $`\|W\|_F^2 = \sum_{ij} W_{ij}^2`$ penalises large entries in the weight matrices but leaves the biases unregularised, which is standard. The scalar $`\lambda \geq 0`$ trades fit against simplicity. At $`\lambda = 0`$, the network is free to interpolate noise. As $`\lambda`$ grows, the optimal $`W`$ shrinks toward zero and the network's output collapses toward its bias, recovering the constant-mean predictor.

We need the gradient of $`\mathcal{L}`$ with respect to $`\theta`$ to run a first-order optimiser. The chain rule decomposes that gradient through the network's layers. Writing the loss as a function of intermediate activations, with $`r_i = y_i - \widehat y_i`$ the residual and $`\odot`$ the elementwise product, the gradient components are:

```math
\begin{aligned}
\nabla_{\widetilde W} \mathcal{L} &= -\tfrac{2}{n} \sum_{i} r_i  h_i^{\top} + 2\lambda  \widetilde W, \\
\nabla_{W} \mathcal{L} &= -\tfrac{2}{n} \sum_{i} \big(\widetilde W^{\top} r_i \odot (1 - h_i \odot h_i)\big)  x_i^{\top} + 2\lambda  W,
\end{aligned}
```

with analogous expressions for $`\widetilde b`$ and $`b`$. The hand derivation is mechanical and brittle to write out. That brittleness is the use case for automatic differentiation. JAX builds the computational graph at trace time and returns the gradient of any scalar-valued function automatically via `jax.grad`. The implementation calls `grad(loss_fn)` and treats the returned function as a black-box gradient oracle.

We need a parameter-update rule that adapts the step size per coordinate. *Adam* (adaptive moment estimation) adds two running averages to plain gradient descent: a first moment that tracks the mean gradient and a second moment that tracks the variance. It divides the update by the square root of the second moment, so steep coordinates step less and shallow coordinates step more. At iteration $`t`$ with hyperparameters $`\beta_1, \beta_2 \in (0, 1)`$, $`\varepsilon > 0`$, and learning rate $`\eta`$:

```math
\begin{aligned}
m_t &= \beta_1  m_{t-1} + (1 - \beta_1)  g_t, \\
v_t &= \beta_2  v_{t-1} + (1 - \beta_2)  g_t \odot g_t, \\
\widehat m_t &= \frac{m_t}{1 - \beta_1^{t}}, \qquad \widehat v_t = \frac{v_t}{1 - \beta_2^{t}}, \\
\theta_t &= \theta_{t-1} - \eta  \frac{\widehat m_t}{\sqrt{\widehat v_t} + \varepsilon}.
\end{aligned}
```

The bias-corrected estimates $`\widehat m_t, \widehat v_t`$ compensate for the running averages being initialised at zero. The element-wise denominator $`\sqrt{\widehat v_t} + \varepsilon`$ is the per-coordinate adaptive step size. In practice Adam reaches a good fit in several hundred to a few thousand updates on small networks. Plain SGD with the same learning rate is several times slower.

## Worked Numerical Example

One forward pass and one gradient step on a one-hidden-unit network make the chain rule concrete. The architecture collapses to scalars: hidden activation $`h = \tanh(w_1 x + b_1)`$ and prediction $`\widehat y = w_2 h + b_2`$. Input $`x = 2`$, target $`y = 0.5`$, initial weights $`w_1 = 0.5`$, $`b_1 = 0`$, $`w_2 = 1.0`$, $`b_2 = 0`$, and plain SGD with $`\eta = 0.1`$ and $`\lambda = 0`$ (no weight decay, so the example isolates the chain rule).

Forward pass. The pre-activation is $`w_1 x + b_1 = 0.5 \cdot 2 + 0 = 1.0`$, so

```math
h = \tanh(1.0) = 0.7616, \qquad \widehat y = 1.0 \cdot 0.7616 + 0 = 0.7616.
```

The squared-error loss (without the factor of $`2/n`$ used in batched training) is

```math
L = \tfrac{1}{2} (y - \widehat y)^2 = \tfrac{1}{2} (0.5 - 0.7616)^2 = \tfrac{1}{2} (-0.2616)^2 = 0.0342.
```

*Backpropagation.* The output-side residual is $`\partial L / \partial \widehat y = -(y - \widehat y) = 0.2616`$. Differentiating $`\widehat y = w_2 h + b_2`$ gives $`\partial L / \partial w_2 = (\partial L / \partial \widehat y) \cdot h = 0.2616 \cdot 0.7616 = 0.1993`$. The signal propagated back into the hidden layer is $`\partial L / \partial h = (\partial L / \partial \widehat y) \cdot w_2 = 0.2616 \cdot 1.0 = 0.2616`$. The tanh derivative at the pre-activation is $`1 - \tanh^2(1.0) = 1 - 0.5800 = 0.4200`$, so

```math
\frac{\partial L}{\partial (w_1 x + b_1)} = \frac{\partial L}{\partial h} \cdot (1 - h^2) = 0.2616 \cdot 0.4200 = 0.1099.
```

Multiplying by $`x = 2`$ gives $`\partial L / \partial w_1 = 0.1099 \cdot 2 = 0.2197`$.

One SGD step. Subtract $`\eta`$ times each gradient:

```math
w_2^{\mathrm{new}} = 1.0 - 0.1 \cdot 0.1993 = 0.9801, \qquad w_1^{\mathrm{new}} = 0.5 - 0.1 \cdot 0.2197 = 0.4780.
```

```math
\boxed{w_1^{\mathrm{new}} = 0.478, \quad w_2^{\mathrm{new}} = 0.980.}
```

Both weights shrink because the prediction overshoots the target. The tanh derivative attenuates the input-side gradient by the factor $`1 - h^2 = 0.42`$, which is why $`w_1`$ moves less per unit of output residual than $`w_2`$ does. Stacking many such updates and replacing the by-hand chain rule with `jax.grad` is the entire training loop the dense tutorial runs.

## Model Setup

| Object | Value | Object | Value |
|:---|:---|:---|:---|
| Input dimension $`d_{\mathrm{in}}`$ | 2 (capital, labour) | Output dimension $`d_{\mathrm{out}}`$ | 1 (predicted output) |
| Hidden units $`H`$ | 16 | Hidden activation $`h`$ | $`\tanh(W x + b)`$ |
| Input weights $`W`$ | $`H \times d_{\mathrm{in}}`$ matrix | Output weights $`\widetilde W`$ | $`d_{\mathrm{out}} \times H`$ matrix |
| Hidden biases $`b`$ | $`H`$-vector | Output bias $`\widetilde b`$ | scalar |
| Prediction $`\widehat y`$ | $`\widetilde W h + \widetilde b`$ | Weight-decay strength $`\lambda`$ | 0, 1e-3, 1e-2 |
| Adam learning rate $`\eta`$ | 0.01 | Adam moment decays $`\beta_1, \beta_2`$ | 0.9, 0.999 |
| Training points $`n`$ | 400 | Test points | 1000 |
| Training steps | 4000 | Cobb-Douglas TFP $`A`$ | 1.0 |
| Cobb-Douglas exponent $`\alpha`$ | 0.4 | Noise scale $`\sigma_\varepsilon`$ | 0.05 |

The input-weight convention $`W x`$ is shared with `adversarial-estimation/`. The tutorial `rum-choice-networks/` uses the transpose form $`W^{\top} x`$.

## Solution Method

Training has three stages: He-scaled parameter initialisation, a fixed number of Adam steps with JAX autodiff supplying the gradient, and held-out evaluation. The bias-variance sweep repeats the full procedure for each weight-decay strength. Plain SGD replaces the Adam update with a single subtraction and runs alongside for comparison.

```
          (X_train, y_train), architecture, lambda, T steps
                              |
                              v
                  +-- [ He initialisation ] --+
                  |     W, b, W_out, b_out    |
                  +---------------------------+
                              |
                              v
    +-------------- training loop ---------------+
    |  theta --> [ JAX gradient oracle ] --> g   |
    |  g     --> [ Adam update ]        --> theta|
    +----------- step < T: repeat ---------------+
                              |
                              v
              +------- [ evaluation ] -------+
              |  theta --> [ train MSE ]      |
              |  theta --> [ test MSE ]       |
              +------------------------------+
                              |
                              v
                   W*, b*, loss history
```

Deeper architectures, convolutional and attention layers, normalising flows, and meta-learning are out of scope. Architecture choices specific to consumer tutorials (RUMnet structure, GAN discriminator design, mechanism heads) live in those tutorials.

## Results

The network fits the Cobb-Douglas surface closely. The figure below places the truth and the neural network fit side by side on the same colour scale.

<img src="figures/fitted-surface.png" alt="Two-panel comparison of the true Cobb-Douglas surface and the neural network fit on a 30 by 30 grid" width="90%">

The right panel is visually indistinguishable from the truth at the plotting scale. The maximum absolute pointwise error across the domain is small, with the largest deviations near the corners where the surface curvature is greatest. There is no systematic bias along either input axis.

The next figure sweeps the weight-decay strength and reports in-sample and out-of-sample fit against the linear and polynomial baselines.

<img src="figures/bias-variance-vs-lambda.png" alt="Train and test MSE versus weight-decay lambda, with horizontal reference lines for linear and degree-3 polynomial baselines and the noise variance" width="85%">

At small weight decay the network reaches the *noise floor*. At large weight decay the penalty dominates the fit term and the network collapses back toward the linear baseline. The degree-3 polynomial baseline matches the network at small weight decay. It has enough flexibility to express a Cobb-Douglas surface on this domain.

The training curves compare Adam against plain SGD with the same learning rate.

<img src="figures/training-curves.png" alt="Training loss versus step on log scale for Adam at three weight-decay strengths and for plain SGD at lambda = 0" width="85%">

Adam reaches the noise floor in a few hundred steps. Plain SGD at the same learning rate remains well above the noise floor at the end of training. The loss landscape has coordinates with very different curvatures. A single global step size compromises between them. The per-coordinate adaptive step size avoids that compromise.

### Diagnostics

| Model | Train MSE | Test MSE |
|:---|---:|---:|
| Linear | -- | -- |
| Polynomial deg-3 | -- | -- |
| Neural network, small weight decay | -- | -- |
| Neural network, large weight decay | -- | -- |

## Takeaway

A one-hidden-layer tanh network is enough to fit a smooth two-input target like the Cobb-Douglas surface to the noise floor. The four moving parts (forward pass, squared-error loss with L2 weight decay, JAX autodiff for the gradient, Adam optimiser) are general. Downstream tutorials reuse the same code with different loss functions and different architectures.

Weight decay trades fit for simplicity. Set it too small and the network can interpolate noise. Set it too large and it collapses to a constant. The practical reason to regularise is that the right strength keeps test error near the noise floor across noise realisations.

The surprise Rumelhart et al. (1986) documented is that the *chain rule*, applied layer by layer, trains networks that earlier researchers considered intractable. The legacy is that every gradient-based neural tutorial, including the three that follow this one in the catalog, rests on exactly the same backpropagation identity derived in that paper.

## See also

- [`structural-econometrics/rum-choice-networks/`](../../structural-econometrics/rum-choice-networks/) -- the neural utility layer reuses this tanh primitive inside a logit choice probability.
- [`structural-econometrics/adversarial-estimation/`](../../structural-econometrics/adversarial-estimation/) -- the shallow discriminator reuses it with a sigmoid output and a cross-entropy loss.
- [`game-theory/deep-optimal-auctions/`](../../game-theory/deep-optimal-auctions/) -- the mechanism head reuses it as a differentiable allocation and payment rule trained by autodiff against a regret penalty.

## References

- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323, 533-536. Original derivation of backpropagation for multilayer networks.
- Hornik, K., Stinchcombe, M., and White, H. (1989). "Multilayer feedforward networks are universal approximators." *Neural Networks*, 2(5), 359-366. Establishes that one hidden layer suffices to approximate any continuous function on a compact set.
- Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press, Chapters 6-7. Feedforward networks and regularisation.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer, Chapter 5. Classical neural networks with Bayesian regularisation framing.
- Kingma, D. P. and Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *International Conference on Learning Representations*. The optimiser used everywhere in the consumer tutorials.
- Athey, S. and Imbens, G. W. (2019). "Machine Learning Methods That Economists Should Know About." *Annual Review of Economics*, 11, 685-725. Positioning for economist readers.
