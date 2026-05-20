# bullshit-detector -- rum-choice-networks -- 2026-05-20

**Bullshit score: 0%** -- Every claim in README.md is grounded verbatim in run.py or the CSV artifacts; the hostile reviewer reads it twice and finds no hole.

## Header
- Claim sources: `structural-econometrics/rum-choice-networks/README.md`
- Code / artifact root: `structural-econometrics/rum-choice-networks/run.py`
- Data artifacts: `structural-econometrics/rum-choice-networks/tables/fit-comparison.csv`, `tables/test-share-fit.csv`, `tables/premium-price-shock.csv`, `tables/learning-curve.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | DGP utility: 7-term decomposition with exact coefficients | HOLDS | none | no |
| 2 | alpha_i, beta_i, kappa, sigma_j, r_ij exact formulas | HOLDS | none | no |
| 3 | Feature vector x_tilde_ijr: 12 named terms | HOLDS | none | no |
| 4 | Linear utility v_theta uses only (p, q, pz, qz) in linear part | HOLDS | none | no |
| 5 | h_ijr = tanh(W^T x_tilde + d), one hidden layer | HOLDS | none | no |
| 6 | P_hat_ij = (1/R) sum_r P_ijr, averaged over R=9 fixed draws | HOLDS | none | no |
| 7 | Penalty = lambda * (theta_net^T theta_net) / d_net | HOLDS | none | no |
| 8 | Logit NLL formula Q_L, normalization a^L_1=0 | HOLDS | none | no |
| 9 | RUMnet initialization: logit params transferred, neural weights small random, biases 0 | HOLDS | none | no |
| 10 | Recapture formula D_{n,Premium} = (s_n^+ - s_n)/(s_Premium - s_Premium^+) | HOLDS | none | no |
| 11 | All four README tables match CSV artifacts exactly | HOLDS | none | no |
| 12 | N_TRAIN=3000, N_TEST=1500, R=9, H=6, lambda=0.012 | HOLDS | none | no |

## Findings

No non-HOLDS findings. All twelve claims ground to verbatim code or CSV artifacts as shown below.

### Finding 1: DGP utility decomposition (HOLDS)

- **Claim source (verbatim):** `v^0_{ij}=\delta_j+\alpha_i p_{ij}+\beta_i q_{ij}+m_j(z_i)+\ell_j(q_{ij},z_i)+\sigma_j(z_i)+r_{ij}(\eta_i).` -- `README.md:44`
- **Code evidence (verbatim):**
  ```python
  return (
      intercept
      + price_taste[:, None] * price
      + quality_taste[:, None] * quality
      + product_context
      + nonlinear_quality
      + latent_match
      + premium_switch
  )
  ```
  `run.py:105-113`
- **Category:** HOLDS

### Finding 2: alpha_i, beta_i exact formulas (HOLDS)

- **Claim source (verbatim):** `\alpha_i=-(1.05+0.22\tanh(z_i)+0.12\eta_i+0.08\eta_i\tanh(z_i)).` and `\beta_i=0.78+0.30\tanh(1.10z_i+0.45\eta_i).` -- `README.md:50-52`
- **Code evidence (verbatim):**
  ```python
  price_taste = -(
      1.05
      + 0.22 * np.tanh(context)
      + 0.12 * eta
      + 0.08 * eta * np.tanh(context)
  )
  quality_taste = 0.78 + 0.30 * np.tanh(1.10 * context + 0.45 * eta)
  ```
  `run.py:88-94`
- **Category:** HOLDS

### Finding 3: m_j, kappa, sigma_j, r_ij exact formulas (HOLDS)

- **Claim source (verbatim):** `m(z_i)=(-0.35\tanh(1.20z_i)+0.10(z_i^2-1),\ldots)`, `\kappa=(-0.15,0.20,0.62)`, `\sigma_j(z_i)=(0,0.12\tanh(1.80z_i)^2,0.35\tanh(1.50z_i)^2)`, `r_{ij}(\eta_i)=0.18\eta_i\tanh(q_{ij}z_i)` -- `README.md:56-68`
- **Code evidence (verbatim):**
  ```python
  product_context = np.column_stack([
      -0.35 * context_tilt + 0.10 * context_hump,
      0.10 * context_tilt - 0.08 * context_hump,
      0.55 * context_tilt - 0.28 * context_hump,
  ])
  nonlinear_quality = (
      np.array([-0.15, 0.20, 0.62])[None, :]
      * np.tanh(1.15 * (quality - 1.25) * context[:, None])
  )
  latent_match = 0.18 * eta[:, None] * np.tanh(quality * context[:, None])
  premium_switch = np.column_stack([
      np.zeros_like(context),
      0.12 * np.tanh(1.80 * context) ** 2,
      0.35 * np.tanh(1.50 * context) ** 2,
  ])
  ```
  `run.py:83-104`
- **Category:** HOLDS

### Finding 4: Feature vector x_tilde_ijr -- 12 terms (HOLDS)

- **Claim source (verbatim):** `\tilde x_{ijr}=(p_{ij},q_{ij},z_i,p_{ij}z_i,q_{ij}z_i,z_i^2,q_{ij}z_i^2,\eta_r,p_{ij}\eta_r,q_{ij}\eta_r,z_i\eta_r,q_{ij}z_i\eta_r).` -- `README.md:76`
- **Code evidence (verbatim):**
  ```python
  return np.concatenate(
      [
          price,
          quality,
          context,
          price * context,
          quality * context,
          context_sq,
          quality * context_sq,
          draw,
          price * draw,
          quality * draw,
          context * draw,
          quality * context * draw,
      ],
      axis=-1,
  ).astype(np.float32)
  ```
  `run.py:223-239`
- **Category:** HOLDS

### Finding 5: Linear part of v_theta uses exactly (p, q, pz, qz) (HOLDS)

- **Claim source (verbatim):** `v_{\theta}(i,j,r)=a_j+b_p p_{ij}+b_q q_{ij}+b_{pz}p_{ij}z_i+b_{qz}q_{ij}z_i+c^{\top}h_{ijr}(\theta).` -- `README.md:82`
- **Code evidence (verbatim):**
  ```python
  linear_utility = intercept + jnp.einsum("nkjd,d->nkj", features[:, :, :, [0, 1, 3, 4]], slopes)
  hidden = jnp.tanh(jnp.einsum("nkjd,dh->nkjh", features, hidden_weights) + hidden_bias)
  utility = linear_utility + jnp.einsum("nkjh,h->nkj", hidden, output_weights)
  ```
  `run.py:260-262`
- **Note:** `features[:,:,:,[0,1,3,4]]` = (p, q, p*z, q*z) -- indices 0,1,3,4 of the 12-element feature vector, matching the four slopes (b_p, b_q, b_pz, b_qz). HOLDS exactly.
- **Category:** HOLDS

### Finding 6: P_hat averaged over R fixed draws (HOLDS)

- **Claim source (verbatim):** `\widehat P_{ij}(\theta)=\frac{1}{R}\sum_{r=1}^{R}P_{ijr}(\theta).` -- `README.md:90`
- **Code evidence (verbatim):**
  ```python
  chosen_log_probability = log_probability[
      jnp.arange(choice.shape[0])[:, None],
      jnp.arange(features.shape[1])[None, :],
      choice[:, None],
  ]
  probability = jnp.mean(jnp.exp(chosen_log_probability), axis=1)
  ```
  `run.py:264-269` (training) and `return softmax_np(utility).mean(axis=1)` `run.py:365` (prediction)
- **Category:** HOLDS

### Finding 7: Penalty formula lambda * theta_net^T theta_net / d_net (HOLDS)

- **Claim source (verbatim):** `Q_R(\theta)=\frac{-1}{N}\sum_{i=1}^{N}\log \max(\widehat P_{i y_i}(\theta),10^{-12})+\lambda\frac{\theta_{\mathrm{net}}^{\top}\theta_{\mathrm{net}}}{d_{\mathrm{net}}}.` -- `README.md:94`
- **Code evidence (verbatim):**
  ```python
  penalty = RUMNET_L2 * jnp.mean(theta[(N_PRODUCTS - 1) + 4 :] ** 2)
  return -jnp.mean(jnp.log(jnp.maximum(probability, 1e-12))) + penalty
  ```
  `run.py:270-271`
- **Note:** `theta[6:]` covers hidden_weights(72) + hidden_bias(6) + output_weights(6) = 84 params. `jnp.mean(x**2)` = `sum(x**2)/84` = `theta_net^T theta_net / d_net` with d_net=84. Mathematically identical. HOLDS.
- **Category:** HOLDS

### Finding 8: Logit NLL and normalization a^L_1=0 (HOLDS)

- **Claim source (verbatim):** `Q_L(a^L,b^L)=\frac{-1}{N}\sum_{i=1}^{N}\log P^L_{i y_i}.` and `a^L_1=0` for normalization -- `README.md:33`, `README.md:125`
- **Code evidence (verbatim):**
  ```python
  def unpack_logit(theta):
      intercept = jnp.concatenate([jnp.zeros(1), theta[: N_PRODUCTS - 1]])
      slopes = theta[N_PRODUCTS - 1 :]
      return intercept, slopes

  def logit_nll(theta, features, choice):
      intercept, slopes = unpack_logit(theta)
      utility = intercept + jnp.einsum("njd,d->nj", features, slopes)
      chosen = utility[jnp.arange(choice.shape[0]), choice]
      return -jnp.mean(chosen - jax.nn.logsumexp(utility, axis=1))
  ```
  `run.py:172-184`
- **Category:** HOLDS

### Finding 9: RUMnet initialization from logit estimate (HOLDS)

- **Claim source (verbatim):** `a_j^{(0)}=\hat a^L_j, \quad (b_p^{(0)},b_q^{(0)},b_{pz}^{(0)},b_{qz}^{(0)})=\hat b^L.` and `The neural weights start as small random numbers and the hidden biases start at zero.` -- `README.md:132-135`
- **Code evidence (verbatim):**
  ```python
  theta = np.zeros(n_parameters, dtype=float)
  theta[: (N_PRODUCTS - 1) + 4] = logit_theta
  cursor = (N_PRODUCTS - 1) + 4
  theta[cursor : cursor + N_FEATURES * HIDDEN_UNITS] = rng.normal(0.0, scale, size=N_FEATURES * HIDDEN_UNITS)
  cursor += N_FEATURES * HIDDEN_UNITS + HIDDEN_UNITS   # skips hidden_bias, stays 0
  theta[cursor : cursor + HIDDEN_UNITS] = rng.normal(0.0, scale, size=HIDDEN_UNITS)
  ```
  `run.py:287-296`
- **Note:** `theta[:6] = logit_theta` transfers a_j (2 params) and b^L (4 params). `theta[6:78]` = hidden_weights W (randomized). `theta[78:84]` = hidden_bias d (remains 0, never filled). `theta[84:90]` = output_weights c (randomized). Exactly matches the README claim. HOLDS.
- **Category:** HOLDS

### Finding 10: Recapture formula (HOLDS)

- **Claim source (verbatim):** `D_{n,\mathrm{Premium}}=\frac{s_n^{+}-s_n}{s_{\mathrm{Premium}}-s_{\mathrm{Premium}}^{+}}.` -- `README.md:148`
- **Code evidence (verbatim):**
  ```python
  lost_share = base_shares[PRICE_SHOCK_PRODUCT] - shocked_shares[PRICE_SHOCK_PRODUCT]
  for product_id, product in enumerate(PRODUCTS):
      if product_id == PRICE_SHOCK_PRODUCT:
          recapture = -1.0
      else:
          recapture = (shocked_shares[product_id] - base_shares[product_id]) / lost_share
  ```
  `run.py:453-458`
- **Category:** HOLDS

### Finding 11: All README tables match CSV artifacts (HOLDS)

- **Claim source:** All four tables in `README.md:233-274`
- **Data evidence:** `tables/fit-comparison.csv`, `tables/test-share-fit.csv`, `tables/premium-price-shock.csv`, `tables/learning-curve.csv` -- every cell matches the README to 4 decimal places. No DATA DRIFT between README prose and CSV files.
- **Category:** HOLDS

### Finding 12: Configuration constants (HOLDS)

- **Claim source (verbatim):** Model Setup table -- N_PRODUCTS=3, N_TRAIN=3000, N_TEST=1500, R=9, H=6, lambda=0.012, LEARNING_CURVE_SIZES=(300,600,1200,3000), PRICE_SHOCK=+0.25 on Premium -- `README.md:101-111`
- **Code evidence (verbatim):**
  ```python
  N_PRODUCTS = len(PRODUCTS)   # 3
  N_TRAIN = 3_000
  N_TEST = 1_500
  N_LATENT_DRAWS = 9
  HIDDEN_UNITS = 6
  N_FEATURES = 12
  RUMNET_L2 = 1.2e-2          # 0.012
  LEARNING_CURVE_SIZES = (300, 600, 1_200, N_TRAIN)
  PRICE_SHOCK_PRODUCT = 2     # Premium (index 2)
  PRICE_SHOCK = 0.25
  ```
  `run.py:27-37`
- **Category:** HOLDS

## Cross-cutting patterns

None. All findings are HOLDS. No disease pattern to flag.

## TDD execution sequence (for the next agent)

**Bullshit score: 0%.** No non-HOLDS findings. No tests to write, no fixes to design.

The audit is clean. Forward work on this tutorial does not require any remediation. Re-run this skill after any future change to `run.py` that touches the loss function, the feature vector, or the DGP coefficients.
