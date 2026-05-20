# bullshit-detector -- zero-intelligence-traders -- 2026-05-20

**Bullshit score: 0%** -- Every hostile line of audit found code that matched it. Even the worst detractor reads this twice and finds no hole.

## Header
- Claim sources: `agent-based-models/zero-intelligence-traders/README.md` (Overview, Equations, Model Setup, Results)
- Code / artifact root: `agent-based-models/zero-intelligence-traders/run.py`
- Data artifacts: `tables/transaction-log.csv`, `tables/market-type-summary.csv`, `tables/agent-mix-summary.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, ultra caveman session)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|-----------------|
| 1 | ZIC bid `U[0, v_i]` | HOLDS | -- | no |
| 2 | ZIC ask `U[c_j, pbar]` | HOLDS | -- | no |
| 3 | Trade price = midpoint of best bid and best ask | HOLDS | -- | no |
| 4 | Arrival probability proportional to active traders | HOLDS | -- | no |
| 5 | Stale quotes deleted after each trade | HOLDS | -- | no |
| 6 | σ_p is population std (divisor T_p) | HOLDS | -- | no |
| 7 | Q*=7, S*=294.00, P*=[70.00, 72.00] (balanced) | HOLDS | -- | no |
| 8 | P* formula: lower=max{c_(Q*),v_(Q*+1)}, upper=min{v_(Q*),c_(Q*+1)} | HOLDS | -- | no |
| 9 | Buyer-heavy P*=[73,78], Q*=8 | HOLDS | -- | no |
| 10 | Seller-heavy P*=[66,70], Q*=8 | HOLDS | -- | no |
| 11 | Thin P*=[68,72], Q*=4 | HOLDS | -- | no |
| 12 | Baseline AE=99.3%, mean price=67.58, σ_p=12.01 | HOLDS | -- | no |
| 13 | ZIP buyer update: z_i^B(t+1)=(1-λ)z_i^B(t)+λ min{v_i, p_t+κ} | HOLDS | -- | no |
| 14 | ZIP seller update: z_j^S(t+1)=(1-λ)z_j^S(t)+λ max{c_j, p_t-κ} | HOLDS | -- | no |
| 15 | ZIP quotes: Gaussian noise around target, clipped to feasible | HOLDS | -- | no |
| 16 | λ=0.35, κ=1.25, quote_noise=0.90 | HOLDS | -- | no |
| 17 | Buyers never bid above value; sellers never ask below cost | HOLDS | -- | no |
| 18 | Transaction log numbers match CSV | HOLDS | -- | no |
| 19 | Market-type summary numbers match CSV | HOLDS | -- | no |
| 20 | Agent-mix summary numbers match CSV | HOLDS | -- | no |

## Findings

All 20 claims HOLD. No non-HOLDS findings. Individual evidence cited below by claim group.

### Claims 1-2: ZIC quote distributions

- **Claim source (verbatim):** "ZIC buyers and sellers draw feasible quotes: $b_i(t)\sim U[0,v_i]$, $a_j(t)\sim U[c_j,\bar p]$." -- `README.md:24-28`
- **Code evidence (verbatim):**
  ```python
  def feasible_random_bid(value: float, rng: np.random.Generator) -> float:
      return float(rng.uniform(0.0, value))

  def feasible_random_ask(cost: float, max_price: float, rng: np.random.Generator) -> float:
      return float(rng.uniform(cost, max_price))
  ```
  `run.py:115-122`
- **Category:** HOLDS

### Claim 3: Transaction price = midpoint

- **Claim source (verbatim):** "$p_t=\frac{1}{2}\left(\max B_t+\min A_t\right)$" -- `README.md:36-39`
- **Code evidence (verbatim):**
  ```python
  price = 0.5 * (best_bid + best_ask)
  ```
  `run.py:273`
- **Data evidence:** Trade 1: accepted bid=66.284, accepted ask=52.687, price=59.486. Check: (66.284+52.687)/2=59.4855, rounds to 59.486 at 3 d.p. -- `tables/transaction-log.csv:2`
- **Category:** HOLDS

### Claim 4: Arrival probability proportional to active traders

- **Claim source (verbatim):** "Draw one active side with probability proportional to active traders." -- `README.md:127`
- **Code evidence (verbatim):**
  ```python
  buyer_probability = active_buyer_count / (active_buyer_count + active_seller_count)
  if rng.random() < buyer_probability:
  ```
  `run.py:248-249`
- **Category:** HOLDS

### Claim 5: Stale quotes deleted after trade

- **Claim source (verbatim):** "remove the matched buyer and seller, and delete their stale quotes." -- `README.md:132`
- **Code evidence (verbatim):**
  ```python
  active_buyers[best_buyer] = False
  active_sellers[best_seller] = False
  bids = {idx: bid for idx, bid in bids.items() if active_buyers[idx]}
  asks = {idx: ask for idx, ask in asks.items() if active_sellers[idx]}
  ```
  `run.py:293-296`
- **Category:** HOLDS

### Claim 6: σ_p is population std

- **Claim source (verbatim):** "$\sigma_p=\sqrt{\frac{1}{T_p}\sum_{t:p_t\ \mathrm{exists}}(p_t-\bar p_T)^2}$" -- `README.md:71-73`
- **Code evidence (verbatim):**
  ```python
  price_sd = float(trades["Price"].std(ddof=0))
  ```
  `run.py:345`
- **Data evidence:** Verified numerically from `tables/transaction-log.csv`: 8 prices [59.486,77.259,78.796,45.886,69.753,65.835,58.516,85.121], mean=67.58, population std=12.01. Matches README:109. -- `tables/transaction-log.csv:2-9`
- **Category:** HOLDS

### Claims 7-11: Benchmark numbers

- **Claim source (verbatim):** "$Q^{\ast}$ | 7 | Efficient quantity" and "$S^{\ast}$ | 294.00" and "$P^{\ast}$ | [70.00, 72.00]" -- `README.md:105-107`
- **Code evidence:** `competitive_benchmark()` at `run.py:80-112`. Gains for balanced market: [75,64,53,42,31,20,9,-2,-13,-24]. Q*=7 (first 7 positive). S*=75+64+53+42+31+20+9=294.0. lower_candidates=[c_7=66, v_8=70], max=70. upper_candidates=[v_7=75, c_8=72], min=72. P*=[70,72].
- **Data evidence:** `tables/market-type-summary.csv:2`: "Balanced 10 x 10,10,10,7,70.00,72.00" -- HOLDS. All four market P* values verified analytically and match CSV.
- **Category:** HOLDS

### Claims 13-16: ZIP update equations and parameters

- **Claim source (verbatim):** "$z_i^B(t+1)=(1-\lambda)z_i^B(t)+\lambda \min\lbrace v_i,p_t+\kappa\rbrace$" and "$z_j^S(t+1)=(1-\lambda)z_j^S(t)+\lambda \max\lbrace c_j,p_t-\kappa\rbrace$" -- `README.md:82-89`
- **Code evidence (verbatim):**
  ```python
  for buyer in zip_buyers:
      if active_buyers[buyer]:
          target = min(values[buyer], last_price + learning.spread)
          buyer_targets[buyer] = (
              (1.0 - learning.learning_rate) * buyer_targets[buyer]
              + learning.learning_rate * target
          )
          buyer_targets[buyer] = float(np.clip(buyer_targets[buyer], 0.0, values[buyer]))

  for seller in zip_sellers:
      if active_sellers[seller]:
          target = max(costs[seller], last_price - learning.spread)
          seller_targets[seller] = (
              (1.0 - learning.learning_rate) * seller_targets[seller]
              + learning.learning_rate * target
          )
  ```
  `run.py:156-174`
- **Code evidence for parameters:**
  ```python
  @dataclass(frozen=True)
  class LearningConfig:
      learning_rate: float = 0.35
      spread: float = 1.25
      quote_noise: float = 0.9
  ```
  `run.py:43-49`
- **Category:** HOLDS

### Claims 15, 17: ZIP quote noise and feasibility constraints

- **Claim source (verbatim):** "Quotes are noisy draws around these targets, clipped so buyers still satisfy $b_i(t)\leq v_i$ and sellers still satisfy $a_j(t)\geq c_j$." -- `README.md:91-92`
- **Code evidence (verbatim):**
  ```python
  quote = buyer_targets[buyer] + rng.normal(0.0, learning.quote_noise)
  return float(np.clip(quote, 0.0, value))
  ```
  `run.py:189-190`
  ```python
  quote = seller_targets[seller] + rng.normal(0.0, learning.quote_noise)
  return float(np.clip(quote, cost, max_price))
  ```
  `run.py:206-207`
- **Category:** HOLDS

### Claims 18-20: Table cross-check

- **Claim source:** README.md Results tables vs CSV files.
- **Data evidence:** `tables/transaction-log.csv` rows 1-8 match README.md:169-178 exactly (8 trades, same event, value, cost, bid, ask, price, surplus). `tables/market-type-summary.csv` rows match README.md:186-191 exactly (4 rows). `tables/agent-mix-summary.csv` rows match README.md:199-203 exactly (3 rows).
- **Category:** HOLDS

## Cross-cutting patterns

- No cross-cutting failure patterns found. The code is unusually faithful to its prose. Every equation in the Equations section has a direct one-to-one mapping to a function in run.py. Parameter values are read from `LearningConfig` dataclass defaults rather than being hardcoded in scattered locations, eliminating the drift vector.
- The only architectural observation (not a finding): the σ_p formula clarification in README.md:76 ("distinct from pbar, the maximum ask support defined above") proactively addresses the one notation collision that could confuse a reader. That is good writing, not a bullshit pattern.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No findings require remediation. No halt, no surface, no guardrails needed.
1. No violated-invariant tests to write.
2. No honest-fix pass-condition tests to write.
3. No fixes needed. Proceed with other work.
4. If any future run.py edit changes the σ_p denominator from `ddof=0` to `ddof=1`, that would constitute a new DILUTED finding (code would then implement sample std while Equations claim population std). Guard: `assert trades["Price"].std(ddof=0) == session_metrics(...)["Price SD"]`.
5. If any future edit changes `learning_rate`, `spread`, or `quote_noise` defaults in `LearningConfig` without regenerating README.md, that would constitute a DATA DRIFT finding. Guard: regenerate and diff README.md after any `LearningConfig` change.
