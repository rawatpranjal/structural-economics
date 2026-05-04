# Game Theory Models

## Models in this section
Low-code computational game theory examples focused on equilibrium concepts
and numerical checks rather than empirical IO applications.

## Common patterns
- Enumerate finite games before using heavier methods
- Report unilateral-deviation or fixed-point residuals
- Prefer closed-form benchmarks plus small numerical checks
- Avoid external game-theory solver dependencies
- Static Cournot and best-response dynamics live together in
  `game-theory/static-games`; do not split off a separate best-response
  tutorial unless it teaches a different game or convergence issue.

## Key economics
- Pure and mixed Nash equilibrium in normal-form games
- Cournot oligopoly and best-response dynamics as fixed-point iteration
- Bayesian Nash equilibrium in first-price auctions
- Quantal response equilibrium as noisy best response
