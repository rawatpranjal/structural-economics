# Computational Finance

## Models in this section
Short computational finance tutorials converted from the old
`rawatpranjal/computational-finance` notebooks and enriched with standard
finance references.

## Catalog role
- Keep this as a compact finance chapter, not a trading toolkit.
- Favor canonical objects: bond pricing, yield curves, return predictability,
  random walks, efficient-market diagnostics, and mean-variance portfolios.
- Be explicit when a tutorial is a teaching analogue rather than a replication
  of a paper that needs unavailable CRSP or zero-coupon bond data.

## Data policy
- Active tutorials must be reproducible without network access.
- Do not call `yfinance`, Treasury, or FRED at runtime.
- Bundled Treasury data are a static teaching snapshot from the source repo.
- Original notebooks live in `_legacy/computational-finance/` for provenance.

## Common patterns
- Use `lib.output.ModelReport` for generated README files.
- Use matplotlib only.
- Keep examples short and inspectable.
- Put data cautions in the generated report when interpretation depends on a
  static or simplified dataset.

## Key economics
- Yield to maturity is the internal rate that prices promised cash flows.
- Treasury CMT data are interpolated par-yield-curve rates, not raw traded
  transaction yields.
- Term-structure regressions test predictability, not mechanical arbitrage.
- Efficient-market tests are joint tests of market efficiency and the expected
  return model.
- Mean-variance frontiers depend on covariance estimates and are sensitive to
  inputs.
