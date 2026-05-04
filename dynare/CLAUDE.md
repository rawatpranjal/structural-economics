# Dynare DSGE Models

## Models in this section
Each subfolder wraps a Dynare `.mod` file with a Python `run.py` that documents
the model, shows calibration, and generates README + IRF figures.

## Catalog role
- Treat Dynare examples as part of the Macroeconomics block.
- Present them as DSGE and shock-propagation tutorials, not as a tooling-first
  Dynare section.
- Keep titles tied to the economic model or shock mechanism: RBC, New
  Keynesian determinacy, and news shocks.
- Pure stochastic-process tutorials belong in `time-series/`; AR persistence
  is no longer part of the active Dynare section.

## Common patterns
- `.mod` files contain model equations in Dynare syntax (kept as-is)
- `run.py` parses the .mod file for documentation, loads pre-computed results
- IRF (impulse response function) plots are the primary visualization
- Models are log-linearized around steady state

## Key economics
- RBC and New Keynesian DSGE models
- News shocks (anticipated future shocks)
- Perturbation methods (1st and 2nd order)
