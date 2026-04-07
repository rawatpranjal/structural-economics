# Dynare DSGE Models

## Models in this section
Each subfolder wraps a Dynare `.mod` file with a Python `run.py` that documents
the model, shows calibration, and generates README + IRF figures.

## Common patterns
- `.mod` files contain model equations in Dynare syntax (kept as-is)
- `run.py` parses the .mod file for documentation, loads pre-computed results
- IRF (impulse response function) plots are the primary visualization
- Models are log-linearized around steady state

## Key economics
- RBC and New Keynesian DSGE models
- News shocks (anticipated future shocks)
- Perturbation methods (1st and 2nd order)
