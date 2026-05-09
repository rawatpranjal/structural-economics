# Linearized DSGE Models

## Models in this section
Each subfolder is a self-contained DSGE tutorial that log-linearizes a small
rational-expectations model around its deterministic steady state and solves
the resulting linear system.

Three tutorials (`rbc`, `nkdsge`, `assetNews`) use the method of undetermined
coefficients - they hand-derive the closed-form linear policy because the
state-space is small enough to make the algebra visible. Each one cross-checks
its closed-form coefficients against Klein-style generalized Schur (QZ)
decomposition via `lib.perturbation.solve_klein`; agreement to ~1e-15 is the
expected pass condition.

The fourth tutorial (`rbc-with-labor`) uses Klein QZ as the primary solver,
because adding endogenous labor pushes the system past what hand-derived
coefficients comfortably handle.

## Catalog role
- Treat these as part of the Macroeconomics block.
- Present them as DSGE and shock-propagation tutorials, not as a tooling
  section. There is no Dynare/Octave/MATLAB invocation; the `.mod` files are
  textbook specifications kept alongside `run.py` for documentation.
- Keep titles tied to the economic model or shock mechanism.
- Pure stochastic-process tutorials belong in `time-series/`.

## Common patterns
- `run.py` reads the `.mod` spec text for the Model Setup section but does
  not execute it.
- The shared `lib/perturbation.py` provides Klein-style QZ via
  `solve_klein(A, B, n_predetermined)` - used both as primary solver and as
  cross-check.
- IRF (impulse response function) plots are the primary visualization.
- Models are log-linearized around steady state; nonlinear perfect-foresight
  benchmarks are shown where they reveal local-approximation accuracy.

## Key economics
- RBC and New Keynesian DSGE models
- News shocks (anticipated future shocks)
- First-order perturbation; Blanchard-Kahn determinacy

## Reference
- Klein, P. (2000). "Using the Generalized Schur Form to Solve a
  Multivariate Linear Rational Expectations Model." *Journal of Economic
  Dynamics and Control*, 24(10), 1405-1423.
- Villemot, S. (2011). "Solving Rational Expectations Models at First
  Order: What Dynare Does." Dynare Working Paper 2, CEPREMAP. - Confirms the
  same generalized Schur algorithm is what Dynare uses for `stoch_simul,
  order=1`.
