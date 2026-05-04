# Optimal Control Models

## Models in this section
Continuous-time optimization: Hamiltonian mechanics, phase diagrams,
finite difference methods for PDEs, and Pontryagin's maximum principle.

## Catalog role
- Treat optimal-control tutorials as the continuous-time methods wing of the
  Macroeconomics block unless a future tutorial is clearly non-macro.
- Order examples from phase-diagram intuition to shooting, Pontryagin, and HJB
  discretization.
- Titles should name the economic object and the control idea together, such as
  Ramsey saddle paths, shooting, Pontryagin cake eating, or finite-difference
  HJB methods.

## Common patterns
- ODE integration via `scipy.integrate.solve_ivp`
- Phase plane analysis with vector fields
- Finite difference schemes for HJB equations
- Analytical vs numerical solution comparison

## Key economics
- Continuous-time cake eating and optimal growth (Ramsey)
- Saddle-path dynamics and transversality conditions
- Euler equation and co-state variables
