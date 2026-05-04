# Tutorial Style Guide

This file records soft writing conventions for `Computational Economics`
tutorials.
It complements, but does not replace, `CLAUDE.md` and `GLOSSARY.md`.

- `CLAUDE.md` is the contract: folder structure, required files, generated
  report sections, reproducibility, and validation expectations.
- `GLOSSARY.md` defines recurring terms, notation, and catalog language.
- `STYLE_GUIDE.md` gives preferred prose and presentation style for future
  tutorial reports and catalog rows.

These are conventions, not hard validation rules. A tutorial can depart from
them when the model, estimator, or source material genuinely calls for it.

## Reader

Write for graduate students and researchers who have already made a first pass
through quantitative-economics or QuantEcon-style material. The reader knows the
standard vocabulary of dynamic programming, equilibrium, estimation, and
simulation, but still benefits from clear notation, explicit economic objects,
and short computational explanations.

Do not write as if the reader is only learning Python, NumPy, or a solver. The
code matters because it computes an economic object.

## Voice

Use plain English and economic language. Start from the economic question, then
explain the computational method as the way the tutorial answers it.

Prefer:

- "The Markov chain changes continuation values."
- "The IV estimate corrects price endogeneity."
- "The policy function shows how borrowing constraints shape saving."

Avoid unnecessary solver-first framing:

- "We use scipy.optimize to solve this problem."
- "This tutorial demonstrates a matrix inversion."
- "The main point is the implementation details."

Name the method when it is useful, but keep the economic object in front.

## Report Structure

Generated tutorial reports should usually follow this reader-facing flow:

1. Overview
2. Equations
3. Model Setup
4. Solution Method
5. Results
6. Takeaway
7. References

`Reproduce` is optional. Every active tutorial still has to regenerate from
`python run.py`, but the generated report does not need a visible `Reproduce`
section when that command would be boilerplate.

Use the sections consistently:

- `Overview`: state the economic object, question, and why the example matters.
- `Equations`: present the model, estimator, moments, or equilibrium
  restrictions that drive the tutorial.
- `Model Setup`: list primitives, calibration, data objects, grids, simulation
  settings, and maintained assumptions.
- `Solution Method`: explain the algorithm, estimator, equilibrium search, or
  simulation workflow.
- `Results`: interpret figures, tables, policies, estimates, residuals, and
  diagnostics.
- `Takeaway`: give a short economic lesson tied to what the code demonstrated.
- `Reproduce`: when included, show the local command, usually `python run.py`.
- `References`: list the canonical source material used by the tutorial.

## Pseudocode

Put pseudocode in `Solution Method`, close to the prose that explains the
algorithm. Use it when the tutorial has a real procedure that is clearer as
steps than as paragraphs.

Good pseudocode should:

- use short Markdown code fences or compact numbered algorithm blocks;
- name inputs and outputs;
- align symbols with the `Equations` section;
- show economically meaningful steps, not implementation-only bookkeeping;
- keep loops, convergence checks, and fixed-point updates visible when they are
  part of the idea.

Avoid pseudocode that copies Python line by line. Do not include file paths,
plotting calls, import details, or data-frame cleanup unless those details are
the point of the tutorial.

## Equations

Define notation near first use. If a symbol is overloaded across economics,
state the local meaning in the tutorial rather than relying on memory.

Keep tutorial-specific notation local. Use `GLOSSARY.md` for shared conventions,
but do not force a global symbol choice when the source literature uses a
different standard notation.

Prefer representative equations over exhaustive derivations. The `Equations`
section should give the reader the mathematical object that the code solves,
estimates, simulates, or checks. Long proofs and derivations belong only when
they are central to the tutorial.

## Figures And Tables

Introduce figures and tables by describing the economic object first, then the
computational diagnostic.

For example:

- first say that a policy function is showing saving behavior by asset state;
- then say that kinks or residuals reveal borrowing constraints or numerical
  accuracy.

Visible figure captions are optional. If the surrounding Results prose already
interprets the figure, omit the caption and keep a concise `alt` string for the
image. If a caption is shown, make it interpretive but not long: it should tell
the reader what the visual means, not only what variables appear on the axes.

Tables should have concise titles and column names. Round numbers enough for
comparison, but keep enough precision for residuals, moment errors, or
equilibrium checks to be auditable.

## Takeaways

Keep takeaways short, economic, and tied to the executed code. The best
takeaways say what the tutorial teaches that the reader can reuse in a nearby
model.

Prefer one compact paragraph unless the tutorial naturally has two or three
distinct lessons. Avoid broad claims that were not demonstrated by the run.

## Catalog Rows

Root catalog rows should help readers choose a tutorial quickly. Use:

- a descriptive title naming the economic object or topic;
- an economic teaching goal, not just a method description;
- a compact method label;
- one clear key insight.

The title should not be a one-word method name when a concrete economic title is
available. The key insight should be specific enough to distinguish the tutorial
from neighboring entries.

## Boundaries

This guide standardizes future writing. It does not require retroactive edits to
generated tutorial `README.md` files, tutorial code, figures, or `lib/output.py`.
When changing the output generator or existing generated reports, follow the
separate contract in `CLAUDE.md` and validate the catalog.
