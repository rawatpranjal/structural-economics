# Structural Economics Library

## Project Overview
Pedagogical library of computational/structural economics models.
Each model lives in a self-contained folder with `run.py` that auto-generates `README.md` + `figures/`.

## Code Style
- Python 3.11+, JAX for numerical computation
- Type hints on all public functions
- Google-style docstrings
- kebab-case folder names, snake_case file/function names
- Each run.py is self-contained: economics + computation + report generation (~100-300 lines)

## Shared Library
- `lib/` contains shared JAX utilities (grids, discretize, interpolate, vfi, output, plotting)
- Models import from lib/ via `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`
- Do not duplicate utility code in model folders

## Output Standard
Every model `run.py` produces:
- `README.md` — auto-generated via `lib.output.ModelReport`
- `figures/*.png` — matplotlib, 150 DPI, academic style
- `tables/*.csv` — optional, if model has tabular results
- `figures/thumb.png` — 200x150 thumbnail for root catalog

## README Section Order
Overview -> Equations -> Model Setup -> Solution Method -> Results -> Economic Takeaway -> Reproduce -> References

## Do Not
- Delete any files — originals are in `_legacy/`
- Use NumPy where JAX works (prefer `jnp` over `np`)
- Add interactive/Plotly visualizations — matplotlib only
- Create Jupyter notebooks — this repo is `.py` + `.md` only (new code)
- Add dependencies beyond what's in `requirements.txt` without discussion
