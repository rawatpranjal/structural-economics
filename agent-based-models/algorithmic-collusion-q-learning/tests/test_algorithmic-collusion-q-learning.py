"""Faithfulness tests for the algorithmic-collusion-q-learning tutorial.

Generated from
bullshit-detector_algorithmic-collusion-q-learning_2026-05-20.md.

Finding 1 (DILUTED, LOW): the Model Setup prose said "13 evenly spaced
prices between those two prices". np.linspace(p_B, p_M, 13) INCLUDES both
endpoints, so only 11 prices lie strictly between the benchmarks; 2 sit
on them. The grid construction is correct; the word "between" was wrong.
The honest fix rewords the prose to "spanning from ... to ... with both
endpoints included".

run.py guards main() under __main__, so importing solve_benchmarks/Params
does not run the whole tutorial.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parents[1]


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "acql_run", TUTORIAL_DIR / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["acql_run"] = module
    spec.loader.exec_module(module)
    return module


def _grid_and_benchmarks():
    mod = _load_run_module()
    bench = mod.solve_benchmarks(mod.Params())
    return bench.grid, bench.bertrand_price, bench.monopoly_price


# --- Finding 1: violated-invariant test --------------------------------------
# The grid construction is correct: the core linspace includes both
# benchmark endpoints, so exactly 11 grid points lie strictly between them
# (not 13, as the old prose "13 prices between" implied). This invariant
# describes the real grid and stays TRUE after the prose fix.

def test_violated_invariant_only_11_prices_strictly_between():
    grid, p_b, p_m = _grid_and_benchmarks()
    n_strictly_between = sum(1 for g in grid if p_b < g < p_m)
    assert n_strictly_between == 11


def test_violated_invariant_benchmarks_are_on_the_grid():
    grid, p_b, p_m = _grid_and_benchmarks()
    assert np.isclose(grid[1], p_b, atol=1e-6)
    assert np.isclose(grid[-2], p_m, atol=1e-6)


# --- Finding 1: honest-fix test ----------------------------------------------
# FAILS on pre-fix prose ("13 ... between those two prices"); PASSES once
# the prose says the endpoints are included.

def test_honest_fix_readme_prose_says_endpoints_included():
    readme = (TUTORIAL_DIR / "README.md").read_text()
    assert "between those two prices" not in readme
    assert "endpoints included" in readme
