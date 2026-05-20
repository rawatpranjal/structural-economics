"""Faithfulness tests for bayesian-dsge-hmc tutorial.

Audit: bullshit-detector_bayesian-dsge-hmc_2026-05-20.md
Findings F2 (MISLABELED: "parallel" chains), F3 (DILUTED HIGH: Takeaway
compares NUTS to baseline on equal-sample-count, not equal-wall-time),
F4 (DILUTED LOW: "ESS on a log scale" sentence placement).

run.py runs the full estimation in main(); claims are tested against the
run.py source text and the generated README rather than by importing.
"""

import inspect
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_PY = (HERE.parent / "run.py").read_text()
README = (HERE.parent / "README.md").read_text()


# --- Finding 2: chains run sequentially, not in parallel -------------------

def test_f2_violated_invariant_chains_run_in_sequential_loop():
    # run_nuts dispatches chains with a Python for-loop; no vmap/pmap/lax.map.
    import importlib.util

    spec = importlib.util.spec_from_file_location("dsge_run", HERE.parent / "run.py")
    mod = importlib.util.module_from_spec(spec)
    # Only need the source of run_nuts; load module body without main().
    src = RUN_PY
    assert "for c in range(num_chains):" in src
    assert "jax.vmap" not in src.split("def run_nuts")[1].split("\ndef ")[0]
    assert "pmap" not in src.split("def run_nuts")[1].split("\ndef ")[0]


def test_f2_honest_fix_prose_does_not_call_chains_parallel():
    assert "run in parallel" not in README
    assert "run independently" in README


# --- Finding 3: Takeaway ESS comparison axis -------------------------------

def test_f3_violated_invariant_unqualified_per_compute_unit_gone():
    # The original unqualified "per compute unit" framing must not survive.
    assert "per compute unit" not in README


def test_f3_honest_fix_takeaway_qualifies_the_comparison():
    low = README.lower()
    # The ESS advantage is qualified as per-sample / equal-draw, and the
    # wall-time caveat (RW-MH faster per second) is stated.
    assert "per raw sample" in low
    assert "wall" in low and "second" in low


# --- Finding 4: "ESS is on a log scale" placement --------------------------

def test_f4_violated_invariant_bare_ess_log_sentence_gone():
    assert "ESS is on a log scale." not in README


def test_f4_honest_fix_sentence_attributes_log_axis_to_figure():
    assert "figure y-axis uses a log scale" in README.lower()
