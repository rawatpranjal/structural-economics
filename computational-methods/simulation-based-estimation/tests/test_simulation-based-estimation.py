"""Faithfulness tests for the simulation-based-estimation tutorial.

Audit: bullshit-detector_simulation-based-estimation_2026-05-20.md.

Finding 1 (MISLABELED): the indirect-inference residuals table called
residual_table with (ii, ii) as the msm / ii positional args, then queried
"Estimator == 'MSM'". The signature names the 2nd arg msm and the 3rd ii, so
passing a real (msm, ii) pair and reading the natural 'MSM' row would yield
MSM data, not II data.

Finding 2 (DILUTED): Results prose claimed the ABC criterion is "on the same
scale as the MSM and II numbers". MSM and ABC share 5 economic moments and the
same scale vector; II uses 6 auxiliary statistics with a different scale, so
the II criterion is not on the same scale.
"""

import importlib.util
from pathlib import Path

import numpy as np

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def _load_run():
    spec = importlib.util.spec_from_file_location("sbe_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Finding 1 -------------------------------------------------------------

def test_ii_table_call_passes_ii_as_second_arg():
    """Violated-invariant: passes on the buggy (ii, ii) call for the II table.

    Fails once the II table call passes a real msm result as the 2nd arg.
    """
    src = RUN_PY.read_text()
    ii_block = src[src.index('"tables/indirect-inference-residuals.csv"'):]
    ii_block = ii_block[: ii_block.index(")")]
    # buggy form: ii passed twice, then the MSM row queried
    assert "ii,\n            ii,\n" in ii_block
    assert 'query("Estimator == \'MSM\'")' in ii_block


def test_ii_table_uses_correct_estimator_label():
    """Honest-fix: passes once the II table queries the 'Indirect inference' row.

    Fails on current buggy code, which queries 'MSM' for the II table.
    """
    src = RUN_PY.read_text()
    ii_block = src[src.index('"tables/indirect-inference-residuals.csv"'):]
    ii_block = ii_block[: ii_block.index("),\n    )")]
    assert "msm,\n            ii,\n" in ii_block
    assert 'query("Estimator == \'Indirect inference\'")' in ii_block


def test_residual_table_labels_third_arg_as_indirect_inference():
    """Honest-fix support: the 3rd arg of residual_table is the II result.

    Passing distinct msm / ii dicts, the 'Indirect inference' row must carry
    the third argument's simulated stats.
    """
    sbe = _load_run()
    dummy_msm = {"simulated_stats": [2.0], "residual": [0.2]}
    dummy_ii = {"simulated_stats": [3.0], "residual": [0.3]}
    df = sbe.residual_table(["X"], np.array([1.0]), dummy_msm, dummy_ii)
    row = df.query("Estimator == 'Indirect inference'").iloc[0]
    assert row["Simulated at estimate"] == 3.0


# --- Finding 2 -------------------------------------------------------------

def test_results_prose_claims_same_scale():
    """Violated-invariant: passes on the buggy 'same scale' claim.

    Fails once the prose drops the tripartite same-scale assertion.
    """
    src = RUN_PY.read_text()
    assert "on the same scale as the MSM and II numbers" in src


def test_results_prose_disclaims_ii_scale():
    """Honest-fix: passes once the prose disclaims that II uses a different scale.

    Fails on current buggy prose.
    """
    src = RUN_PY.read_text()
    assert "MSM and ABC" in src and "II uses a different" in src
