"""Faithfulness tests for dcegm-retirement-saving tutorial.

Audit: bullshit-detector_dcegm-retirement-saving_2026-05-20.md
Finding F1 (DILUTED LOW: "centered at 2.8" reads as the arithmetic mean of a
lognormal whose mean is 3.06; 2.8 is the median).

run.py runs the full solver in main(); claims are tested against the run.py
source text and the generated README rather than by importing.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_PY = (HERE.parent / "run.py").read_text()
README = (HERE.parent / "README.md").read_text()

INITIAL_ASSET_MEAN = 2.8
INITIAL_ASSET_SIGMA = 0.42


# --- Finding 1: 2.8 is the lognormal median, not its arithmetic mean -------

def test_f1_violated_invariant_2p8_is_the_median_not_the_mean():
    # rng.lognormal(mean=log(2.8), sigma) -> median exp(log 2.8) = 2.8,
    # arithmetic mean exp(log 2.8 + sigma^2/2) != 2.8.
    median = np.exp(np.log(INITIAL_ASSET_MEAN))
    arith_mean = INITIAL_ASSET_MEAN * np.exp(INITIAL_ASSET_SIGMA ** 2 / 2)
    assert abs(median - 2.8) < 1e-10
    assert abs(arith_mean - 2.8) > 0.2  # mean is ~3.06, materially above 2.8


def test_f1_honest_fix_prose_calls_2p8_a_median():
    # The misleading "centered at 2.8" wording is gone; 2.8 is named the median.
    assert "centered at 2.8" not in README
    assert "centered at" not in RUN_PY
    low = README.lower()
    assert "median" in low
    # The arithmetic mean is now stated alongside the median.
    assert "arithmetic mean 3.06" in low
