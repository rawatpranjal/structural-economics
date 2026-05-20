"""Faithfulness tests for auction-valuation-recovery tutorial.

Audit: bullshit-detector_auction-valuation-recovery_2026-05-20.md
Findings F1 (DILUTED: rank-convention precision gap),
F2 (DILUTED: bid-level DataFrame named ambiguously).
"""

import importlib.util
import inspect
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_PATH = HERE.parent / "run.py"

_spec = importlib.util.spec_from_file_location("auction_run", RUN_PATH)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

README = (HERE.parent / "README.md").read_text()
MAIN_SRC = inspect.getsource(run.main)


# --- Finding 1: empirical CDF uses 1-indexed ranks -------------------------

def test_f1_violated_invariant_cdf_is_one_indexed():
    # Code returns rank/N with 1-indexed ranks: minimum maps to 1/N, not 0.
    result = run.empirical_cdf_at_sample(np.array([1.0, 2.0, 3.0]))
    assert result[0] == 1 / 3
    assert result[-1] == 1.0


def test_f1_honest_fix_readme_states_rank_convention():
    assert "rank(b_i)" in README
    assert "1-indexed" in README


# --- Finding 2: trimmed-share denominator is the bid-level DataFrame -------

def test_f2_violated_invariant_trim_denominator_is_bid_count():
    bids_df = run.simulate_auctions(3000, 4, 20260508)
    # The trimmed-share denominator counts bids (12000), not auctions (3000).
    assert len(bids_df) == 3000 * 4


def test_f2_honest_fix_denominator_variable_renamed():
    # Ambiguous name 'auctions' for a bid-level frame is gone; 'all_bids' used.
    assert "len(all_bids)" in MAIN_SRC
    assert "len(auctions)" not in MAIN_SRC
