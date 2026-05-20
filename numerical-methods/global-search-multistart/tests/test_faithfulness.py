"""TDD faithfulness tests for global-search-multistart.

Derived from bullshit-detector_global-search-multistart_2026-05-20.md.

For each finding there is a pair of tests:
- ``violated_invariant``: encodes the buggy behaviour. It PASSES on the
  pre-fix code (proving the bug is real) and FAILS once the bug is fixed.
- ``honest_fix``: encodes the faithful behaviour. It FAILS on the pre-fix
  code and PASSES once the tutorial is fixed.

The tests do not import ``run.py`` (importing it would execute the whole
tutorial). Instead they read the generated artifacts and the ``run.py``
source text, and they re-derive the single-start convergence directly.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

TUT = Path(__file__).resolve().parents[1]
RUN_PY = (TUT / "run.py").read_text()
README = (TUT / "README.md").read_text()
METHOD_CSV = pd.read_csv(TUT / "tables" / "method_comparison.csv")
MULTI_CSV = pd.read_csv(TUT / "tables" / "multistart_results.csv")


# ---------------------------------------------------------------------------
# Shared profit model (kept in sync with run.py calibration).
# ---------------------------------------------------------------------------
A_L, b_L = 10.0, 5.0
A_H, b_H = 8.0, 1.0
c = 0.5
lam = 0.6
P_LO, P_HI = c + 1e-3, 8.0


def _profit(p):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    d_l = np.maximum(0.0, A_L - b_L * p)
    d_h = np.maximum(0.0, A_H - b_H * p)
    out = (p - c) * (lam * d_l + (1 - lam) * d_h)
    return float(out.item()) if out.size == 1 else out


def _neg_profit(p):
    return -_profit(p[0]) if np.ndim(p) > 0 else -_profit(p)


def _lbfgsb_from(p0):
    res = minimize(_neg_profit, x0=np.array([p0]),
                   method="L-BFGS-B", bounds=[(P_LO, P_HI)])
    p_final = float(res.x[0])
    return p_final, _profit(p_final)


def _p0_single():
    """The single-start initial price hardcoded in run.py."""
    m = re.search(r"p0_single\s*=\s*([0-9.]+)", RUN_PY)
    assert m, "p0_single assignment not found in run.py"
    return float(m.group(1))


# ===========================================================================
# Finding 1: single-start L-BFGS-B must land at the LOCAL peak, not global.
# ===========================================================================
def test_f1_violated_invariant_single_start_finds_global():
    """PASSES on buggy code: single-start L-BFGS-B reports the global peak."""
    row = METHOD_CSV.iloc[0]
    assert row["Method"] == "Single-start L-BFGS-B"
    assert row["Found global?"] == "yes"


def test_f1_honest_fix_single_start_misses_global():
    """PASSES only after p0_single lands genuinely in the low basin."""
    row = METHOD_CSV.iloc[0]
    assert row["Method"] == "Single-start L-BFGS-B"
    assert row["Found global?"] == "no"
    # The single start must converge to the low-price peak (~1.603), not 4.25.
    assert float(row["Estimated optimum"]) < 2.0
    # The hardcoded p0_single must itself be inside the empirical low basin.
    p_final, _ = _lbfgsb_from(_p0_single())
    assert p_final < 2.0, f"p0_single converges to {p_final}, not the low peak"


# ===========================================================================
# Finding 2: basin boundary is the empirical ~1.5, not the kink p=2.0.
# ===========================================================================
def test_f2_violated_invariant_kink_claimed_as_boundary():
    """PASSES on buggy code: README equates the basin boundary with the kink."""
    assert "Starts below the kink at $p_L^{\\max} = 2.00$ converge to the low peak" in README


def test_f2_honest_fix_boundary_matches_empirical_data():
    """PASSES only after the prose stops equating the boundary with the kink.

    The committed multistart CSV proves the L-BFGS-B basin boundary is near
    1.5: there exist starts below the kink (p<2.0) that land high-price.
    """
    below_kink_high = MULTI_CSV.query("`Starting price` < 2.0 and Basin == 'high-price'")
    assert not below_kink_high.empty, "data must show below-kink starts going high"
    # The faithful README must NOT claim every below-kink start goes low.
    assert "Starts below the kink at $p_L^{\\max} = 2.00$ converge to the low peak" not in README
    # The README must name the empirical boundary near 1.5.
    assert "1.5" in README


# ===========================================================================
# Finding 3: the local/global gap and its percentage must be consistent.
# ===========================================================================
def test_f3_violated_invariant_zero_gap_with_30_percent():
    """PASSES on buggy code: a 0.000 gap is paired with a hardcoded 30 percent."""
    assert "$-0.000$" in README or "$0.000$" in README
    assert "30 percent profit improvement" in README


def test_f3_honest_fix_gap_and_percent_consistent():
    """PASSES only when the stated gap and percentage agree arithmetically."""
    m = re.search(r"gap between local and global on this calibration is "
                  r"\$([0-9.]+)\$, a ([0-9]+) percent", README)
    assert m, "gap/percentage sentence not found in faithful form"
    gap = float(m.group(1))
    pct = float(m.group(2))
    assert gap > 0.5, "the gap must be the genuine non-zero local/global gap"
    # gap as a percentage improvement over the local profit (~4.136).
    local_profit = float(METHOD_CSV.iloc[0]["Profit"])
    expected_pct = gap / local_profit * 100.0
    assert pct == pytest.approx(expected_pct, abs=2.0)


# ===========================================================================
# Finding 4: method numbering must not conflict between sections.
# ===========================================================================
def test_f4_violated_invariant_conflicting_method_1():
    """PASSES on buggy code: 'Method 1' names two different methods."""
    assert README.count("### Method 1: Multi-start") == 1
    assert README.count("### Method 1: Single-start") == 1


def test_f4_honest_fix_no_method_numbering_conflict():
    """PASSES only when no method number labels two different methods."""
    labels = re.findall(r"### Method (\d+): (.+)", README)
    by_number = {}
    for num, name in labels:
        by_number.setdefault(num, set()).add(name.strip())
    for num, names in by_number.items():
        assert len(names) == 1, f"Method {num} labels conflicting methods: {names}"
