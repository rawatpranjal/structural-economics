"""Faithfulness tests for the aiyagari-hact DT-vs-CT comparison.

Covers the four findings from
bullshit-detector_aiyagari-hact_2026-05-20.md. Each finding gets two tests:

  * a violated-invariant test: PASSES on the original buggy artifact,
    FAILS once the finding is honestly fixed.
  * an honest-fix test: FAILS on the original buggy artifact,
    PASSES once the finding is honestly fixed.

run.py executes the whole tutorial on import, so the comparison-table
prose and the DT mass computation are tested against the static source
text of README.md and run.py rather than by importing run.py.
"""

import re
from pathlib import Path

TUTORIAL = Path(__file__).resolve().parents[1]
README = TUTORIAL / "README.md"
RUN_PY = TUTORIAL / "run.py"
CMP_CSV = TUTORIAL / "tables" / "dt-vs-ct-comparison.csv"


def _cmp_row(label: str) -> tuple[float, float, float]:
    """Return (HACT, Discrete-time, Absolute gap) for a comparison-table row."""
    for line in CMP_CSV.read_text().splitlines():
        # The MPC row label is quoted because it contains a comma.
        stripped = line.strip().strip('"')
        if stripped.startswith(label):
            tail = line.rsplit(",", 3)[-3:]
            return tuple(float(x) for x in tail)
    raise AssertionError(f"row {label!r} not found in {CMP_CSV}")


def _comparison_description() -> str:
    """The prose paragraph that summarises the DT-vs-CT comparison table."""
    text = README.read_text()
    # The paragraph sits between the HACT table and the side-by-side table.
    para = [
        ln for ln in text.splitlines()
        if "The two solvers agree" in ln
    ]
    assert para, "comparison-table description paragraph not found in README"
    return para[0]


# ---------------------------------------------------------------------------
# Finding 1: "wealth Gini agrees to three decimal places"
# ---------------------------------------------------------------------------

def test_finding1_violated_invariant_gini_three_decimals():
    """Buggy prose claims three-decimal agreement; the Gini gap is far larger.

    PASSES on the buggy README (the false 'three decimal places' phrasing is
    present). FAILS once the prose is corrected to an honest bound.
    """
    desc = _comparison_description()
    assert "three decimal places" in desc


def test_finding1_honest_fix_gini_bound_matches_table():
    """Honest prose must state a bound the comparison table actually meets.

    FAILS on the buggy README ('three decimal places' implies gap < 0.001
    while the table gap is 0.0114). PASSES once the prose drops the false
    claim and any stated bound is >= the real gap.
    """
    _, _, gap = _cmp_row("Wealth Gini")
    desc = _comparison_description()
    assert "three decimal places" not in desc
    # Any numeric bound the prose states for the Gini must not be tighter
    # than the actual gap printed in the table.
    for bound in re.findall(r"0\.0\d{2,4}", desc):
        if float(bound) < 0.01:  # a Gini-scale bound, not a pp figure
            assert float(bound) >= gap


# ---------------------------------------------------------------------------
# Finding 2: "mass at borrowing limit agrees to one percentage point"
# ---------------------------------------------------------------------------

def test_finding2_violated_invariant_mass_one_pp():
    """Buggy prose claims one-pp agreement on the mass at the limit.

    PASSES on the buggy README ('one percentage point' phrasing present).
    FAILS once the prose is corrected.
    """
    desc = _comparison_description()
    assert "one percentage point" in desc


def test_finding2_honest_fix_mass_claim_matches_table():
    """Honest prose must not claim tighter mass agreement than the table.

    FAILS on the buggy README. PASSES once the false 'one percentage point'
    claim is gone and the prose is consistent with the table's mass gap.
    """
    _, _, gap_frac = _cmp_row("Mass at borrowing limit")
    gap_pp = gap_frac * 100.0
    desc = _comparison_description()
    assert "one percentage point" not in desc
    # If the prose states a pp bound for the mass, it must cover the gap.
    for m in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*percentage point", desc):
        assert float(m) >= gap_pp - 0.5


# ---------------------------------------------------------------------------
# Finding 3: "two Ginis agree to within 0.011" vs table value 0.0114
# ---------------------------------------------------------------------------

def test_finding3_violated_invariant_rounded_gini_gap():
    """Figure prose rounds the Gini gap to 0.011 while the table says 0.0114.

    PASSES on the buggy README (the rounded-down 0.011 string is present).
    FAILS once the prose reports the gap at the table's precision.
    """
    text = README.read_text()
    assert "agree to within $0.011$" in text


def test_finding3_honest_fix_gini_gap_full_precision():
    """Figure prose must report the Gini gap at the comparison-table precision.

    FAILS on the buggy README (0.011 != 0.0114). PASSES once the figure prose
    reports the same value the table column shows.
    """
    _, _, gap = _cmp_row("Wealth Gini")
    text = README.read_text()
    m = re.search(r"two Ginis agree to within \$([0-9.]+)\$", text)
    assert m, "figure prose about the Gini gap not found"
    assert float(m.group(1)) == round(gap, 4)


# ---------------------------------------------------------------------------
# Finding 4: HACT and DT "mass at borrowing limit" use different objects
# ---------------------------------------------------------------------------

def test_finding4_violated_invariant_dt_mass_is_point_mass():
    """Buggy DT mass is the bare node-0 probability, not an interval integral.

    The HACT side integrates the density over a <= a_min + 0.02, but the DT
    side reads only dt_marginal[0]. PASSES on the buggy run.py.
    """
    src = RUN_PY.read_text()
    assert "dt_mass_at_constraint = float(dt_marginal[0])" in src


def test_finding4_honest_fix_dt_mass_uses_same_interval():
    """The DT mass must be summed over the same [a_min, a_min + 0.02] window.

    FAILS on the buggy run.py (bare dt_marginal[0]). PASSES once the DT mass
    is an interval sum over the same physical window the HACT side uses, so
    the comparison-table row compares like for like.
    """
    src = RUN_PY.read_text()
    assert "dt_mass_at_constraint = float(dt_marginal[0])" not in src
    # The fixed line must restrict the DT marginal to the same window.
    m = re.search(r"dt_mass_at_constraint\s*=\s*float\(\s*np\.sum\("
                  r"\s*dt_marginal\[(.+?)\]\s*\)\s*\)", src)
    assert m, "DT mass must be an np.sum over a sliced dt_marginal"
    window = m.group(1)
    assert "a_grid_dt" in window and "0.02" in window


def test_finding4_honest_fix_mass_objects_are_comparable():
    """After the fix the two mass rows must agree within a comparable bound.

    Both the HACT and DT mass numbers should measure the fraction of
    households within 0.02 of the borrowing limit. FAILS on the buggy CSV
    (point mass vs interval integral, gap 0.0296). PASSES once both sides
    integrate the same window.
    """
    hact, dt, gap = _cmp_row("Mass at borrowing limit")
    assert abs(hact - dt) == round(gap, 4)
    assert gap < 0.015
