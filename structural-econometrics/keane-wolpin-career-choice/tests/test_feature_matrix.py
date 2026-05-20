"""Findings 3 and 4: feature_matrix normalization disclosure and empty-path shape.

Finding 3: the Equations section documents phi with raw state variables, but
feature_matrix normalizes every input. The README must disclose the
normalization so the fitted coefficient vector is interpretable.

Finding 4: the empty-state guard returns a (0, 11) array while the populated
path produces 12 columns. The two paths must agree on column count.
"""

import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TUTORIAL_DIR))

import run  # noqa: E402
from run import CareerPrimitives, feature_matrix  # noqa: E402

README = TUTORIAL_DIR / "README.md"


def test_violated_invariant_feature_matrix_normalizes_schooling():
    """Code returns normalized schooling, not the raw E value."""
    assert feature_matrix([(10, 0, 0)], 0, CareerPrimitives())[0, 1] != 10


def test_honest_fix_readme_discloses_normalization():
    """README basis text must disclose that phi uses normalized inputs."""
    src = Path(run.__file__).read_text()
    idx = src.index("the basis vector is")
    basis_block = src[idx:idx + 1200]
    assert "normaliz" in basis_block.lower()


def test_violated_invariant_empty_path_column_count():
    """The empty-state guard must produce the same column count as populated."""
    populated_cols = feature_matrix([(10, 0, 0)], 0, CareerPrimitives()).shape[1]
    empty_cols = feature_matrix([], 0, CareerPrimitives()).shape[1]
    assert empty_cols == populated_cols


def test_honest_fix_empty_path_shape_is_12():
    """The empty-state guard must return exactly (0, 12)."""
    assert feature_matrix([], 0, CareerPrimitives()).shape == (0, 12)
