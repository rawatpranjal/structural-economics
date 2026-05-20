"""Faithfulness regression tests for the convex-time-budget-present-bias tutorial.

These tests pin the two findings raised by the 2026-05-20 bullshit-detector
audit:

- Finding 1 (FALSE): the Takeaway claimed NLS "lets the Stone-Geary minima be
  jointly estimated", but ``fit_nls`` estimates only ``(beta, delta, alpha)``
  and the demand function hard-codes ``omega_1 = omega_2 = 0``.
- Finding 2 (DILUTED): the identification figure was captioned "profile
  log-likelihood" with no qualification, but ``neg_ll_concentrated`` in the
  profile loop uses a plain Gaussian sum-of-squares criterion that ignores the
  Tobit corner-censoring terms used by the point estimator.

Both fixes are prose corrections: the code (and every published number) is
correct, so the README/run.py strings are brought into line with what the code
actually does.

For each finding there are two tests:

- ``test_<finding>_violated_invariant`` -- asserts the BUGGY state. It must
  PASS before the fix and FAIL after the fix.
- ``test_<finding>_honest_fix`` -- asserts the FIXED state. It must FAIL before
  the fix and PASS after the fix.
"""
import inspect
from pathlib import Path

import run

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
README_PATH = TUTORIAL_DIR / "README.md"
RUN_PATH = TUTORIAL_DIR / "run.py"


def _takeaway_text() -> str:
    """Return the Takeaway section of the generated README."""
    text = README_PATH.read_text()
    assert "## Takeaway" in text, "README has no Takeaway section"
    after = text.split("## Takeaway", 1)[1]
    # Stop at the next top-level section so we only inspect the Takeaway prose.
    return after.split("\n## ", 1)[0]


def _profile_block() -> str:
    """Return the run.py source for the profile-log-likelihood loop."""
    src = RUN_PATH.read_text()
    assert "neg_ll_concentrated" in src, "profile block not found in run.py"
    start = src.index("# Identification figure")
    end = src.index("# Figures and tables")
    return src[start:end]


# ---------------------------------------------------------------------------
# Finding 1: Stone-Geary minima are never estimated
# ---------------------------------------------------------------------------
def test_finding1_violated_invariant():
    """Buggy state: the Takeaway claims NLS jointly estimates Stone-Geary minima.

    The code is the ground truth: ``fit_nls`` estimates only three parameters
    and never touches omega, so the Takeaway claim is false. This test pins
    that false claim. PASSES on the buggy README; FAILS once the claim is
    corrected.
    """
    # Code ground truth: NLS is strictly a 3-parameter fit with omega == 0.
    src = inspect.getsource(run.fit_nls)
    assert "omega" not in src
    assert "beta, delta, alpha = theta" in inspect.getsource(run.nls_residuals)
    # The buggy Takeaway nonetheless claims the minima are jointly estimated.
    assert "Stone-Geary minima be jointly estimated" in _takeaway_text()


def test_finding1_honest_fix():
    """Fixed state: the Takeaway no longer claims NLS estimates Stone-Geary minima.

    The audit's accepted fix (Option A) keeps omega fixed at zero in code and
    corrects the prose. The honest end state is that the Takeaway does not
    assert NLS "jointly estimated" the Stone-Geary minima. The word
    "Stone-Geary" may still appear (e.g. to state the minima are fixed at
    zero), so the test targets the false capability claim, not the term.

    FAILS on buggy code; PASSES once the false claim is removed.
    """
    takeaway = _takeaway_text()
    assert "Stone-Geary minima be jointly estimated" not in takeaway
    assert "lets the Stone-Geary" not in takeaway
    # If Stone-Geary is mentioned at all in the Takeaway, it must describe the
    # minima as fixed/zero, never as estimated.
    if "Stone-Geary" in takeaway:
        assert "jointly estimate" not in takeaway


# ---------------------------------------------------------------------------
# Finding 2: profile log-likelihood ignores Tobit censoring
# ---------------------------------------------------------------------------
def test_finding2_violated_invariant():
    """Buggy state: the profile loop is Gaussian but the prose does not say so.

    ``neg_ll_concentrated`` sums squared residuals over all observations and
    never calls logcdf/logsf, so censored cells are scored with the interior
    density. The buggy README captions the figure "profile log-likelihood"
    with no qualification, implying the Tobit criterion used for point
    estimation. This test pins that undisclosed mismatch. PASSES on the buggy
    README; FAILS once the prose discloses the Gaussian criterion.
    """
    # Code ground truth: the profile loop uses a Gaussian, uncensored criterion.
    block = _profile_block()
    assert "neg_ll_concentrated" in block
    assert "np.sum(resid**2)" in block
    assert "logcdf" not in block
    assert "logsf" not in block
    # The buggy README never discloses that the profile drops the censoring.
    readme = README_PATH.read_text()
    assert "Gaussian sum-of-squares" not in readme
    assert "without the Tobit censoring" not in readme


def test_finding2_honest_fix():
    """Fixed state: prose discloses the profile uses the Gaussian criterion.

    The accepted fix is a one-line disclosure (audit Step 2): the Results prose
    and the figure caption must state that the profile log-likelihood uses the
    Gaussian sum-of-squares criterion, not the full two-limit Tobit likelihood
    used for point estimation. FAILS on buggy code; PASSES after the prose fix.
    """
    readme = README_PATH.read_text()
    run_src = RUN_PATH.read_text()
    # The Results prose around the identification figure discloses the
    # Gaussian (uncensored) criterion.
    assert "Gaussian" in readme.split("identification-profile.png")[0].rsplit(
        "## ", 1
    )[1] or "without the Tobit censoring" in readme
    # The disclosure lives in README.md (prose was split from run.py).
    assert ("Gaussian sum-of-squares" in readme
            or "without the Tobit censoring" in readme)
