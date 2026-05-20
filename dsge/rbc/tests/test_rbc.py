"""Faithfulness tests for the RBC tutorial.

Covers bullshit-detector findings 1 and 2 (2026-05-20 audit):
  - Finding 1 (DATA DRIFT): the Equations prose showed a stale C/Y = 0.76,
    inconsistent with the {:.2f} format string and the Model Setup table.
  - Finding 2 (MISLABELED): the Overview said two solvers "run in parallel"
    while the code runs them sequentially in one thread.
"""
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TUTORIAL_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

RUN_SRC = (TUTORIAL_DIR / "run.py").read_text()
README = (TUTORIAL_DIR / "README.md").read_text()


def test_readme_c_over_y_two_decimal_display():
    """Retargeted from run.py source to README.md prose.

    The Equations section in README.md must render C/Y to two decimal places
    (format contract previously enforced via the {:.2f} f-string in run.py).
    """
    alpha, beta, delta = 0.33, 0.99, 0.025
    mpk = 1.0 / beta - 1.0 + delta
    k = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    y = k ** alpha
    c_over_y = (y - delta * k) / y
    two = f"{c_over_y:.2f}"
    assert f"$C/Y = {two}$" in README


def test_readme_c_over_y_displays_are_consistent_roundings():
    """Honest fix: the Equations C/Y and the Model Setup C/Y agree.

    The audit (Finding 1) assumed {:.2f} of C/Y would render 0.77 and
    contradict the 0.765 Model Setup value. The actual C/Y is 0.76496,
    so {:.2f} renders 0.76 and {:.3f} renders 0.765 -- the same number
    at two precisions, not a contradiction. This test pins that: after a
    clean regeneration the two displays must be consistent roundings.
    """
    alpha, beta, delta = 0.33, 0.99, 0.025
    mpk = 1.0 / beta - 1.0 + delta
    k = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    y = k ** alpha
    c_over_y = (y - delta * k) / y
    two = f"{c_over_y:.2f}"
    three = f"{c_over_y:.3f}"
    assert f"$C/Y = {two}$" in README  # Equations prose
    assert f"| $C/Y$ | {three} |" in README  # Model Setup table
    assert three.startswith(two)  # consistent roundings of one value


def test_solvers_run_single_threaded():
    """Violated invariant: the two solvers are called sequentially.

    The source has no threading / multiprocessing; the calls sit one after
    another in main(). Passes on current code.
    """
    assert "import threading" not in RUN_SRC
    assert "import multiprocessing" not in RUN_SRC
    assert "solve_log_linear_policy(" in RUN_SRC
    assert "klein_qz_policy_fixed(" in RUN_SRC


def test_overview_does_not_claim_parallel_execution():
    """Honest fix: Overview must not say the solvers run in parallel.

    Fails on the stale README ("run in parallel"); passes once the prose
    is corrected to a sequential description.
    """
    assert "run in parallel" not in README
