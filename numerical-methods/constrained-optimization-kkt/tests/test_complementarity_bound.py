"""Finding 2: the KKT figure traces complementarity with the exact barrier
multipliers mu_t = t / x_t, while the KKT *table* uses the heuristic
recover_multipliers. The two differ by a factor of n along the central path,
so the figure's barrier complementarity equals exactly n*t while the table's
does not. The Results prose must tell the reader the figure and table use
different multiplier formulas.
"""
import numpy as np
from pathlib import Path
from scipy.optimize import brentq

TUTORIAL = Path(__file__).resolve().parents[1]
README = TUTORIAL / "README.md"

A = np.array([4.0, 3.0, 0.5])
I_TOTAL = 3.0


def _central_path_point(t: float):
    """Solve the barrier subproblem for x and lambda at parameter t."""
    def x_of_lambda(lam):
        d = A - lam
        return 0.5 * (d + np.sqrt(d ** 2 + 4 * t))

    lam = brentq(lambda l: x_of_lambda(l).sum() - I_TOTAL, -100.0, 100.0, xtol=1e-13)
    return x_of_lambda(lam), lam


def test_violated_invariant_figure_complementarity_is_exactly_n_times_t():
    """PASSES on buggy code and stays true: with the exact barrier
    multipliers mu_t = t / x_t the figure's complementarity is exactly n*t,
    NOT the ~1e-8 the heuristic-recovery table row shows.

    This is the underlying numeric fact the README prose was glossing over.
    """
    t = 1e-8
    n = 3
    x, _ = _central_path_point(t)
    mu_t = t / x
    figure_compl = float(np.sum(mu_t * np.abs(x)))
    assert abs(figure_compl - n * t) < 1e-15
    # And it is plainly NOT 1e-8 (the heuristic-table value): off by factor 3.
    assert abs(figure_compl - 1e-8) > 1e-9


def test_honest_fix_prose_distinguishes_figure_and_table_multipliers():
    """FAILS on buggy code: the Results prose never tells the reader the
    figure complementarity uses exact barrier multipliers while the table
    uses heuristic recovery.

    After the honest fix this test PASSES.
    """
    text = README.read_text()
    assert "exact barrier multipliers" in text
    # The figure n*t value and the heuristic table value are named as distinct.
    assert "heuristic" in text
