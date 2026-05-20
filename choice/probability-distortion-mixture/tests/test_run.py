"""Faithfulness regression tests for choice/probability-distortion-mixture.

Each finding from bullshit-detector_probability-distortion-mixture_2026-05-20.md
gets two tests:

* ``violated_invariant`` -- PASSES on the buggy code, FAILS once the finding is
  honestly fixed. It pins down the exact defect the audit named.
* ``honest_fix`` -- FAILS on the buggy code, PASSES once the finding is honestly
  fixed. It is the acceptance criterion for the fix.

The tutorial's ``run.py`` executes a long simulation when run as a script but is
import-safe (``main`` is guarded by ``if __name__``). To avoid running the full
EM fit, these tests inspect the ``run.py`` source text and exercise the small
pure helpers, never ``main``.
"""
import inspect
import re
from pathlib import Path

import numpy as np

import run

RUN_SRC = inspect.getsource(run)
TUT_DIR = Path(__file__).resolve().parents[1]
README = (TUT_DIR / "README.md").read_text()


def _minimize_bounds() -> list[list[str]]:
    """Every ``bounds=[...]`` literal passed to scipy.optimize.minimize in run.py."""
    return re.findall(r"bounds=\[(.*?)\]", RUN_SRC, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Finding 1: RRP figure labelled gain-domain but computed over all domains
# ---------------------------------------------------------------------------
def test_f1_violated_invariant_rrp_pools_all_domains():
    """Buggy code: the RRP groupby runs on the unfiltered df (all 3 domains).

    PASSES on buggy code, FAILS after the gain-domain filter is added.
    """
    rrp_block = RUN_SRC.split('df["ev"] =')[1].split("rrp_by_p")[0]
    # Buggy form references the raw `df`; honest fix references a gain-only frame.
    assert 'df["ev"] = df["p"]' in RUN_SRC
    assert "domain" not in rrp_block


def test_f1_honest_fix_rrp_filtered_to_gain_domain():
    """Honest fix: RRP is computed on a df filtered to domain == 'gain'.

    FAILS on buggy code, PASSES after the gain-domain filter is added.
    """
    # A gain-domain filter must exist on the path that feeds rrp_by_p.
    pre_groupby = RUN_SRC.split("rrp_by_p")[0]
    assert re.search(r'domain"\]\s*==\s*"gain"', pre_groupby), (
        "expected a domain == 'gain' filter before the RRP groupby"
    )
    # The figure-4 title must no longer over-claim a domain the data does not match.
    assert 'title("Median relative risk premia in the gain domain")' not in RUN_SRC or (
        re.search(r'domain"\]\s*==\s*"gain"', pre_groupby) is not None
    )


# ---------------------------------------------------------------------------
# Finding 2: Equations state lambda >= 1 but optimiser bound is lambda >= 0.5
# ---------------------------------------------------------------------------
def test_f2_violated_invariant_lambda_lower_bound_is_half():
    """Buggy code: the lambda bound passed to minimize is (0.5, 5.0).

    PASSES on buggy code, FAILS after the bound is tightened to (1.0, 5.0).
    """
    bounds = _minimize_bounds()
    assert bounds, "no minimize bounds found in run.py"
    assert any("(0.5, 5.0)" in b for b in bounds)


def test_f2_honest_fix_lambda_lower_bound_matches_equations():
    """Honest fix: every minimize bound enforces lambda >= 1, matching Equations.

    FAILS on buggy code, PASSES after the bound is tightened to (1.0, 5.0).
    """
    bounds = _minimize_bounds()
    assert bounds, "no minimize bounds found in run.py"
    for b in bounds:
        assert "(0.5, 5.0)" not in b, f"lambda lower bound still 0.5 in: {b}"
        assert "(1.0, 5.0)" in b, f"lambda bound not (1.0, 5.0) in: {b}"


# ---------------------------------------------------------------------------
# Finding 3: README claims EM monotone "by construction"; xi update is an approx
# ---------------------------------------------------------------------------
def test_f3_violated_invariant_xi_update_is_approximate():
    """The xi M-step uses argmax over types, an acknowledged approximation.

    PASSES on buggy code and after the fix -- it documents the algorithmic gap.
    """
    em_src = inspect.getsource(run.fit_mixture_em)
    assert "approximation" in em_src
    assert "argmax(posteriors" in em_src


def test_f3_honest_fix_readme_acknowledges_approximation():
    """Honest fix: README qualifies the monotonicity claim for the xi update.

    FAILS on buggy code (README states "by construction" with no caveat),
    PASSES after the qualifying sentence is added.
    """
    assert "by construction" not in README, (
        "README still claims EM is monotone 'by construction' without caveat"
    )
    assert "approximation" in README, (
        "README must disclose the maximum-posterior xi update is an approximation"
    )


# ---------------------------------------------------------------------------
# Finding 4: C=3 initial values equal the true DGP (oracle start), undisclosed
# ---------------------------------------------------------------------------
def _theta_init_c3() -> np.ndarray:
    block = RUN_SRC.split("theta_init_c3 = np.array([")[1].split("])")[0]
    rows = re.findall(r"\[([^\]]+)\]", block)
    return np.array([[float(v) for v in r.split(",")] for r in rows])


def _types_true() -> np.ndarray:
    block = RUN_SRC.split("types_true = np.array([")[1].split("])")[0]
    rows = re.findall(r"\[([^\]]+)\]", block)
    return np.array([[float(v) for v in r.split(",")] for r in rows])


def test_f4_violated_invariant_c3_init_equals_true_dgp():
    """Buggy code: theta_init_c3 is element-wise identical to types_true.

    PASSES on buggy code. Stays informative after the fix: the audit's chosen
    remedy is disclosure, so the init may still equal the DGP -- see the
    honest-fix test for the acceptance criterion.
    """
    assert np.allclose(_theta_init_c3(), _types_true())


def test_f4_honest_fix_oracle_start_is_disclosed_or_removed():
    """Honest fix: either the init no longer equals the DGP, or the README
    discloses that it does and that recovery partly reflects the oracle start.

    FAILS on buggy code (init equals DGP, README is silent),
    PASSES once the disclosure sentence is added (or the init is perturbed).
    """
    init_equals_dgp = np.allclose(_theta_init_c3(), _types_true())
    disclosed = "oracle" in README.lower() or "coincide with the true" in README
    assert (not init_equals_dgp) or disclosed, (
        "C=3 init equals the true DGP but README does not disclose the oracle start"
    )
