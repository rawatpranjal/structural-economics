"""Faithfulness tests for theory-of-the-firm tutorial.

Audit: bullshit-detector_theory-of-the-firm_2026-05-20.md
Finding 1 (DILUTED, LOW): prose gloss "capture share b_g(s) of marginal value"
is imprecise; the code and FOC implement revenue sharing (b_g of theta*x), not
marginal-value sharing (b_g of V'(x)=theta-x).

run.py no longer contains prose: prose lives in README.md.
Prose tests read README.md; computation tests read run.py source.
"""
from pathlib import Path

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"
README = Path(__file__).resolve().parents[1] / "README.md"


def _src() -> str:
    return RUN_PY.read_text()


def _readme() -> str:
    return README.read_text()


def test_finding1_violated_invariant_code_uses_revenue_sharing():
    """Violated-invariant: the implemented private FOC is b_g*theta - x = 0
    (revenue sharing). It must NOT be the marginal-value form b_g*(theta - x).
    Passed on the buggy prose; still passes after the prose fix because the
    code/FOC were always correct."""
    src = _src()
    assert r"b_g(s)\theta - x = 0" not in src  # LaTeX FOC lives in README now
    assert r"b_g*(theta - x)" not in src
    assert r"b_g(s)*(\theta - x)" not in src


def test_finding1_honest_fix_prose_names_revenue_not_marginal_value():
    """Honest-fix: the Equations prose must name the share as a share of
    revenue (or marginal productivity), not 'marginal value'. Fails on the
    buggy prose; passes after the fix."""
    readme = _readme()
    assert "of marginal value" not in readme
    assert ("of revenue" in readme) or ("of marginal productivity" in readme)


def test_finding1_recheck_model_setup_table_not_marginal():
    """Recheck residual (DILUTED, LOW): the Model Setup glossary row for
    b_g(s) must not call it 'marginal investment return'. That phrase admits
    a V'(x)=theta-x reading inconsistent with the revenue-sharing code/FOC.
    The row must instead name the share as a share of revenue theta*x,
    consistent with the Equations section. Fails on the residual phrasing;
    passes after the fix."""
    readme = _readme()
    assert "marginal investment return" not in readme
    assert r"Share of revenue $\theta x$ captured by the investor" in readme
