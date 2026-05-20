"""Faithfulness tests for the perturbation-linearization tutorial.

Audit: bullshit-detector_perturbation-linearization_2026-05-20.md, Finding 1.
Pseudocode step 1 claimed the code differentiates F at the steady state, but
perturbation_transition hardcodes the Taylor coefficients directly. No
differentiation step (sympy / np.gradient / numeric derivative) exists.
"""

from pathlib import Path

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def test_code_has_no_differentiation_step():
    """Violated-invariant: passes because no diff step exists in the code.

    perturbation_transition reads Taylor coefficients off the parameters.
    This test stays a true description of the code; it would only fail if a
    genuine differentiation step were added.
    """
    src = RUN_PY.read_text()
    diff_kw = ["np.gradient", "sympy", "scipy.misc.derivative", "jax.grad"]
    func_start = src.index("def perturbation_transition")
    func_end = src.index("\n\n\n", func_start)
    func_src = src[func_start:func_end]
    assert not any(kw in func_src for kw in diff_kw)


def test_pseudocode_does_not_claim_differentiation():
    """Honest-fix: passes once pseudocode step 1 no longer says 'Differentiate F'.

    Fails on current buggy prose ('1. Differentiate F at x_bar ...').
    """
    src = RUN_PY.read_text()
    assert "Differentiate F at x_bar" not in src
