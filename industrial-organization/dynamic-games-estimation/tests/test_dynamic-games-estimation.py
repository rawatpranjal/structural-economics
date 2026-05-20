"""Faithfulness tests for the dynamic-games-estimation tutorial.

The bullshit-detector audit (2026-05-20) flagged two findings:

Finding 1 (DILUTED/MED): the W_theta equation defines pi_bar as the
expected flow payoff with a prose-only note that "the logit action shock
is integrated out", but the code adds an explicit entropy correction
H_Bernoulli(p) + Euler-gamma to expected_flow before solving the Bellman
system. A replicator following the bare equation gets wrong W values.
The honest fix shows the entropy term explicitly in the Equations section.

Finding 2 (DATA DRIFT/LOW): the Results prose formats the model RMSE with
3 decimals, so 0.001354 prints as "0.001" -- a 26% understatement versus
the diagnostics CSV value 0.00135. The honest fix prints more precision.

Tests read source / README / CSV as text; they do not execute ``run.py``.
"""
import inspect
import re
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
README = FOLDER / "README.md"
RUN_PY = FOLDER / "run.py"
DIAG_CSV = FOLDER / "tables" / "estimator-diagnostics.csv"


def _readme() -> str:
    return README.read_text()


def _run_src() -> str:
    return RUN_PY.read_text()


# --- Finding 1: W_theta equation omits the entropy correction -------------

def test_violated_code_adds_entropy_not_in_equation():
    """Violated invariant: the code's policy_transition_and_flow adds an
    entropy + EULER_GAMMA correction to expected flow, while the README
    Equations section shows neither term. PASSES on the buggy state,
    FAILS once the equation discloses the entropy term."""
    src = _run_src()
    func_src = src[src.index("def policy_transition_and_flow"):
                   src.index("def policy_transition_and_flow") + 900]
    code_has_entropy = "entropy" in func_src and "EULER_GAMMA" in func_src
    eq_section = _readme()
    # Equations section, between the "Equations" header and "Model Setup".
    eq = eq_section.split("## Equations")[1].split("## Model Setup")[0]
    eq_hides_entropy = (
        "entropy" not in eq.lower()
        and "euler" not in eq.lower()
        and "\\text{H}" not in eq
        and "H(" not in eq
    )
    assert code_has_entropy and eq_hides_entropy


def test_fixed_equation_discloses_entropy_term():
    """Honest fix: the README Equations section shows the entropy /
    Euler-gamma correction explicitly in the pi_bar definition. FAILS on
    the buggy state, PASSES after the fix."""
    eq = _readme().split("## Equations")[1].split("## Model Setup")[0]
    assert ("entropy" in eq.lower()) or ("euler" in eq.lower()) or ("\\gamma" in eq)
    # The pi_bar definition must be more than the bare expected-flow average.
    assert "\\bar\\pi" in eq or "bar\\pi" in eq


# --- Finding 2: model RMSE printed at too-coarse precision ----------------

def _readme_model_rmse() -> float:
    text = _readme()
    chunk = text.split("model-implied RMSE against truth is **")[1]
    return float(chunk.split("**")[0])


def _csv_model_rmse() -> float:
    for line in DIAG_CSV.read_text().splitlines():
        if line.startswith("Model policy RMSE"):
            return float(line.split(",", 1)[1])
    raise AssertionError("Model policy RMSE row not found in diagnostics CSV")


def test_violated_model_rmse_understated_in_prose():
    """Violated invariant: the README prose RMSE differs from the CSV value
    by more than 0.0002 (3-decimal truncation understates it). PASSES on
    the buggy state, FAILS once the prose prints enough precision."""
    assert abs(_readme_model_rmse() - _csv_model_rmse()) > 0.0002


def test_fixed_model_rmse_matches_csv():
    """Honest fix: the README prose RMSE matches the diagnostics CSV value
    to within 0.0001. FAILS on the buggy state, PASSES after the fix."""
    assert abs(_readme_model_rmse() - _csv_model_rmse()) < 0.0001
