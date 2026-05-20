"""Faithfulness tests for vertical-contracts tutorial.

Audit: bullshit-detector_vertical-contracts_2026-05-20.md
Finding 1 (DILUTED, LOW): retail_outcome applies a margin floor
`max(..., wholesale + 0.05)` that the Equations section did not document.
Finding 2 (DILUTED, MED): the discount threshold tau is used in the equation
but its value (4) was never disclosed in the README.

run.py guards main() under __main__, so importing the module is side-effect
free and the helper functions can be exercised directly.
"""
import importlib.util
from pathlib import Path

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"
README = Path(__file__).resolve().parents[1] / "README.md"


def _load_run():
    spec = importlib.util.spec_from_file_location("vc_run", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _src() -> str:
    return RUN_PY.read_text()


def _readme() -> str:
    return README.read_text()


# --- Finding 1: undocumented margin floor -----------------------------------

def test_finding1_violated_invariant_floor_present_in_code():
    """Violated-invariant: the code applies the wholesale + 0.05 margin floor.
    This documents the implemented behaviour; it stays true after the fix
    because the fix documents the floor rather than removing it."""
    import inspect
    mod = _load_run()
    assert "wholesale + 0.05" in inspect.getsource(mod.retail_outcome)


def test_finding1_honest_fix_floor_documented_in_readme():
    """Honest-fix: the README must disclose the margin floor (value 0.05).
    Fails on the buggy README, which only printed the interior formula."""
    text = _readme()
    assert "0.05" in text
    assert "floor" in text.lower()


def test_finding1_floor_is_numerically_vacuous():
    """The audit claims the floor never binds for any of the 12 products under
    any contract. Confirms the documented guard is dead under this calibration
    so documenting (not removing) it is faithful."""
    mod = _load_run()
    products = mod.product_catalog()
    contracts = ["Wholesale only", "All-unit discount", "Slotting fees"]
    for contract in contracts:
        for idx in products.index:
            row = products.loc[idx]
            is_mars = row["Manufacturer"] == "Mars"
            wholesale = row["Marginal cost"] + 0.42
            if contract == "All-unit discount" and is_mars:
                wholesale -= 0.18  # most generous discount case
            interior = (row["Demand intercept"] + 4.0 * wholesale) / (2 * 4.0)
            assert interior >= wholesale + 0.05, (contract, idx)


# --- Finding 2: undisclosed threshold tau -----------------------------------

def test_finding2_violated_invariant_code_threshold_is_four():
    """Violated-invariant: the discount threshold default is 4. Holds on the
    buggy code and after the fix; the fix only adds README disclosure."""
    mod = _load_run()
    import inspect
    sig = inspect.signature(mod.evaluate_subset)
    assert sig.parameters["threshold"].default == 4


def test_finding2_honest_fix_threshold_value_disclosed_in_readme():
    """Honest-fix: tau = 4 must appear in the rendered README. Fails on the
    buggy README, which used the symbol tau with no value."""
    text = _readme()
    assert ("\\tau=4" in text) or ("\\tau = 4" in text)
