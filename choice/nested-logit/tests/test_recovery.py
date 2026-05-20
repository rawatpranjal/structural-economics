"""TDD test for nested-logit recovery framing (audit finding 4).

The audit flagged the parameter table as DILUTED: with degenerate instruments
sigma was 30% biased and the README never warned the reader. After the honest
instrument fix, sigma should recover well and the README prose should describe
recovery accurately.

These tests read the generated README.md as text, matching the audit's
pass-condition assertion exactly.
"""
from pathlib import Path

_TUT_DIR = Path(__file__).resolve().parents[1]
_README = _TUT_DIR / "README.md"
_RUN_PY = _TUT_DIR / "run.py"


def test_finding4_violated_invariant_no_bias_warning_in_run_py():
    """Bug: the ModelReport source in run.py never warns about finite-sample bias.

    Retargeted to README.md: prose now lives in README, not run.py.
    PASSES when README does NOT contain the phrase (original buggy state);
    FAILS after the honest fix adds an honest recovery description to README.
    """
    src = _README.read_text().lower()
    assert "finite-sample" not in src and "finite sample" not in src


def test_finding4_honest_fix_readme_describes_recovery_honestly():
    """Honest fix: the README explicitly discusses estimation accuracy / bias.

    FAILS on buggy README; PASSES after the honest fix regenerates it.
    """
    text = _README.read_text().lower()
    assert "finite-sample" in text or "finite sample" in text or "bias" in text


def test_finding4_honest_fix_sigma_recovered_within_tolerance():
    """Honest fix: with valid instruments, the nested-logit sigma estimate is
    close to the true 0.700.

    FAILS on buggy code (gap ~= 0.213); PASSES after the honest instrument fix.
    """
    csv = (_TUT_DIR / "tables" / "parameter-estimates.csv").read_text()
    rows = [r for r in csv.splitlines() if r.startswith("sigma")]
    assert rows, "sigma row missing from parameter-estimates.csv"
    nested_sigma = float(rows[0].split(",")[3])
    assert abs(nested_sigma - 0.700) < 0.10
