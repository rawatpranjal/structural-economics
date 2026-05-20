"""Faithfulness tests for the behavioral-NK tutorial.

Covers bullshit-detector findings 7 and 8 (2026-05-20 audit): the prose claim
that the behavioral forward-guidance response shrinks monotonically with the
news horizon is false. Both the rational and behavioral models show a
non-monotone (hump-shaped) absolute date-0 output response, a residual
forward-guidance puzzle that abs() in the figure hides.
"""
import importlib.util
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TUTORIAL_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_run():
    """Import run.py without executing main() (guarded by __name__)."""
    name = "behavioral_nk_run"
    spec = importlib.util.spec_from_file_location(name, TUTORIAL_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


def _abs_output_0(label, m):
    run = _load_run()
    calibration = run.Calibration()
    setting = run.AttentionSetting(label, m, m, "#000000")
    summary = run.forward_guidance_summary(calibration, setting)
    summary = summary.sort_values("Horizon")
    return [abs(v) for v in summary["Output"].to_numpy()]


def test_behavioral_fg_response_not_monotone():
    """Violated invariant: claim says |behavioral output[0]| shrinks with H.

    Passes on the buggy state because the response is in fact NOT monotone
    (it has a hump near H=7-10), so the all(...) monotone claim is false.
    """
    b_out = _abs_output_0("Behavioral NK", 0.85)
    monotone = all(b_out[h + 1] < b_out[h] for h in range(len(b_out) - 1))
    assert not monotone


def test_behavioral_fg_hump_band_exists():
    """Honest fix: confirm the residual FG puzzle hump (rises H=6..10)."""
    b_out = _abs_output_0("Behavioral NK", 0.85)
    assert all(b_out[h + 1] > b_out[h] for h in range(6, 10))


def test_rational_fg_response_rises_after_minimum():
    """Honest fix: the rational model also has a non-monotone hump."""
    r_out = _abs_output_0("Rational NK", 1.0)
    assert r_out[11] > r_out[5]


def test_readme_does_not_claim_monotone_shrink():
    """Honest fix: README prose must not claim a simple monotone shrink.

    Fails on the buggy README, which says the behavioral response shrinks
    the farther away the wedge is, with no qualification.
    """
    readme = (TUTORIAL_DIR / "README.md").read_text()
    assert "the more the behavioral response shrinks" not in readme


def test_behavioral_not_smaller_at_every_horizon():
    """Violated invariant: the recheck audit (2026-05-20) Finding 3.

    The README claims the behavioral response is smaller than the rational
    one "at every horizon". The data refutes the quantifier: H=0 ties
    exactly and H=5 has |behavioral| > |rational|. This test asserts the
    exceptions exist, so it PASSES on current data (true regardless of
    prose) and would FAIL only if the claim were literally true.
    """
    b_out = _abs_output_0("Behavioral NK", 0.85)
    r_out = _abs_output_0("Rational NK", 1.0)
    strictly_smaller_everywhere = all(b < r for b, r in zip(b_out, r_out))
    assert not strictly_smaller_everywhere
    assert b_out[0] == r_out[0]  # H=0 tie: contemporaneous wedge, no discounting
    assert b_out[5] > r_out[5]  # H=5 exception: sign-reversal phase offset


def test_readme_drops_every_horizon_quantifier():
    """Honest fix: README prose must not claim smaller "at every horizon".

    Fails on the stale README (which still carries the overstated
    quantifier) and passes once run.py is fixed and README regenerated.
    """
    readme = (TUTORIAL_DIR / "README.md").read_text()
    assert "smaller than the rational one at every horizon" not in readme
