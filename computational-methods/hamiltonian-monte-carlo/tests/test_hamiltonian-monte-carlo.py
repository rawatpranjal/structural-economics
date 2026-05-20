"""Faithfulness tests for the hamiltonian-monte-carlo tutorial.

Each audit finding gets two tests:
  * test_*_violated_invariant -- passes on the original buggy prose.
  * test_*_honest_fix        -- passes only once the prose is corrected.

The README is generated from run.py, so the honest-fix tests assert on the
generated README and the committed CSV tables.
"""

from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
README = HERE / "README.md"
METHOD_CSV = HERE / "tables" / "method-comparison.csv"
SWEEP_CSV = HERE / "tables" / "stepsize-sweep.csv"
ACF_CSV = HERE / "tables" / "acf-summary.csv"


def readme_text() -> str:
    return README.read_text()


# --- Finding 1: "acceptance rates of 0.7 to 0.9 are typical" -----------------

def test_f1_violated_invariant_committed_acceptance_outside_claimed_range():
    """The HMC run's committed acceptance rate is outside the claimed 0.7-0.9."""
    df = pd.read_csv(METHOD_CSV)
    acc = float(
        df.loc[df["Method"] == "Hamiltonian Monte Carlo", "Acceptance rate"].iloc[0]
    )
    assert not (0.7 <= acc <= 0.9)


def test_f1_honest_fix_prose_does_not_claim_0_7_to_0_9_typical():
    """README must not present 0.7-0.9 as the typical rate for this run."""
    text = readme_text()
    assert "acceptance rates of 0.7 to 0.9 are typical" not in text


# --- Finding 2: "ESS largest at acceptance around 0.6 to 0.8" ----------------

def test_f2_violated_invariant_no_sweep_row_in_0_6_to_0_8():
    """No step-size-sweep row has acceptance inside the claimed 0.6-0.8 band."""
    df = pd.read_csv(SWEEP_CSV)
    acc = df["Acceptance rate"].astype(float)
    assert not ((acc >= 0.6) & (acc <= 0.8)).any()


def test_f2_honest_fix_prose_does_not_claim_sweep_matches_0_6_to_0_8():
    """README must not claim the sweep's ESS peak matches the 0.6-0.8 band."""
    text = readme_text()
    assert (
        "acceptance around 0.6 to 0.8, "
        "which matches the asymptotic-optimal acceptance" not in text
    )


# --- Finding 3 (recheck residual): ACF lag claim unbacked by an artifact -----

def test_f3_violated_invariant_acf_lag_claim_present_in_readme():
    """The README states a specific ACF lag count for theta_1. That claim is
    only honest if a committed artifact records the same number."""
    text = readme_text()
    assert "within " in text and " lags" in text


def test_f3_honest_fix_acf_csv_exists_and_backs_readme_lag_claim():
    """Honest-fix: tables/acf-summary.csv must exist, carry the per-series lag
    counts, and its hmc_x value must equal the lag count printed in the README.
    Fails before the artifact is committed; passes after."""
    assert ACF_CSV.exists(), "no committed ACF artifact to back the README lag claim"
    df = pd.read_csv(ACF_CSV)
    assert {"series", "lag_below_0.05"}.issubset(df.columns)
    csv_lag = int(df.loc[df["series"] == "hmc_x", "lag_below_0.05"].iloc[0])
    text = readme_text()
    # Anchor on the ACF sentence specifically; "within" also appears earlier.
    readme_lag = int(
        text.split("drops to near zero within ")[1].split(" lags")[0]
    )
    assert csv_lag == readme_lag


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
