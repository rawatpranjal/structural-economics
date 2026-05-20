"""Faithfulness tests for the HJB growth tutorial.

Covers bullshit-detector findings 1 and 2 (2026-05-20 audit), both DATA DRIFT:
  - Finding 1: the grid endpoint k_max is displayed via {:.2f}; 14.7997
    rounds to 14.80, which is a correct two-decimal rounding (delta 3e-4),
    not a contradiction.
  - Finding 2: HJB convergence diagnostics (iterations, residual) are
    runtime values; a clean regeneration keeps README and CSV consistent.
"""
import csv
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TUTORIAL_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

README = (TUTORIAL_DIR / "README.md").read_text()


def test_readme_k_max_display_is_correct_rounding():
    """Honest fix: the displayed grid endpoint is a faithful rounding.

    k_max = 2 * k_ss = 2 * 7.39976... = 14.7997..., which rounds to 14.80
    at two decimals. The README must show that rounding, not a value that
    misrepresents the grid endpoint.
    """
    alpha, A, rho, delta = 0.36, 1.0, 0.05, 0.05
    k_ss = (alpha * A / (rho + delta)) ** (1.0 / (1.0 - alpha))
    k_max = 2.0 * k_ss
    assert abs(k_max - 14.80) < 1e-3
    assert f"{k_max:.2f}]$" in README


def test_convergence_diagnostics_consistent_readme_and_csv():
    """Honest fix: HJB iterations and residual agree across README and CSV.

    After a clean regeneration the runtime convergence numbers committed to
    the CSV must match the numbers narrated in the README prose.
    """
    csv_path = TUTORIAL_DIR / "tables" / "steady-state.csv"
    rows = {r[0].strip(): r[-1].strip()
            for r in csv.reader(csv_path.open()) if r}
    iters = rows["HJB iterations"]
    residual = rows["HJB residual"]
    assert f"**{iters} iterations**" in README
    # CSV residual like 5.34e-07; README narrates the same to 2 sig figs.
    mantissa = residual.split("e")[0][:4]
    assert mantissa in README
