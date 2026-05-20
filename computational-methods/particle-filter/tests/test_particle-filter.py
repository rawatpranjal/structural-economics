"""Faithfulness tests for the particle-filter tutorial.

Audit: bullshit-detector_particle-filter_2026-05-20.md, Finding 1.
The measurement-noise sweep was called with n_particles=350, n_runs=20
while the Model Setup table and Results prose advertise 500 / 50.
"""

from pathlib import Path

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def test_sweep_uses_documented_particle_count():
    """Violated-invariant: passes on the buggy 350-particle sweep call.

    Fails once the sweep is fixed to use the documented 500 particles.
    """
    src = RUN_PY.read_text()
    assert "n_particles=350" in src


def test_sweep_config_matches_readme():
    """Honest-fix: passes once the sweep uses the documented 500 / 50 config.

    Fails on current buggy code (350 / 20).
    """
    src = RUN_PY.read_text()
    assert "measurement_noise_sweep(n_periods, n_particles=500, n_runs=50)" in src
