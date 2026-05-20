"""Faithfulness tests for the dynamic-entry-exit tutorial.

These tests are derived directly from the bullshit-detector audit
`bullshit-detector_dynamic-entry-exit_2026-05-20.md`. Each finding contributes
two tests:

* a "violated-invariant" test that PASSES on the original buggy code, and
* an "honest-fix" test that FAILS on the original buggy code.

After the finding is fixed properly, the violated-invariant test must FAIL and
the honest-fix test must PASS.
"""
import inspect
import sys
from pathlib import Path

_TUTORIAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TUTORIAL_DIR))

import run  # noqa: E402

_SOLVE_PARAMS = dict(N_max=30, a=10, b=1, c=2, f=0.5, K=5.0, beta=0.95, tol=1e-8)


# ---------------------------------------------------------------------------
# Finding 1: _exit_prob must integrate over the Binomial survivor distribution,
# not use the self-loop value proxy V[N-1].
# ---------------------------------------------------------------------------

def test_finding1_violated_invariant_self_loop_proxy():
    """Original bug: _exit_prob uses V[N - 1] and never touches the binomial.

    PASSES on buggy code, FAILS once the fix integrates over survivors.
    """
    src = inspect.getsource(run._exit_prob)
    assert "V[N - 1]" in src and "binom" not in src


def test_finding1_honest_fix_integrates_over_survivors():
    """Honest fix: _exit_prob integrates the continuation value over the
    Binomial survivor distribution, matching Delta(N) in the Equations section.

    The honest fix factors the survivor integral into the helper
    ``_continuation_value`` and calls it from ``_exit_prob``. The test follows
    that call graph: ``_continuation_value`` must use ``binom`` and
    ``_exit_prob`` must invoke it. The buggy code had neither: ``_exit_prob``
    used the self-loop proxy ``V[N - 1]`` and ``_continuation_value`` did not
    exist.

    FAILS on buggy code, PASSES once the fix is applied.
    """
    assert hasattr(run, "_continuation_value")
    assert "binom" in inspect.getsource(run._continuation_value)
    assert "_continuation_value" in inspect.getsource(run._exit_prob)


def test_finding1_exit_prob_uses_same_delta_as_value_update():
    """The exit probability and the converged value function must come from the
    same Delta(N). With a consistent integrated continuation value,
    V(N) = sigma * log(1 + exp(Delta/sigma)) and
    p_exit(N) = 1 / (1 + exp(Delta/sigma)) imply
    exp(V(N)/sigma) - 1 == exp(-logit(p_exit)).

    Equivalently p_exit(N) == 1 / exp(V(N)/sigma), which holds exactly only when
    both objects are computed from the identical Delta(N).

    FAILS on buggy code (two different Deltas), PASSES on the honest fix.
    """
    import numpy as np

    V, exit_prob, _, N_grid, _ = run.solve_model(**_SOLVE_PARAMS)
    sigma = 1.0
    # V(N) = sigma * log(1 + exp(Delta/sigma))  =>  exp(V/sigma) - 1 = exp(Delta/sigma)
    # p_exit = 1 / (1 + exp(Delta/sigma))       =>  exp(Delta/sigma) = 1/p_exit - 1
    lhs = np.exp(V / sigma) - 1.0
    rhs = 1.0 / exit_prob - 1.0
    # Compare only where exit_prob is large enough to avoid catastrophic
    # cancellation in 1/p_exit.
    mask = exit_prob > 1e-6
    assert np.allclose(lhs[mask], rhs[mask], rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Finding 2: the reported convergence error is the dampened step (0.3 * true
# residual), but the README presents it as the undampened sup-norm.
# ---------------------------------------------------------------------------

def test_finding2_violated_invariant_dampened_error():
    """Original bug: info['error'] measures `V_update - V`, the dampened step,
    which equals 0.3 * (V_new - V). The convergence diagnostic is therefore the
    dampened step, not the undampened sup-norm residual the README claims.

    PASSES on buggy code, FAILS once the fix measures `V_new - V`.

    The audit's TDD draft used `error / 0.3 > tol` for this invariant, but the
    decisive signal is *which difference* the code reduces over. We assert on
    the source text so the test is exact rather than coupled to a particular
    converged magnitude.
    """
    src = inspect.getsource(run.solve_model)
    # The original code computed the convergence error from the dampened
    # update V_update; the honest fix computes it from V_new.
    assert "error = np.max(np.abs(V_update - V))" in src


def test_finding2_honest_fix_undampened_residual():
    """Honest fix: info['error'] is the undampened sup-norm residual
    max_N |V_new(N) - V_n(N)|, the quantity the README and pseudocode describe.

    The fixed point is V = V_new, so the honest stopping diagnostic is the
    undampened residual. At convergence it is below the tolerance.

    FAILS on buggy code (error is the dampened step), PASSES on the honest fix.
    """
    src = inspect.getsource(run.solve_model)
    assert "np.abs(V_new - V)" in src
    *_, info = run.solve_model(**_SOLVE_PARAMS)
    # The undampened residual is itself below tolerance at convergence.
    assert info["error"] < _SOLVE_PARAMS["tol"]


# ---------------------------------------------------------------------------
# Finding 3: the Solution Method pseudocode labels the integration variable "S"
# ambiguously; it should state explicitly that survivors are drawn from the
# N-1 rivals.
# ---------------------------------------------------------------------------

def test_finding3_violated_invariant_pseudocode_ambiguous():
    """Original bug: pseudocode says 'rival survivors S' without stating the
    integration is over the N-1 rivals.

    PASSES on buggy code, FAILS once the prose is disambiguated.
    """
    src = inspect.getsource(run.main)
    assert "for each possible number of rival survivors S:" in src
    assert "N-1 rivals" not in src and "N - 1 rivals" not in src


def test_finding3_honest_fix_pseudocode_states_n_minus_1_rivals():
    """Honest fix: the pseudocode explicitly says the survivor count is drawn
    from the N-1 rivals (Binomial(N-1, p_stay)).

    FAILS on buggy code, PASSES once the prose is fixed.
    """
    src = inspect.getsource(run.main)
    assert "N-1 rivals" in src or "N - 1 rivals" in src
