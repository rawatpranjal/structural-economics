"""Faithfulness tests for the rbc-capital-tax tutorial.

Covers bullshit-detector finding 1 (DILUTED): the Euler-refinement pseudocode
in the README presents a full sweep over every state followed by an atomic
update of the saving rule g_K, i.e. a Jacobi scheme. The original code
updated ``policy_k[iz, :]`` inside the ``for iz`` loop, so later rows of the
sweep already saw the refreshed earlier rows -- a Gauss-Seidel update on the
iz dimension that the pseudocode does not describe.

The honest fix accumulates the saving rule into a ``policy_k_new`` buffer and
swaps it after the full iz-loop, mirroring the existing ``policy_c_new`` /
``policy_c`` pattern, so the executed update order matches the pseudocode.

run.py only runs main() under ``__main__``; importing it just defines
functions, so these tests inspect the source of solve_rbc_tax directly.
"""

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from run import solve_rbc_tax  # noqa: E402

SRC = inspect.getsource(solve_rbc_tax)


def _euler_block() -> str:
    """Return the Euler-refinement loop body from solve_rbc_tax source."""
    start = SRC.index("for euler_iter")
    end = SRC.index("return {")
    return SRC[start:end]


def test_violated_invariant_gauss_seidel_update_in_loop():
    """Violated-invariant test for finding 1.

    Passes on the buggy code: the saving-rule update sits inside the
    ``for iz`` loop, indented one level past it (Gauss-Seidel). After the
    honest Jacobi fix this exact in-loop update is gone, so the test FAILS.
    """
    block = _euler_block()
    # The in-loop Gauss-Seidel write is indented under `for iz` (12 spaces).
    assert "            policy_k[iz, :] = np.clip" in block


def test_honest_fix_jacobi_buffer_and_swap():
    """Honest-fix test for finding 1.

    Fails on the buggy code. After the fix the saving rule is accumulated
    into a ``policy_k_new`` buffer per row and the whole array is swapped
    in after the iz-loop completes, matching the Jacobi pseudocode.
    """
    block = _euler_block()
    # A per-row write into the buffer, indented under `for iz`.
    fills_buffer = "policy_k_new[iz, :]" in block
    # The atomic swap happens after the loop (8-space indent, not 12).
    atomic_swap = "        policy_k = policy_k_new" in block
    # No Gauss-Seidel in-loop write into the live array survives.
    no_gauss_seidel = "            policy_k[iz, :] = np.clip" not in block
    assert fills_buffer and atomic_swap and no_gauss_seidel
