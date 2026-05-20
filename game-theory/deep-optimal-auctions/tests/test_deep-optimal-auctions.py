"""Faithfulness tests for the deep-optimal-auctions tutorial.

Each audited finding gets two tests:
  - violated_invariant: captures the bug; PASSES on the buggy state, FAILS after the fix.
  - honest_fix: captures the faithful state; FAILS on the buggy state, PASSES after the fix.

run.py trains a network on import, so prose/claim tests read the run.py
source text instead of importing run.py.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUN_PY = (HERE / "run.py").read_text()


# --- Finding 1: audit-table column "Mean regret" stores max-of-means ---

def test_finding1_violated_invariant():
    # Buggy code: a column labelled "Mean regret" stores np.max(mean_regrets),
    # the maximum over bidders of each bidder's mean regret.
    assert '"Mean regret": f"{float(np.max(mean_regrets)):.4f}"' in RUN_PY


def test_finding1_honest_fix():
    # Honest fix: the column header names the max-over-bidders quantity it
    # actually computes, matching the figure y-axis "Largest mean bidder regret".
    assert '"Mean regret": f"{float(np.max(mean_regrets)):.4f}"' not in RUN_PY
    assert '"Largest mean bidder regret": f"{float(np.max(mean_regrets)):.4f}"' in RUN_PY
