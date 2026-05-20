"""Faithfulness tests for choice/houtman-maks-rational-subsets/.

Audit: bullshit-detector_houtman-maks-rational-subsets_2026-05-20.md
Finding 1 (DILUTED): greedy pseudocode named only the primary participation
key, omitting the strict-degree and obs-id tie-breakers in the code.
Finding 2 (DILUTED): pseudocode did not make explicit the more-than-one-
observation component filter present in run.py:154.
"""
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent.parent
README = TUTORIAL_DIR / "README.md"


def test_finding1_violated_invariant_no_tiebreaker_in_readme():
    """Violated invariant: README pseudocode never named the tie-breaker.

    PASSED on the buggy README; FAILS once the honest fix names it.
    The audit's loose ``"tie"`` token collides with a reference title
    (Kwantitatieve), so this test keys on the precise tie-breaker phrasing.
    """
    text = README.read_text()
    assert "strict-arc degree" not in text and "breaking ties" not in text


def test_finding1_honest_fix_tiebreaker_disclosed():
    """Honest fix: pseudocode states the strict-degree / obs-id tie-breakers.

    FAILED on the buggy README; PASSES once the tie-breaker line exists.
    """
    text = README.read_text()
    assert "strict-arc degree" in text and "breaking ties" in text


def test_finding2_violated_invariant_no_component_size_filter():
    """Violated invariant: pseudocode omitted the >1-observation filter.

    PASSED on the buggy README; FAILS once the filter is made explicit.
    """
    assert "more than one observation" not in README.read_text()


def test_finding2_honest_fix_component_size_filter_disclosed():
    """Honest fix: pseudocode names the more-than-one-observation filter.

    FAILED on the buggy README; PASSES once the filter line exists.
    """
    assert "more than one observation" in README.read_text()
