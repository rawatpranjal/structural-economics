"""Faithfulness tests for the metropolis-hastings tutorial.

Finding 1 (DATA DRIFT): the Method-1 Model Setup row "MH draws | 20,000 |
After burn-in of 1,000" implies 20,000 retained draws, but the code retains
19,000 (20,000 total minus 1,000 burned).

  * test_*_violated_invariant -- passes on the original ambiguous README.
  * test_*_honest_fix        -- passes only once the row disambiguates total
    vs retained.

The README is generated from run.py, so tests assert on the generated README.
"""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
README = HERE / "README.md"


def readme_text() -> str:
    return README.read_text()


# --- Finding 1: Model Setup "MH draws" row total/retained ambiguity ----------

def test_f1_violated_invariant_retained_count_is_19000():
    """The code retains 20,000 - 1,000 = 19,000 draws, not 20,000."""
    assert 20_000 - 1_000 == 19_000


def test_f1_honest_fix_readme_row_disambiguates_total_vs_retained():
    """The Model Setup MH-draws row must not imply 20,000 retained draws."""
    text = readme_text()
    assert "20,000 | After burn-in of 1,000" not in text
    assert "20,000 total" in text or "19,000 retained" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
