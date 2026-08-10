"""The claim checker must fail on a document that drifts from the artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT, accepted_run_directory

RUN = accepted_run_directory()

pytestmark = pytest.mark.skipif(not RUN.is_dir(), reason="committed accepted run is unavailable")


MANUSCRIPT = ROOT / "paper" / "main.tex"


def _run(*documents: Path) -> subprocess.CompletedProcess[str]:
    arguments = [sys.executable, str(ROOT / "tools" / "verify_claims.py"), "--run", str(RUN)]
    for document in documents:
        arguments += ["--document", str(document)]
    return subprocess.run(arguments, capture_output=True, text=True, cwd=ROOT)


def test_the_committed_documents_pass_every_claim() -> None:
    """Both documents are checked together, as `make verify-claims` checks them.

    A claim stated in the manuscript but not in the README is still a claim the
    project makes, and it is still resolved against the artifact that decides
    it. Checking either document alone would report the other's claims as
    missing rather than as unverified.
    """
    result = _run(ROOT / "README.md", MANUSCRIPT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout


def test_a_drifting_manuscript_number_is_rejected(tmp_path: Path) -> None:
    """The manuscript's calibration figures are checked, not only the README's."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    anchor = "significantly worse than the benchmark in $92.19\\%$"
    assert anchor in text, "anchor no longer present in the manuscript"
    document = tmp_path / "main.tex"
    document.write_text(
        text.replace(anchor, "significantly worse than the benchmark in $12.34\\%$"),
        encoding="utf-8",
    )
    result = _run(ROOT / "README.md", document)
    assert result.returncode == 1, result.stdout


@pytest.mark.parametrize(
    ("original", "corrupted"),
    [
        ("correlate at 0.570", "correlate at 0.210"),
        ("loses 4.77% of its capital", "loses 1.23% of its capital"),
        ("gross return of 7.77%", "gross return of 21.40%"),
        ("72-configuration grid", "9-configuration grid"),
        ("1,004 provider rows", "9,999 provider rows"),
    ],
)
def test_a_drifting_number_is_rejected(tmp_path: Path, original: str, corrupted: str) -> None:
    """Changing one figure in the prose must turn the check red.

    The manuscript is passed alongside the corrupted README so that the drift
    is the only reason the check can fail. Checking the README alone would fail
    on the manuscript's own claims being absent, and the test would pass
    whatever the corruption did.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert original in text, f"anchor no longer present: {original}"
    document = tmp_path / "README.md"
    document.write_text(text.replace(original, corrupted), encoding="utf-8")
    result = _run(document, MANUSCRIPT)
    assert result.returncode == 1, result.stdout


def test_deleting_a_claim_sentence_is_also_rejected(tmp_path: Path) -> None:
    """Otherwise a claim could be removed rather than corrected."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(text.replace("deflated Sharpe ratio of", "value of"), encoding="utf-8")
    result = _run(document, MANUSCRIPT)
    assert result.returncode == 1
    assert "anchor text absent" in result.stdout


def test_a_stale_generated_block_is_rejected(tmp_path: Path) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(text.replace("| Sharpe |", "| Sharpe (stale) |"), encoding="utf-8")
    result = _run(document, MANUSCRIPT)
    assert result.returncode == 1
    assert "STALE" in result.stdout
