"""The claim checker must fail on a document that drifts from the artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT, accepted_run_directory

RUN = accepted_run_directory()

pytestmark = pytest.mark.skipif(not RUN.is_dir(), reason="committed accepted run is unavailable")


def _run(document: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "verify_claims.py"),
            "--run",
            str(RUN),
            "--document",
            str(document),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_the_committed_readme_passes_every_claim() -> None:
    result = _run(ROOT / "README.md")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout


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
    """Changing one figure in the prose must turn the check red."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert original in text, f"anchor no longer present: {original}"
    document = tmp_path / "README.md"
    document.write_text(text.replace(original, corrupted), encoding="utf-8")
    result = _run(document)
    assert result.returncode == 1, result.stdout


def test_deleting_a_claim_sentence_is_also_rejected(tmp_path: Path) -> None:
    """Otherwise a claim could be removed rather than corrected."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(text.replace("deflated Sharpe ratio of", "value of"), encoding="utf-8")
    result = _run(document)
    assert result.returncode == 1
    assert "anchor text absent" in result.stdout


def test_a_stale_generated_block_is_rejected(tmp_path: Path) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(text.replace("| Sharpe |", "| Sharpe (stale) |"), encoding="utf-8")
    result = _run(document)
    assert result.returncode == 1
    assert "STALE" in result.stdout
