"""The manuscript must typeset, and the rendered pages must be inspectable."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "20260728T223101Z-8b27df3-2a71b8"
PDF = ROOT / "paper" / "main.pdf"

pytestmark = [
    pytest.mark.skipif(not RUN.is_dir(), reason="committed accepted run is unavailable"),
    pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic is not installed"),
]


@pytest.fixture(scope="module")
def rendered() -> tuple[int, str]:
    """Typeset the manuscript and return its page count and full text."""
    fitz = pytest.importorskip("fitz", reason="pymupdf is needed to inspect the PDF")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_paper.py"),
            "--run",
            str(RUN),
            "--author",
            "Test Author",
            "--affiliation",
            "Test Affiliation",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    document = fitz.open(PDF)
    return document.page_count, "".join(page.get_text() for page in document)


def test_the_manuscript_typesets(rendered: tuple[int, str]) -> None:
    page_count, _ = rendered
    assert 10 <= page_count <= 20, f"unexpected length: {page_count} pages"


def test_no_unresolved_reference_or_citation(rendered: tuple[int, str]) -> None:
    """A missing label ships as `??` and a missing bib key as `[?]`."""
    _, text = rendered
    assert "??" not in text
    assert "[?]" not in text


def test_the_notice_does_not_claim_the_paper_is_under_review(
    rendered: tuple[int, str],
) -> None:
    """The style file's preprint option says "Preprint. Under review." by default."""
    _, text = rendered
    assert "Under review" not in text
    assert "Preprint." in text


def test_every_declared_table_and_figure_is_referenced(rendered: tuple[int, str]) -> None:
    _, text = rendered
    tables = {int(number) for number in re.findall(r"Table (\d+):", text)}
    figures = {int(number) for number in re.findall(r"Figure (\d+):", text)}
    assert tables == set(range(1, len(tables) + 1)), f"gap in table numbering: {sorted(tables)}"
    assert figures == set(range(1, len(figures) + 1)), f"gap in figure numbering: {sorted(figures)}"
    assert len(figures) == 9


def test_the_headline_numbers_reach_the_page(rendered: tuple[int, str]) -> None:
    """A table that renders but drops its numbers would still typeset cleanly."""
    _, text = rendered
    flattened = re.sub(r"\s+", " ", text)
    for value in ("0.0452", "0.1267", "0.6891", "0.9970", "0.5697", "177.2", "62 of 120"):
        assert value in flattened, f"missing from the rendered PDF: {value}"


def test_an_author_name_is_required() -> None:
    """The affiliation is the one claim no artifact can verify, so it is not inferred."""
    from bist_predict.paper import __name__ as _  # noqa: F401

    sys.path.insert(0, str(ROOT / "tools"))
    from build_paper import _authors  # type: ignore[import-not-found]

    with pytest.raises(ValueError, match="author name is required"):
        _authors("", "Some Institution")
