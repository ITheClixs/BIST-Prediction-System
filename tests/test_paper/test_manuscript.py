"""The manuscript must typeset, and the rendered pages must be inspectable."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys

import pytest

from bist_predict.figures import FIGURE_BUILDERS
from tests.conftest import ROOT, accepted_run_directory

RUN = accepted_run_directory()
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
    assert 22 <= page_count <= 40, f"unexpected length: {page_count} pages"


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
    assert len(figures) == len(FIGURE_BUILDERS)


def test_the_headline_numbers_reach_the_page(rendered: tuple[int, str]) -> None:
    """A table that renders but drops its numbers would still typeset cleanly."""
    _, text = rendered
    flattened = re.sub(r"\s+", " ", text)
    headline = (
        "0.0452",  # deflated Sharpe ratio
        "0.1267",  # search threshold at the realised trial dispersion
        "0.2203",  # search threshold if the trials were independent
        "0.6891",  # Hansen SPA, consistent recentring
        "0.9970",  # White Reality Check
        "0.5697",  # mean within-session correlation
        "177.2",  # effective independent rows
        "62 of 120",  # sessions holding risk
        "0.1132",  # smallest detectable out-of-sample R-squared
        "15,126",  # sessions needed for the reference effect
        "0.3098",  # information coefficient the cost schedule demands
        "7.01",  # effective independent trials
    )
    for value in headline:
        assert value in flattened, f"missing from the rendered PDF: {value}"


def test_the_propositions_are_stated_and_proved(rendered: tuple[int, str]) -> None:
    """Each numbered proposition in the body must have a proof in the appendix."""
    _, text = rendered
    flattened = re.sub(r"\s+", " ", text)
    for index in (1, 2, 3):
        assert f"Proposition {index}" in flattened
        assert f"Proof of Proposition {index}" in flattened


def test_an_author_name_is_required() -> None:
    """The affiliation is the one claim no artifact can verify, so it is not inferred."""
    from bist_predict.paper import __name__ as _  # noqa: F401

    sys.path.insert(0, str(ROOT / "tools"))
    from build_paper import _authors  # type: ignore[import-not-found]

    with pytest.raises(ValueError, match="author name is required"):
        _authors("", "Some Institution")


def test_the_rendered_manuscript_is_committed_and_deterministic(
    rendered: tuple[int, str],
) -> None:
    """TeX stamps the build time, so an unpinned clock makes the copy look modified."""
    committed = ROOT / "paper.pdf"
    assert committed.is_file(), "paper.pdf is the artifact a reader opens first"
    first = hashlib.sha256(committed.read_bytes()).hexdigest()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_paper.py"),
            "--run",
            str(RUN),
            "--author",
            "Determinism Check",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashlib.sha256(committed.read_bytes()).hexdigest() != first, (
        "a different author block must change the document"
    )
    second = hashlib.sha256(committed.read_bytes()).hexdigest()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_paper.py"),
            "--run",
            str(RUN),
            "--author",
            "Determinism Check",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashlib.sha256(committed.read_bytes()).hexdigest() == second, (
        "the same inputs must render to the same bytes"
    )
