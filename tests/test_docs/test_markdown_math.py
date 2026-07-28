"""Each GitHub math rule is checked by reintroducing the defect it guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.research.markdown_math import check_markdown_math

ROOT = Path(__file__).resolve().parents[2]

CLEAN = r"""
# Title

Inline math $r_{i,t+1}$ and a display block:

$$r_{i,t+1}^{\mathrm{exec}} = \frac{C_{i,t+1}}{O_{i,t+1}} - 1, \qquad y = \mathbb{1} \lbrace r > 0 \rbrace.$$

Text after.
"""


def test_a_clean_document_reports_nothing() -> None:
    assert check_markdown_math(CLEAN) == []


def test_the_project_readme_is_clean() -> None:
    issues = check_markdown_math((ROOT / "README.md").read_text(encoding="utf-8"))
    assert issues == [], "\n".join(str(issue) for issue in issues)


@pytest.mark.parametrize(
    ("defect", "rule"),
    [
        # A display block split across lines: Markdown reprocesses the interior.
        ("$$\nx = a + b\n$$", "multi-line-display"),
        # operatorname is rejected by GitHub's restricted MathJax build.
        (r"$$y = \operatorname{sign}(x)$$", "banned-macro"),
        (r"$$y = \DeclareMathOperator{\sgn}{sgn}$$", "banned-macro"),
        (r"$$\def\x{1} x$$", "banned-macro"),
        (r"$$\newcommand{\x}{1} x$$", "banned-macro"),
        (r"$$\require{color} x$$", "banned-macro"),
        # Backslash-escaped ASCII punctuation is consumed by Markdown first.
        (r"$$a \, b$$", "escaped-punctuation"),
        (r"$$a \! b$$", "escaped-punctuation"),
        (r"$$a \; b$$", "escaped-punctuation"),
        (r"$$\{ x \}$$", "escaped-punctuation"),
        (r"$$\texttt{a\_b}$$", "escaped-punctuation"),
        # A bare asterisk is read as Markdown emphasis.
        (r"$$a * b$$", "bare-asterisk"),
        # An odd inline delimiter leaves math open to the end of the document.
        ("text $x + 1 and more text", "unbalanced-inline"),
    ],
)
def test_each_defect_is_detected(defect: str, rule: str) -> None:
    issues = check_markdown_math(f"# Doc\n\n{defect}\n")
    assert rule in {issue.rule for issue in issues}, f"{rule} not flagged in {defect!r}"


def test_the_permitted_alternatives_are_not_flagged() -> None:
    """The replacements the rules point at must themselves survive the checker."""
    permitted = (
        r"$$a \quad b \qquad c$$"
        "\n"
        r"$$\lbrace x \rbrace \mathrm{sign}(x) \cdot y \times z$$"
    )
    assert check_markdown_math(f"# Doc\n\n{permitted}\n") == []


def test_fenced_code_is_exempt() -> None:
    """A shell block full of $ and * is not mathematics."""
    document = "# Doc\n\n```bash\necho $HOME && ls *.py\n```\n"
    assert check_markdown_math(document) == []


def test_inline_code_is_exempt() -> None:
    document = "# Doc\n\nUse `make reproduce RUN_ID=$ID` and `a * b`.\n"
    assert check_markdown_math(document) == []


def test_a_multi_line_block_is_flagged_even_when_delimiters_balance() -> None:
    """Balanced-delimiter checking alone passes this and ships broken math."""
    document = "# Doc\n\n$$\nx = a\n+ b\n$$\n"
    assert document.count("$$") == 2
    assert "multi-line-display" in {issue.rule for issue in check_markdown_math(document)}
