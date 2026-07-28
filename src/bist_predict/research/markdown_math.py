"""Reject LaTeX that GitHub's Markdown-then-MathJax pipeline silently destroys.

GitHub renders a document as Markdown first and hands the surviving text to a
restricted MathJax build second. Several constructs that are valid LaTeX do not
survive that order, and the failure is silent in a local preview: the page ships
with a red error box or a mangled equation.

Balanced-delimiter checking is not enough. Every rule below corresponds to a
distinct way the pipeline breaks, and each one is exercised by a test that
reintroduces the defect.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["MathIssue", "check_markdown_math", "math_spans"]

_BANNED_MACROS = (
    "operatorname",
    "def",
    "newcommand",
    "renewcommand",
    "require",
    "DeclareMathOperator",
    "text ",
)
_BANNED_ESCAPES = (r"\,", r"\!", r"\;", r"\:", r"\{", r"\}", r"\_", r"\%", r"\&", r"\#")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class MathIssue:
    """One rendering defect, located and explained."""

    line: int
    rule: str
    detail: str
    excerpt: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.detail} -- {self.excerpt}"


def _content_lines(document: str) -> list[tuple[int, str]]:
    """Return numbered lines outside fenced code blocks."""
    lines: list[tuple[int, str]] = []
    inside_fence = False
    for number, line in enumerate(document.splitlines(), start=1):
        if _FENCE.match(line):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            lines.append((number, line))
    return lines


def _strip_inline_code(line: str) -> str:
    return re.sub(r"`[^`]*`", lambda match: " " * len(match.group(0)), line)


def math_spans(line: str) -> list[str]:
    """Return the mathematical spans of one line, display first then inline."""
    working = _strip_inline_code(line)
    spans = re.findall(r"\$\$(.+?)\$\$", working)
    remainder = re.sub(r"\$\$.+?\$\$", " ", working)
    spans.extend(re.findall(r"(?<!\$)\$([^$]+?)\$(?!\$)", remainder))
    return spans


def _check_display_blocks(lines: Sequence[tuple[int, str]]) -> Iterable[MathIssue]:
    for number, line in lines:
        stripped = _strip_inline_code(line)
        if stripped.count("$$") % 2 == 1:
            yield MathIssue(
                number,
                "multi-line-display",
                "a $$ block must open and close on one line; Markdown reprocesses the "
                "interior lines of a multi-line block as prose and destroys the equation",
                line.strip()[:90],
            )


def _check_inline_delimiters(lines: Sequence[tuple[int, str]]) -> Iterable[MathIssue]:
    for number, line in lines:
        stripped = _strip_inline_code(line)
        without_display = re.sub(r"\$\$.+?\$\$", " ", stripped)
        if without_display.count("$") % 2 == 1:
            yield MathIssue(
                number,
                "unbalanced-inline",
                "an odd number of inline $ delimiters leaves math open to the end of the file",
                line.strip()[:90],
            )


def _check_spans(lines: Sequence[tuple[int, str]]) -> Iterable[MathIssue]:
    for number, line in lines:
        for span in math_spans(line):
            for macro in _BANNED_MACROS:
                if f"\\{macro}" in span:
                    yield MathIssue(
                        number,
                        "banned-macro",
                        rf"\{macro.strip()} is not in GitHub's allowed macro set; "
                        r"use \mathrm instead",
                        span.strip()[:90],
                    )
            for escape in _BANNED_ESCAPES:
                if escape in span:
                    yield MathIssue(
                        number,
                        "escaped-punctuation",
                        f"{escape!r} loses its backslash to Markdown before MathJax sees it; "
                        r"use \quad for spacing and \lbrace / \rbrace for braces",
                        span.strip()[:90],
                    )
            if re.search(r"(?<![\\\w])\*", span):
                yield MathIssue(
                    number,
                    "bare-asterisk",
                    r"a bare * inside math is read as Markdown emphasis; use \cdot or \times",
                    span.strip()[:90],
                )


def check_markdown_math(document: str) -> list[MathIssue]:
    """Return every rendering defect found in a Markdown document."""
    lines = _content_lines(document)
    issues = [
        *_check_display_blocks(lines),
        *_check_inline_delimiters(lines),
        *_check_spans(lines),
    ]
    return sorted(issues, key=lambda issue: (issue.line, issue.rule))
