"""Manuscript tables are generated, escaped, and structurally well formed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bist_predict.paper.tables import (
    TABLE_BUILDERS,
    escape_latex,
    render_all_tables,
    split_row,
)

RUN = Path(__file__).resolve().parents[2] / "runs" / "20260728T223101Z-8b27df3-2a71b8"

pytestmark = pytest.mark.skipif(not RUN.is_dir(), reason="committed accepted run is unavailable")


@pytest.fixture(scope="module")
def metrics() -> dict[str, object]:
    return json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))


def test_split_row_ignores_escaped_ampersands() -> None:
    r"""``Buy \& hold`` is one cell, not two.

    A splitter that counts every ``&`` reports an extra column for any row
    containing a literal ampersand, and then the column-count check below
    "finds" a defect that is really in the checker.
    """
    assert split_row(r"Buy \& hold & 1.00 & 2.00 \\") == [r"Buy \& hold", "1.00", "2.00"]
    assert len(split_row(r"a & b \\")) == 2


def test_escaping_covers_every_latex_special_character() -> None:
    escaped = escape_latex("a_b & c% d# e$ f{g}h~i^j")
    for character in ("_", "&", "%", "#", "$", "{", "}", "~", "^"):
        assert f"\\{character}" in escaped or character in {"~", "^"}
    assert r"\_" in escaped and r"\&" in escaped and r"\%" in escaped


def test_model_names_with_underscores_are_escaped(metrics: dict[str, object]) -> None:
    """``zero_return`` unescaped is a LaTeX compile error, not a typo."""
    table = TABLE_BUILDERS["prediction"](metrics)
    assert r"zero\_return" in table
    assert "zero_return" not in table.replace(r"zero\_return", "")


@pytest.mark.parametrize("name", sorted(TABLE_BUILDERS))
def test_every_row_has_the_declared_column_count(name: str, metrics: dict[str, object]) -> None:
    table = TABLE_BUILDERS[name](metrics)
    spec = table.split(r"\begin{tabular}{")[1].split("}")[0]
    expected = sum(1 for character in spec if character in "lrc")
    body = table.split(r"\midrule")[1].split(r"\bottomrule")[0]
    header = table.split(r"\toprule")[1].split(r"\midrule")[0]
    for row in (header, *body.strip().splitlines()):
        if not row.strip():
            continue
        assert len(split_row(row)) == expected, f"{name}: {row!r}"


@pytest.mark.parametrize("name", sorted(TABLE_BUILDERS))
def test_every_table_is_balanced_and_labelled(name: str, metrics: dict[str, object]) -> None:
    table = TABLE_BUILDERS[name](metrics)
    assert table.count(r"\begin{table}") == table.count(r"\end{table}") == 1
    assert table.count(r"\begin{tabular}") == table.count(r"\end{tabular}") == 1
    assert rf"\label{{tab:{name}}}" in table
    assert table.count("{") == table.count("}")


def test_tables_carry_the_run_numbers(metrics: dict[str, object]) -> None:
    """A table that does not contain the artifact's own numbers is decorative."""
    tables = render_all_tables(metrics)
    portfolio = metrics["portfolio"]
    assert f"{float(portfolio['sharpe']):.4f}" in tables["portfolio"]
    assert str(int(portfolio["invested_sessions"])) in tables["portfolio"]
    sharpe = metrics["inference"]["portfolio_sharpe"]
    assert f"{float(sharpe['deflated_sharpe_ratio']):.4f}" in tables["sharpe"]


def test_every_declared_table_is_rendered(metrics: dict[str, object]) -> None:
    assert set(render_all_tables(metrics)) == set(TABLE_BUILDERS)
