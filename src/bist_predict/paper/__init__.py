"""Manuscript generation from the immutable run artifacts."""

from bist_predict.paper.tables import (
    TABLE_BUILDERS,
    escape_latex,
    render_all_tables,
    split_row,
)

__all__ = ["TABLE_BUILDERS", "escape_latex", "render_all_tables", "split_row"]
