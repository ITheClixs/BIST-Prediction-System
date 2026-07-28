"""One shared visual language for every figure in the report.

Colour is assigned by the role a mark plays, not by the order a series happens
to arrive in: the null is always the same hue, fitted models are always the
same hue, and rejection is always the same hue. The four categorical slots come
from a palette validated for deuteranopia, protanopia and tritanopia
separation; the worst adjacent pair sits at OKLab dE 9.2 under deuteranopia,
above the 8.0 target.
"""

from __future__ import annotations

from collections.abc import Iterator
import textwrap
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.layout_engine import ConstrainedLayoutEngine  # noqa: E402

__all__ = [
    "COLOURS",
    "FIGURE_DPI",
    "caption",
    "figure",
    "save_figure",
]

_CAPTION_FONT_SIZE = 8.0
_CAPTION_WRAP_COLUMNS = 108

COLOURS = {
    # Validated categorical slots.
    "null": "#eb6834",  # the zero-return benchmark: the thing to beat
    "model": "#2a78d6",  # any fitted model
    "portfolio": "#4a3aa7",  # the executed strategy
    "reference": "#1baf7a",  # equal-weight universe and other references
    # Semantic roles.
    "adverse": "#e34948",  # significantly worse than the null
    "neutral": "#8a8983",
    "grid": "#d9d8d2",
    "ink": "#0b0b0b",
    "muted": "#52514e",
    "band": "#c9d8ef",
    "surface": "#fcfcfb",
}

FIGURE_DPI = 200

_RC: dict[str, object] = {
    "figure.facecolor": COLOURS["surface"],
    "axes.facecolor": COLOURS["surface"],
    "savefig.facecolor": COLOURS["surface"],
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "medium",
    "axes.labelsize": 9.0,
    "axes.labelcolor": COLOURS["muted"],
    "axes.edgecolor": COLOURS["grid"],
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": COLOURS["grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "xtick.color": COLOURS["muted"],
    "ytick.color": COLOURS["muted"],
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "text.color": COLOURS["ink"],
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def figure(width: float, height: float) -> Iterator[plt.Figure]:
    """Yield a figure under the shared style, closed on exit."""
    with plt.rc_context(_RC):  # type: ignore[arg-type]
        created = plt.figure(figsize=(width, height), dpi=FIGURE_DPI)
        try:
            yield created
        finally:
            plt.close(created)


def caption(fig: plt.Figure, text: str) -> None:
    """Place a wrapped caption under the plot and reserve room for it.

    The caption is laid out in figure coordinates and the constrained-layout
    rectangle is shrunk to match, so a long caption never widens the canvas and
    squashes the axes into a corner. Saving with a tight bounding box does
    exactly that, which is why ``save_figure`` does not.
    """
    lines = textwrap.wrap(" ".join(text.split()), width=_CAPTION_WRAP_COLUMNS)
    height_inches = len(lines) * _CAPTION_FONT_SIZE * 1.45 / 72.0
    reserved = min(0.42, (height_inches + 0.10) / fig.get_figheight())
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(rect=(0.0, reserved, 1.0, 1.0 - reserved))
    fig.text(
        0.012,
        reserved * 0.62,
        "\n".join(lines),
        fontsize=_CAPTION_FONT_SIZE,
        color=COLOURS["muted"],
        va="center",
        ha="left",
        linespacing=1.45,
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    """Write both a raster and a vector copy of one figure.

    The README embeds the PNG; the manuscript embeds the PDF. Raster art at
    report DPI is visibly soft once it reaches print.
    """
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=FIGURE_DPI)
    fig.savefig(pdf)
    return png, pdf
