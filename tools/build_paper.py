"""Generate the manuscript's tables and typeset the preprint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bist_predict.paper.tables import escape_latex, render_all_tables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def _grid_appendix(metrics: dict[str, object]) -> str:
    """Render every configuration trial.

    Seventy-two rows do not fit in a ``table`` float: the overflow silently runs
    off the page and prints over the folio. ``longtable`` breaks across pages and
    repeats the header.
    """
    trials = metrics["configuration_sensitivity"]["trials"]  # type: ignore[index]
    ordered = sorted(trials, key=lambda trial: -float(trial["per_period_sharpe"]))
    header = r"    Configuration & Folds & Sessions & Net return & Sharpe & Round trips \\"
    lines = [
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{longtable}{lrrrrr}",
        r"  \caption{Every configuration in the grid, ordered by per-session Sharpe ratio.",
        r"  Each row is a complete re-run of the evaluation.}",
        r"  \label{tab:grid} \\",
        r"    \toprule",
        header,
        r"    \midrule",
        r"  \endfirsthead",
        r"    \toprule",
        header,
        r"    \midrule",
        r"  \endhead",
        r"    \bottomrule",
        r"  \endfoot",
    ]
    for trial in ordered:
        lines.append(
            "    "
            + " & ".join(
                (
                    rf"\texttt{{{escape_latex(trial['trial_id'])}}}",
                    str(int(trial["fold_count"])),
                    str(int(trial["session_count"])),
                    rf"{float(trial['net_return']) * 100:.2f}\%",
                    f"{float(trial['per_period_sharpe']):.4f}",
                    str(int(trial["trade_count"])),
                )
            )
            + r" \\"
        )
    lines += [r"\end{longtable}", r"\end{center}", ""]
    return "\n".join(lines)


def _authors(author: str, affiliation: str, address: str = "") -> str:
    """Render the author block.

    The affiliation is the one claim in this document that no artifact in the
    repository can verify, so it is supplied explicitly and never inferred.
    """
    if not author.strip():
        raise ValueError("an author name is required; it is not inferred")
    lines = [escape_latex(author.strip())]
    if affiliation.strip():
        lines.append(escape_latex(affiliation.strip()))
    if address.strip():
        lines.append(escape_latex(address.strip()))
    return "\\author{%\n  " + "\\\\\n  ".join(lines) + "\n}\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the tables and run tectonic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="accepted run directory")
    parser.add_argument("--author", required=True, help="author name for the title block")
    parser.add_argument("--affiliation", default="", help="institution, verbatim; may be empty")
    parser.add_argument("--address", default="", help="city and country, verbatim; may be empty")
    parser.add_argument("--skip-typeset", action="store_true", help="write tables only")
    arguments = parser.parse_args(argv)

    metrics = json.loads((arguments.run / "metrics.json").read_text(encoding="utf-8"))
    generated = PAPER / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    tables = render_all_tables(metrics)
    tables["grid_appendix"] = _grid_appendix(metrics)
    for name, body in tables.items():
        (generated / f"{name}.tex").write_text(body, encoding="utf-8")
    (generated / "authors.tex").write_text(
        _authors(arguments.author, arguments.affiliation, arguments.address), encoding="utf-8"
    )
    print(f"wrote {len(tables) + 1} generated files to {generated}")

    if arguments.skip_typeset:
        return 0
    result = subprocess.run(
        ["tectonic", "--keep-logs", "main.tex"],
        cwd=PAPER,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(result.stderr[-4000:])
    if result.returncode != 0:
        return result.returncode
    print(f"typeset {PAPER / 'main.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
