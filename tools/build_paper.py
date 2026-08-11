"""Generate the manuscript's tables and typeset the preprint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bist_predict.paper.appendix import render_all_appendices  # noqa: E402
from bist_predict.paper.calibration_tables import (  # noqa: E402
    render_all_calibration_tables,
)
from bist_predict.paper.tables import escape_latex, render_all_tables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MUTATION_HARNESS = ROOT / "tools" / "mutation_check.py"
RENDERED = ROOT / "paper.pdf"
CALIBRATION = ROOT / "calibration" / "study.json"


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
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CALIBRATION,
        help="simulation calibration study written by 'bist-predict calibrate'",
    )
    parser.add_argument("--skip-typeset", action="store_true", help="write tables only")
    arguments = parser.parse_args(argv)

    run = arguments.run
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    # The calibration study measures the estimators, not the market, so it is a
    # separate artifact with its own hash and is not regenerated per run. The
    # manuscript cites both, so neither may be missing at build time.
    if not arguments.calibration.exists():
        raise SystemExit(
            f"calibration study not found at {arguments.calibration}; "
            "run 'bist-predict calibrate' first"
        )
    study = json.loads(arguments.calibration.read_text(encoding="utf-8"))
    generated = PAPER / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    tables = render_all_tables(metrics)
    tables.update(render_all_calibration_tables(study))
    tables.update(
        render_all_appendices(
            metrics=metrics,
            config=yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")),
            folds=json.loads((run / "folds.json").read_text(encoding="utf-8")),
            mutation_source=MUTATION_HARNESS.read_text(encoding="utf-8"),
        )
    )
    for name, body in tables.items():
        (generated / f"{name}.tex").write_text(body, encoding="utf-8")
    (generated / "authors.tex").write_text(
        _authors(arguments.author, arguments.affiliation, arguments.address), encoding="utf-8"
    )
    print(f"wrote {len(tables) + 1} generated files to {generated}")

    if arguments.skip_typeset:
        return 0
    # TeX stamps the build time into the PDF, so an unchanged manuscript rebuilds
    # to different bytes every time and a committed copy looks perpetually
    # modified. Pinning the clock to the run's creation time makes the rendered
    # document a deterministic function of the bundle it reports.
    created = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["created_at"]
    environment = {
        **os.environ,
        "SOURCE_DATE_EPOCH": str(int(datetime.fromisoformat(created).timestamp())),
        "FORCE_SOURCE_DATE": "1",
    }
    result = subprocess.run(
        ["tectonic", "--keep-logs", "main.tex"],
        cwd=PAPER,
        capture_output=True,
        text=True,
        env=environment,
    )
    sys.stderr.write(result.stderr[-4000:])
    if result.returncode != 0:
        return result.returncode
    print(f"typeset {PAPER / 'main.pdf'}")
    # The rendered manuscript is the artifact a reader wants first, so a copy is
    # committed at the repository root. The LaTeX sources and the intermediate
    # files stay under paper/, where they are ignored.
    RENDERED.write_bytes((PAPER / "main.pdf").read_bytes())
    print(f"copied {RENDERED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
