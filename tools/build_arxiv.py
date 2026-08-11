"""Assemble a self-contained arXiv submission from the manuscript sources.

arXiv compiles the source it is given, in a directory that contains only what
the tarball carries. The working manuscript reads its figures from
``../docs/figures``, which is outside the paper directory and therefore outside
any submission, so the package is staged rather than uploaded in place: figure
paths are rewritten, every referenced figure is copied in, and the result is
compiled once from the staging directory to prove that it stands alone.

The bibliography is shipped as a ``.bbl``. arXiv will run BibTeX, but only if
the ``.bib`` resolves and the style file is present, and a stale or missing
bibliography is the single most common cause of a submission that builds
locally and fails there.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGURES = ROOT / "docs" / "figures"

# arXiv's metadata form truncates rather than rejects, so the limit is checked
# here instead of being discovered after submission.
ABSTRACT_LIMIT = 1920

_FIGURE_REFERENCE = re.compile(r"\{\.\./docs/figures/([^}]+)\}")


def _staged_source(text: str) -> tuple[str, list[str]]:
    """Return the manuscript with local figure paths, and the figures it needs."""
    wanted: list[str] = []

    def rewrite(match: re.Match[str]) -> str:
        name = match.group(1)
        wanted.append(name)
        return "{figures/" + name + "}"

    return _FIGURE_REFERENCE.sub(rewrite, text), wanted


def _plain_abstract(text: str) -> str:
    """Return the abstract as arXiv's metadata field will hold it."""
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if match is None:
        raise SystemExit("the manuscript has no abstract")
    body = match.group(1)
    body = re.sub(r"\\emph\{([^}]*)\}", r"\1", body)
    body = re.sub(r"\$[^$]*\$", "X", body)
    body = re.sub(r"\\[a-zA-Z]+", "", body)
    body = body.replace("{", "").replace("}", "").replace("---", "-")
    return re.sub(r"\s+", " ", body).strip()


def _stage(destination: Path) -> list[str]:
    """Copy every file the submission needs into ``destination``."""
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "figures").mkdir(parents=True)

    source, wanted = _staged_source((PAPER / "main.tex").read_text(encoding="utf-8"))
    (destination / "main.tex").write_text(source, encoding="utf-8")
    shutil.copy2(PAPER / "neurips_2023.sty", destination / "neurips_2023.sty")
    shutil.copy2(PAPER / "references.bib", destination / "references.bib")
    shutil.copytree(PAPER / "generated", destination / "generated")

    missing = [name for name in wanted if not (FIGURES / name).is_file()]
    if missing:
        raise SystemExit(
            "figures referenced by the manuscript are absent; run 'make figures' first: "
            + ", ".join(sorted(missing))
        )
    for name in sorted(set(wanted)):
        shutil.copy2(FIGURES / name, destination / "figures" / name)
    return sorted(set(wanted))


def _compile(destination: Path) -> None:
    """Compile the staged copy, keeping the bibliography arXiv will need."""
    result = subprocess.run(
        ["tectonic", "--keep-intermediates", "--keep-logs", "main.tex"],
        cwd=destination,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit("the staged submission does not compile on its own")
    overfull = sorted({line for line in result.stderr.splitlines() if "Overfull" in line})
    for line in overfull:
        sys.stderr.write(line + "\n")
    if not (destination / "main.bbl").is_file():
        raise SystemExit("no main.bbl was produced; arXiv would build an empty bibliography")


def _archive(destination: Path, archive: Path) -> None:
    """Write the tarball, excluding everything the compile left behind.

    arXiv rejects a submission carrying a PDF alongside the source, and ignores
    logs and auxiliary files, so only the inputs and the bibliography go in.
    """
    keep_suffixes = {".tex", ".sty", ".bib", ".bbl", ".pdf"}
    members: list[Path] = []
    for path in sorted(destination.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(destination)
        if relative.parts[0] == "figures":
            members.append(path)
            continue
        if path.suffix in keep_suffixes and path.suffix != ".pdf":
            members.append(path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        for path in members:
            bundle.add(path, arcname=str(path.relative_to(destination)))


def main(argv: Sequence[str] | None = None) -> int:
    """Stage, compile and archive the submission, then print its metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging", type=Path, default=ROOT / "paper" / "arxiv", help="staging directory"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "paper" / "arxiv-submission.tar.gz",
        help="tarball to write",
    )
    arguments = parser.parse_args(argv)

    figures = _stage(arguments.staging)
    _compile(arguments.staging)
    _archive(arguments.staging, arguments.archive)

    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract = _plain_abstract(text)
    title = re.search(r"\\title\{(.*?)\}\s*\n", text, re.S)
    size = arguments.archive.stat().st_size

    print(f"staged   {arguments.staging}")
    print(f"figures  {len(figures)}")
    print(f"archive  {arguments.archive} ({size / 1024:.0f} KB)")
    print()
    if title is not None:
        flat = re.sub(r"\s+", " ", title.group(1).replace("\\\\", " ")).strip()
        print(f"Title: {flat}")
    print(f"Abstract: {len(abstract)} characters (arXiv limit {ABSTRACT_LIMIT})")
    if len(abstract) > ABSTRACT_LIMIT:
        print("  OVER THE LIMIT: shorten the abstract before submitting.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
