"""Build every report figure from one immutable run bundle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bist_predict.figures import build_all_figures  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Render the figures and print what each one computed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="accepted run directory")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs" / "figures", help="output directory"
    )
    arguments = parser.parse_args(argv)
    manifest = build_all_figures(arguments.run, arguments.output)
    for record in manifest["figures"]:
        extras = {
            key: value for key, value in record.items() if key not in {"figure", "png", "pdf"}
        }
        print(f"{record['figure']}: {extras}")
    print(f"\n{manifest['figure_count']} figures written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
