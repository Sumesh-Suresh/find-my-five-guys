"""Combine per-commit scores into per-contributor totals.

Reads a results CSV produced by `evaluate.py` (columns:
committer_datetime, author_name, score, reason) and writes a new CSV
with one row per distinct contributor, where the sum of `score` becomes
`points`.

Usage:
    python combine_contributor.py
    python combine_contributor.py --input initial-gemini-flash3.1.csv
    python combine_contributor.py --input initial-gemini-flash3.1.csv \
        --output combined-initial-gemini-flash3.1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_INPUT = Path("initial-gemini-flash3.1.csv")


def combine(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

    combined = (
        df.groupby("author_name", as_index=False)
        .agg(points=("score", "sum"), commits=("score", "size"))
        .sort_values(["points", "commits"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return combined


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    output_path = args.output or args.input.with_name(f"combined-{args.input.name}")

    combined = combine(args.input)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} contributor row(s) to {output_path.resolve()}")


if __name__ == "__main__":
    main()
