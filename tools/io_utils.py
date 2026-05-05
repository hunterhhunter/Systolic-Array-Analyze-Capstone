"""Small output helpers shared by experiment CLIs.

The repository often writes both Parquet and CSV summaries.  Pandas requires
an optional engine (pyarrow or fastparquet) for Parquet; CSV should still be
produced when it was explicitly requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def write_dataframe_outputs(
    df: pd.DataFrame,
    output: Path | str | None,
    csv_output: Path | str | None = None,
    *,
    require_parquet: bool = False,
) -> list[Path]:
    """Write a DataFrame to CSV/Parquet with robust optional Parquet handling.

    CSV is written first so `--csv-output` remains useful on minimal
    environments that do not have a Parquet engine installed.  When Parquet is
    requested but unavailable, this function raises only if no CSV fallback was
    requested or `require_parquet=True`.
    """

    written: list[Path] = []

    if csv_output is not None:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        written.append(csv_path)

    if output is None:
        return written

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        if not written or out_path.resolve() not in {x.resolve() for x in written}:
            df.to_csv(out_path, index=False)
            written.append(out_path)
        return written

    try:
        df.to_parquet(out_path, index=False)
        written.append(out_path)
    except ImportError:
        if require_parquet or csv_output is None:
            raise
        print(f"warning: skipped Parquet output {out_path} because pyarrow/fastparquet is not installed")

    return written
