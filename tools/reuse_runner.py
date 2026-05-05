"""CLI for applying the analytic reuse correction to SCALE-Sim reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tools.io_utils import write_dataframe_outputs
from tools.aggregator import parse_compute_report, summarize_compute_report
from tools.reuse_model import REUSE_MODEL_NAME, reuse_aware_cycles


def build_reuse_dataframe(
    compute_report: Path | str,
    dataflow: str = "ws",
    fold_fraction: float = 0.5,
) -> pd.DataFrame:
    records = parse_compute_report(compute_report)
    corrected = reuse_aware_cycles(
        records, dataflow=dataflow, fold_fraction=fold_fraction
    )
    rows = []
    for rec, reuse_cycles in zip(records, corrected):
        prefetch = rec.total_cycles_incl_prefetch - rec.total_cycles_compute
        rows.append(
            {
                "layer_id": rec.layer_id,
                "compute_cycles": rec.total_cycles_compute,
                "total_cycles": rec.total_cycles_incl_prefetch,
                "prefetch_cycles": prefetch,
                "reuse_aware_cycles": reuse_cycles,
                "reuse_aware_cycles_est": reuse_cycles,
                "reuse_model": REUSE_MODEL_NAME,
                "reuse_fold_fraction": fold_fraction,
                "overall_util_pct": rec.overall_util_pct,
                "mapping_eff_pct": rec.mapping_eff_pct,
                "compute_util_pct": rec.compute_util_pct,
            }
        )
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        write_dataframe_outputs(df, path, None, require_parquet=True)
    else:
        write_dataframe_outputs(df, path, path)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply reuse-aware correction to a COMPUTE_REPORT")
    ap.add_argument("compute_report", type=Path)
    ap.add_argument("--dataflow", default="ws")
    ap.add_argument("--fold-fraction", type=float, default=0.5)
    ap.add_argument("--output", type=Path, help="per-tile output .csv or .parquet")
    ap.add_argument("--summary-output", type=Path, help="one-row summary .csv or .parquet")
    args = ap.parse_args()

    df = build_reuse_dataframe(
        args.compute_report,
        dataflow=args.dataflow,
        fold_fraction=args.fold_fraction,
    )
    if args.output:
        _write(df, args.output)
    else:
        print(df.to_string(index=False))

    if args.summary_output:
        summary = summarize_compute_report(args.compute_report, dataflow=args.dataflow)
        summary["reuse_fold_fraction"] = args.fold_fraction
        summary_df = pd.DataFrame([summary])
        _write(summary_df, args.summary_output)


if __name__ == "__main__":
    main()
