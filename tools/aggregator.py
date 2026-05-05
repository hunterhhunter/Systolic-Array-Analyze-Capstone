"""Per-tile SCALE-Sim COMPUTE_REPORT → layer/model rollup (master_plan §3-3).

L4 contract (frozen):
  COMPUTE_REPORT.csv columns =
    [0] LayerID
    [1] Total Cycles (incl. prefetch)   ← prefetch+ (row[1])
    [2] Total Cycles                    ← compute-only (row[2])
    [3] Stall Cycles
    [4] Overall Util %                  (float)
    [5] Mapping Efficiency %            (float)
    [6] Compute Util %                  (float)
  Each data line ends with a trailing comma (SCALE-Sim quirk).

Phase 3 게이트는 parse + dual-metric aggregate. reuse_correction은
tools.reuse_model 의 placeholder 수식을 호출 (Phase 6에서 fit 예정).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Literal, NamedTuple


MetricKind = Literal["compute", "total"]


class TileRecord(NamedTuple):
    layer_id: int
    total_cycles_incl_prefetch: int
    total_cycles_compute: int
    stall_cycles: int
    overall_util_pct: float
    mapping_eff_pct: float
    compute_util_pct: float


_REQUIRED_HEADER_TOKENS = (
    "LayerID",
    "Total Cycles (incl. prefetch)",
    "Total Cycles",
    "Stall Cycles",
    "Overall Util %",
    "Mapping Efficiency %",
    "Compute Util %",
)


def _strip_cells(row: list[str]) -> list[str]:
    return [c.strip() for c in row if c.strip() != ""]


def parse_compute_report(path: Path | str) -> list[TileRecord]:
    """Parse a SCALE-Sim COMPUTE_REPORT.csv into per-tile records.

    Raises ValueError if the header does not match the L4-locked schema.
    Empty rows (e.g. trailing newline noise) are skipped.
    """
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty COMPUTE_REPORT: {path}")
        header_cells = _strip_cells(header)
        if len(header_cells) < 7:
            raise ValueError(
                f"unexpected COMPUTE_REPORT header (too few columns): {header_cells!r}"
            )
        for i, expected in enumerate(_REQUIRED_HEADER_TOKENS):
            if header_cells[i] != expected:
                raise ValueError(
                    f"COMPUTE_REPORT header drift at column {i}: "
                    f"expected {expected!r}, got {header_cells[i]!r}"
                )

        records: list[TileRecord] = []
        for row in reader:
            cells = _strip_cells(row)
            if not cells:
                continue
            if len(cells) < 7:
                raise ValueError(f"COMPUTE_REPORT row too short: {cells!r}")
            records.append(
                TileRecord(
                    layer_id=int(cells[0]),
                    total_cycles_incl_prefetch=int(cells[1]),
                    total_cycles_compute=int(cells[2]),
                    stall_cycles=int(cells[3]),
                    overall_util_pct=float(cells[4]),
                    mapping_eff_pct=float(cells[5]),
                    compute_util_pct=float(cells[6]),
                )
            )
    return records


def aggregate(
    tile_records: Iterable[TileRecord], metric_kind: MetricKind
) -> dict:
    """Layer-level rollup for a single cycle metric.

    metric_kind='compute' → sum of row[2] (compute-only).
    metric_kind='total'   → sum of row[1] (incl. prefetch).
    The two MUST stay separate (Phase 6 figures plot both columns).
    """
    recs = list(tile_records)
    if not recs:
        raise ValueError("aggregate: no tile records")
    if metric_kind == "compute":
        sum_cycles = sum(r.total_cycles_compute for r in recs)
    elif metric_kind == "total":
        sum_cycles = sum(r.total_cycles_incl_prefetch for r in recs)
    else:
        raise ValueError(f"unknown metric_kind: {metric_kind!r}")
    n = len(recs)
    return {
        "metric_kind": metric_kind,
        "sum_cycles": sum_cycles,
        "sum_stall": sum(r.stall_cycles for r in recs),
        "mean_overall_util_pct": sum(r.overall_util_pct for r in recs) / n,
        "mean_mapping_eff_pct": sum(r.mapping_eff_pct for r in recs) / n,
        "mean_compute_util_pct": sum(r.compute_util_pct for r in recs) / n,
        "n_tiles": n,
    }


def reuse_correction(
    tile_records: Iterable[TileRecord], dataflow: str = "ws", fold_fraction: float | None = None
) -> list[float]:
    """Wire to tools.reuse_model — heuristic neighbor-tile reuse correction."""
    from tools.reuse_model import DEFAULT_FOLD_FRACTION_WS, reuse_aware_cycles

    effective_fold = DEFAULT_FOLD_FRACTION_WS if fold_fraction is None else fold_fraction
    return reuse_aware_cycles(list(tile_records), dataflow=dataflow, fold_fraction=effective_fold)


def summarize_compute_report(
    path: Path | str,
    dataflow: str = "ws",
    workload: str | None = None,
    arch: str | None = None,
) -> dict:
    """Summarize any SCALE-Sim COMPUTE_REPORT.csv.

    This is intentionally topology-agnostic: callers can pass metadata such as
    workload/arch when available, but the core rollup only depends on the report.
    """
    records = parse_compute_report(path)
    compute = aggregate(records, metric_kind="compute")
    total = aggregate(records, metric_kind="total")
    reuse = reuse_correction(records, dataflow=dataflow)
    from tools.reuse_model import DEFAULT_FOLD_FRACTION_WS, REUSE_MODEL_NAME

    reuse_sum = float(sum(reuse))
    return {
        "workload": workload or Path(path).parent.name,
        "arch": arch or Path(path).parent.name,
        "dataflow": dataflow,
        "n_tiles": compute["n_tiles"],
        "compute_cycles": compute["sum_cycles"],
        "total_cycles": total["sum_cycles"],
        "reuse_aware_cycles": reuse_sum,
        "reuse_aware_cycles_est": reuse_sum,
        "reuse_model_calibrated": False,
        "reuse_model": REUSE_MODEL_NAME,
        "reuse_fold_fraction": DEFAULT_FOLD_FRACTION_WS,
        "mean_overall_util_pct": compute["mean_overall_util_pct"],
        "mean_mapping_eff_pct": compute["mean_mapping_eff_pct"],
        "mean_compute_util_pct": compute["mean_compute_util_pct"],
        "stall": compute["sum_stall"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a SCALE-Sim COMPUTE_REPORT.csv")
    ap.add_argument("compute_report", type=Path)
    ap.add_argument("--dataflow", default="ws")
    ap.add_argument("--workload")
    ap.add_argument("--arch")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = ap.parse_args()

    row = summarize_compute_report(
        args.compute_report,
        dataflow=args.dataflow,
        workload=args.workload,
        arch=args.arch,
    )
    import pandas as pd

    df = pd.DataFrame([row])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from tools.io_utils import write_dataframe_outputs

        write_dataframe_outputs(
            df,
            args.output,
            None if args.format == "parquet" else args.output,
            require_parquet=(args.format == "parquet"),
        )
        print(f"wrote {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
