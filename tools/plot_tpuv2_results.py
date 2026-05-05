"""Plot previously saved TPUv2 experiment CSV/Parquet results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tools.tpuv2_experiment import DEFAULT_CSV, DEFAULT_PLOT_DIR, _extract_case_tags, enrich_metrics, make_plots


def read_results(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots from TPUv2 experiment result CSV/Parquet")
    ap.add_argument("--input", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    ap.add_argument("--enriched-output", type=Path, default=None, help="optional CSV path for enriched metrics")
    args = ap.parse_args()

    df = read_results(args.input)
    if "group_from_case" not in df.columns:
        df = _extract_case_tags(df)
    needed = {"cycles_per_mac", "macs_per_cycle", "speedup_vs_baseline"}
    if not needed.issubset(df.columns):
        df = enrich_metrics(df)

    generated = make_plots(df, args.plot_dir)
    if args.enriched_output:
        args.enriched_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.enriched_output, index=False)
        print(f"wrote {args.enriched_output}")
    if generated:
        print("generated plots/tables:")
        for path in generated:
            print(f"  {path}")
    else:
        print("no plots generated because there were no ok rows")


if __name__ == "__main__":
    main()
