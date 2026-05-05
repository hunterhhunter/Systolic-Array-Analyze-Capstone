"""Rebuild plots for the broad deep-model TPUv2 experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tools.deep_model_tpuv2_experiment import DEFAULT_CSV, DEFAULT_ENRICHED_CSV, DEFAULT_PLOT_DIR, _add_deep_tags, make_deep_plots
from tools.tpuv2_experiment import enrich_metrics


def read_results(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots from deep-model TPUv2 result CSV/Parquet")
    ap.add_argument("--input", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    ap.add_argument("--enriched-output", type=Path, default=DEFAULT_ENRICHED_CSV)
    args = ap.parse_args()

    df = read_results(args.input)
    df = _add_deep_tags(enrich_metrics(df))
    args.enriched_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.enriched_output, index=False)
    print(f"wrote {args.enriched_output}")
    generated = make_deep_plots(df, args.plot_dir)
    if generated:
        print("generated plots/tables:")
        for path in generated:
            print(f"  {path}")
    else:
        print("no plots generated because there were no ok rows")


if __name__ == "__main__":
    main()
