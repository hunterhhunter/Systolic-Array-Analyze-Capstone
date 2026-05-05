"""Demo wrapper around the general MatMul sweep runner."""

from __future__ import annotations

from pathlib import Path

from tools.sweep_runner import DEFAULT_CONFIGS, REPO, run_sweep

DEMO_DIR = REPO / "outputs" / "demo"
RESULTS_PARQUET = REPO / "results" / "demo.parquet"
RESULTS_CSV = REPO / "results" / "demo.csv"


def main() -> None:
    df = run_sweep(
        configs=DEFAULT_CONFIGS,
        out_path=RESULTS_PARQUET,
        output_root=DEMO_DIR,
        clean=True,
        csv_out=RESULTS_CSV,
        fail_fast=True,
        require_any_ok=True,
    )
    print(f"\nwrote {RESULTS_PARQUET} ({len(df)} rows x {len(df.columns)} cols)")
    print(f"wrote {RESULTS_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
