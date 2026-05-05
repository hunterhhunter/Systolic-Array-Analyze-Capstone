"""Public CLI: SCALE-Sim topology -> run + summary CSV/parquet."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any
import sys

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCALESIM_ROOT = REPO / "SCALE-Sim"
if SCALESIM_ROOT.exists() and str(SCALESIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALESIM_ROOT))

import configparser


from tools.aggregator import summarize_compute_report
from tools.linalg_mlir import display_path, repo_path
from tools.result_schema import RUN_EXPERIMENT_COLUMNS
from tools.io_utils import write_dataframe_outputs

DEFAULT_ARCH_CFG = REPO / "SCALE-Sim" / "configs" / "tpuv2.cfg"
DEFAULT_LAYOUT = REPO / "SCALE-Sim" / "layouts" / "conv_nets" / "test.csv"
RESULTS_DIR = REPO / "results"


def _count_topology_rows(path: Path) -> int:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return sum(1 for row in rows[1:] if any(cell.strip() for cell in row))


def _load_dataflow_and_run_name(arch_cfg: Path) -> tuple[str, str, str]:
    cp = configparser.ConfigParser()
    if not cp.read(arch_cfg):
        raise FileNotFoundError(arch_cfg)
    arch = cp["architecture_presets"]
    arr_h = arch.getint("ArrayHeight")
    arr_w = arch.getint("ArrayWidth")
    dataflow = arch.get("Dataflow")
    run_name = cp["general"].get("run_name", fallback=arch_cfg.stem)
    return dataflow, run_name, f"{arr_h}x{arr_w}_{dataflow}"


def default_results_path(run_name: str) -> Path:
    return RESULTS_DIR / f"{run_name}.parquet"


def run_scalesim_experiment(
    *,
    topology: Path,
    kind: str,
    arch_cfg: Path = DEFAULT_ARCH_CFG,
    layout: Path = DEFAULT_LAYOUT,
    output_root: Path,
    results_path: Path,
    run_name: str,
    clean: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    topology = repo_path(topology)
    arch_cfg = repo_path(arch_cfg)
    layout = repo_path(layout)
    output_root = repo_path(output_root)
    results_path = repo_path(results_path)

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataflow, scale_run_name, arch_label = _load_dataflow_and_run_name(arch_cfg)
    from scalesim.scale_sim import scalesim

    sim = scalesim(
        save_disk_space=True,
        verbose=verbose,
        config=str(arch_cfg),
        topology=str(topology),
        layout=str(layout),
        input_type_gemm=(kind == "matmul"),
    )
    sim.run_scale(top_path=str(output_root))
    report = output_root / scale_run_name / "COMPUTE_REPORT.csv"
    if not report.exists():
        raise RuntimeError(f"SCALE-Sim did not emit {report}")

    row: dict[str, Any] = summarize_compute_report(
        report,
        dataflow=dataflow,
        workload=run_name,
        arch=arch_label,
    )
    expected_tiles = _count_topology_rows(topology)
    row.update(
        {
            "run_name": run_name,
            "kind": kind,
            "topology_csv": display_path(topology),
            "arch_cfg": display_path(arch_cfg),
            "layout": layout.stem,
            "layout_path": display_path(layout),
            "compute_report": display_path(report),
            "status": "ok" if row["n_tiles"] == expected_tiles else "csv_drift",
        }
    )

    df = pd.DataFrame([row], columns=list(RUN_EXPERIMENT_COLUMNS))
    csv_path = results_path.with_suffix(".csv")
    write_dataframe_outputs(df, results_path, csv_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SCALE-Sim and summarize result metrics")
    ap.add_argument("--topology", type=Path)
    ap.add_argument("--run-dir", type=Path, help="uses <run-dir>/topology.csv and <run-dir>/sim")
    ap.add_argument("--kind", choices=("matmul", "conv2d"), required=True)
    ap.add_argument("--arch-cfg", type=Path, default=DEFAULT_ARCH_CFG)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.topology and not args.run_dir:
        raise SystemExit("--topology or --run-dir is required")
    run_dir = repo_path(args.run_dir) if args.run_dir else None
    topology = repo_path(args.topology) if args.topology else run_dir / "topology.csv"  # type: ignore[operator]
    run_name = args.name or (run_dir.name if run_dir else topology.stem)
    output_root = (
        repo_path(args.output_root)
        if args.output_root
        else (run_dir / "sim" if run_dir else REPO / "outputs" / "experiments" / run_name / "sim")
    )
    results = repo_path(args.results) if args.results else default_results_path(run_name)

    df = run_scalesim_experiment(
        topology=topology,
        kind=args.kind,
        arch_cfg=args.arch_cfg,
        layout=args.layout,
        output_root=output_root,
        results_path=results,
        run_name=run_name,
        clean=not args.no_clean,
        verbose=args.verbose,
    )
    print(f"wrote {results}")
    print(f"wrote {results.with_suffix('.csv')}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
