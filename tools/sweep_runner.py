"""General MatMul sweep runner.

Runs the full path for one or more MNK MatMul workloads:

    MNK + tile -> generated MLIR schedule -> iree-opt tiled MLIR
    -> SCALE-Sim GEMM topology -> SCALE-Sim run -> aggregated result row

The default arguments preserve the original Phase 4 smoke sweep
``32x32x64 x {8x8x8,16x16x16} x walking_8x8_ws``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from scalesim.scale_config import scale_config
from scalesim.scale_sim import scalesim

from tools.aggregator import aggregate, parse_compute_report, reuse_correction
from tools.iree_tile_mnk import (
    MnkTileSpec,
    render_merged_mlir,
    run_iree_transform,
    write_topology_from_spec,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
RESULTS_PARQUET = RESULTS_DIR / "results.parquet"
SWEEP_DIR = REPO / "outputs" / "small_sweep"
DEFAULT_ARCH_CFG = REPO / "SCALE-Sim" / "configs" / "walking_8x8_ws.cfg"
DEFAULT_LAYOUT = REPO / "SCALE-Sim" / "layouts" / "conv_nets" / "test.csv"

PARQUET_COLUMNS = (
    "workload",
    "m",
    "n",
    "k",
    "tile",
    "tile_m",
    "tile_n",
    "tile_k",
    "arch",
    "arch_cfg",
    "dataflow",
    "n_tiles",
    "compute_cycles",
    "total_cycles",
    "reuse_aware_cycles",
    "mean_overall_util_pct",
    "mean_mapping_eff_pct",
    "mean_compute_util_pct",
    "stall",
    "status",
)

Status = Literal["ok", "invalid_config", "iree_fail", "sim_fail", "csv_drift"]


@dataclass(frozen=True)
class TileShape:
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


@dataclass(frozen=True)
class MnkShape:
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"m{self.m}_n{self.n}_k{self.k}"

    @property
    def workload(self) -> str:
        return f"matmul_{self.label}"


@dataclass(frozen=True)
class ArchSpec:
    cfg: Path
    run_name: str
    label: str
    dataflow: str


@dataclass(frozen=True)
class SweepConfig:
    mnk: MnkShape
    tile_shape: TileShape
    arch_spec: ArchSpec

    @property
    def workload(self) -> str:
        return self.mnk.workload

    @property
    def tile(self) -> str:
        return self.tile_shape.label

    @property
    def arch(self) -> str:
        return self.arch_spec.label

    @property
    def arch_cfg(self) -> Path:
        return self.arch_spec.cfg

    @property
    def arch_run_name(self) -> str:
        return self.arch_spec.run_name

    @property
    def dataflow(self) -> str:
        return self.arch_spec.dataflow

    def to_mnk_tile_spec(self) -> MnkTileSpec:
        return MnkTileSpec(
            m=self.mnk.m,
            n=self.mnk.n,
            k=self.mnk.k,
            tile_m=self.tile_shape.m,
            tile_n=self.tile_shape.n,
            tile_k=self.tile_shape.k,
            name=self.workload,
        )


def parse_mnk(value: str) -> MnkShape:
    parts = _parse_dim_triplet(value, "MNK")
    return MnkShape(m=parts[0], n=parts[1], k=parts[2])


def parse_tile(value: str) -> TileShape:
    parts = _parse_dim_triplet(value, "tile")
    return TileShape(m=parts[0], n=parts[1], k=parts[2])


def _parse_dim_triplet(value: str, label: str) -> tuple[int, int, int]:
    raw = value.lower().replace(",", "x").split("x")
    if len(raw) != 3:
        raise argparse.ArgumentTypeError(f"{label} must be formatted like 32x32x64")
    try:
        dims = tuple(int(x) for x in raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{label} contains a non-integer: {value}") from e
    if any(x <= 0 for x in dims):
        raise argparse.ArgumentTypeError(f"{label} dimensions must be positive: {value}")
    return dims  # type: ignore[return-value]


def load_arch_spec(cfg_path: Path) -> ArchSpec:
    cfg_path = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
    cfg = scale_config()
    cfg.read_conf_file(str(cfg_path))
    arr_h, arr_w = cfg.get_array_dims()
    dataflow = cfg.get_dataflow()
    return ArchSpec(
        cfg=cfg_path,
        run_name=cfg.get_run_name(),
        label=f"{arr_h}x{arr_w}_{dataflow}",
        dataflow=dataflow,
    )


def build_configs(
    mnks: tuple[MnkShape, ...],
    tiles: tuple[TileShape, ...],
    arch_cfgs: tuple[Path, ...],
) -> tuple[SweepConfig, ...]:
    archs = tuple(load_arch_spec(path) for path in arch_cfgs)
    return tuple(
        SweepConfig(mnk=mnk, tile_shape=tile, arch_spec=arch)
        for mnk in mnks
        for tile in tiles
        for arch in archs
    )


DEFAULT_CONFIGS = build_configs(
    mnks=(MnkShape(32, 32, 64),),
    tiles=(TileShape(8, 8, 8), TileShape(16, 16, 16)),
    arch_cfgs=(DEFAULT_ARCH_CFG,),
)


def _empty_row(cfg: SweepConfig, status: Status) -> dict:
    return {
        "workload": cfg.workload,
        "m": cfg.mnk.m,
        "n": cfg.mnk.n,
        "k": cfg.mnk.k,
        "tile": cfg.tile,
        "tile_m": cfg.tile_shape.m,
        "tile_n": cfg.tile_shape.n,
        "tile_k": cfg.tile_shape.k,
        "arch": cfg.arch,
        "arch_cfg": str(cfg.arch_cfg.relative_to(REPO) if cfg.arch_cfg.is_relative_to(REPO) else cfg.arch_cfg),
        "dataflow": cfg.dataflow,
        "n_tiles": 0,
        "compute_cycles": 0,
        "total_cycles": 0,
        "reuse_aware_cycles": 0.0,
        "mean_overall_util_pct": 0.0,
        "mean_mapping_eff_pct": 0.0,
        "mean_compute_util_pct": 0.0,
        "stall": 0,
        "status": status,
    }


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("->", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _generate_and_tile(cfg: SweepConfig, input_mlir: Path, tiled_mlir: Path) -> Status | None:
    try:
        input_mlir.parent.mkdir(parents=True, exist_ok=True)
        input_mlir.write_text(render_merged_mlir(cfg.to_mnk_tile_spec()))
        run_iree_transform(input_mlir, tiled_mlir)
    except ValueError:
        return "invalid_config"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "iree_fail"
    return None


def _emit_topology(cfg: SweepConfig, out_csv: Path) -> tuple[int, Status | None]:
    try:
        n_tiles = write_topology_from_spec(cfg.to_mnk_tile_spec(), out_csv)
    except Exception:
        return 0, "csv_drift"
    return n_tiles, None


def _scalesim(cfg: SweepConfig, topology: Path, sim_dir: Path) -> tuple[Path | None, Status | None]:
    if not cfg.arch_cfg.exists():
        return None, "sim_fail"
    sim_dir.mkdir(parents=True, exist_ok=True)
    try:
        sim = scalesim(
            save_disk_space=True,
            verbose=False,
            config=str(cfg.arch_cfg),
            topology=str(topology),
            layout=str(DEFAULT_LAYOUT),
            input_type_gemm=True,
        )
        sim.run_scale(top_path=str(sim_dir))
    except Exception:
        return None, "sim_fail"

    report = sim_dir / cfg.arch_run_name / "COMPUTE_REPORT.csv"
    if not report.exists():
        return None, "sim_fail"
    return report, None


def _build_ok_row(cfg: SweepConfig, n_tiles_expected: int, report_path: Path) -> dict:
    records = parse_compute_report(report_path)
    if len(records) != n_tiles_expected:
        raise ValueError(
            f"{cfg.workload}/{cfg.tile}/{cfg.arch}: SCALE-Sim emitted {len(records)} rows, "
            f"expected {n_tiles_expected}"
        )
    compute = aggregate(records, metric_kind="compute")
    total = aggregate(records, metric_kind="total")
    reuse = reuse_correction(records, dataflow=cfg.dataflow)
    row = _empty_row(cfg, "ok")
    row.update(
        {
            "n_tiles": compute["n_tiles"],
            "compute_cycles": compute["sum_cycles"],
            "total_cycles": total["sum_cycles"],
            "reuse_aware_cycles": float(sum(reuse)),
            "mean_overall_util_pct": compute["mean_overall_util_pct"],
            "mean_mapping_eff_pct": compute["mean_mapping_eff_pct"],
            "mean_compute_util_pct": compute["mean_compute_util_pct"],
            "stall": compute["sum_stall"],
        }
    )
    return row


def run_config(cfg: SweepConfig, root: Path = SWEEP_DIR) -> dict:
    run_dir = root / cfg.workload / cfg.arch / cfg.dataflow / cfg.tile
    input_mlir = run_dir / "input_with_schedule.mlir"
    tiled = run_dir / "tiled.mlir"
    topology = run_dir / "topology.csv"
    sim_dir = run_dir / "sim"

    if status := _generate_and_tile(cfg, input_mlir, tiled):
        return _empty_row(cfg, status)

    n_tiles, status = _emit_topology(cfg, topology)
    if status:
        return _empty_row(cfg, status)

    report, status = _scalesim(cfg, topology, sim_dir)
    if status or report is None:
        return _empty_row(cfg, status or "sim_fail")

    try:
        return _build_ok_row(cfg, n_tiles, report)
    except ValueError:
        return _empty_row(cfg, "csv_drift")


def run_sweep(
    configs: tuple[SweepConfig, ...] = DEFAULT_CONFIGS,
    out_path: Path = RESULTS_PARQUET,
    output_root: Path = SWEEP_DIR,
    clean: bool = True,
    csv_out: Path | None = None,
) -> pd.DataFrame:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [run_config(cfg, root=output_root) for cfg in configs]
    df = pd.DataFrame(rows, columns=list(PARQUET_COLUMNS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a general MatMul MNK/tile/HW sweep")
    ap.add_argument("--mnk", type=parse_mnk, action="append", default=None, help="MNK shape, e.g. 32x32x64")
    ap.add_argument("--tiles", type=parse_tile, nargs="+", default=None, help="tile shapes, e.g. 8x8x8 16x16x16")
    ap.add_argument("--arch-cfg", type=Path, nargs="+", default=None, help="SCALE-Sim config paths")
    ap.add_argument("--output", type=Path, default=RESULTS_PARQUET)
    ap.add_argument("--csv-output", type=Path, help="optional CSV summary output")
    ap.add_argument("--output-root", type=Path, default=SWEEP_DIR)
    ap.add_argument("--no-clean", action="store_true", help="keep existing output root")
    args = ap.parse_args()

    mnks = tuple(args.mnk) if args.mnk else (MnkShape(32, 32, 64),)
    tiles = tuple(args.tiles) if args.tiles else (TileShape(8, 8, 8), TileShape(16, 16, 16))
    arch_cfgs = tuple(args.arch_cfg) if args.arch_cfg else (DEFAULT_ARCH_CFG,)
    configs = build_configs(mnks=mnks, tiles=tiles, arch_cfgs=arch_cfgs)

    df = run_sweep(
        configs=configs,
        out_path=args.output,
        output_root=args.output_root,
        clean=not args.no_clean,
        csv_out=args.csv_output,
    )
    print(f"\nwrote {args.output} ({len(df)} rows x {len(df.columns)} cols)")
    if args.csv_output:
        print(f"wrote {args.csv_output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
