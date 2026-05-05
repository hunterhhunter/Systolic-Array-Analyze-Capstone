"""General MatMul sweep runner.

Runs the full path for one or more MNK MatMul workloads:

    MNK + tile -> generated MLIR schedule -> iree-opt tiled MLIR
    -> SCALE-Sim GEMM topology -> SCALE-Sim run -> aggregated result row

The default arguments preserve the original Phase 4 smoke sweep
``32x32x64 x {8x8x8,16x16x16} x walking_8x8_ws``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import shutil
import subprocess
import sys
import configparser
import re
import traceback
import time
import threading
from datetime import datetime

import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parent.parent
SCALESIM_ROOT = REPO / "SCALE-Sim"
if SCALESIM_ROOT.exists() and str(SCALESIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALESIM_ROOT))


from tools.aggregator import aggregate, parse_compute_report, reuse_correction, TileRecord
from tools.result_schema import SWEEP_COLUMNS
from tools.io_utils import write_dataframe_outputs
from tools.iree_tile_mnk import (
    MnkTileSpec,
    render_merged_mlir,
    run_iree_transform,
    write_topology_from_spec,
)

RESULTS_DIR = REPO / "results"
RESULTS_PARQUET = RESULTS_DIR / "results.parquet"
SWEEP_DIR = REPO / "outputs" / "small_sweep"
DEFAULT_ARCH_CFG = REPO / "SCALE-Sim" / "configs" / "walking_8x8_ws.cfg"
DEFAULT_LAYOUT = REPO / "SCALE-Sim" / "layouts" / "conv_nets" / "test.csv"
DEFAULT_CACHE_ROOT = REPO / "outputs" / "cache" / "sweep_runner"

PARQUET_COLUMNS = SWEEP_COLUMNS


Status = Literal["ok", "invalid_config", "iree_fail", "sim_fail", "csv_drift"]
ParallelBackend = Literal["thread", "process"]
TopologyMode = Literal["raw", "grouped_full"]


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[{_now()}] {message}", file=sys.__stdout__, flush=True)


def _cfg_label(cfg: "SweepConfig") -> str:
    return (
        f"{cfg.case_name} | mnk={cfg.mnk.label} | tile={cfg.tile} | "
        f"array={cfg.arch_spec.array_h}x{cfg.arch_spec.array_w} | "
        f"bw={cfg.arch_spec.bandwidth} | "
        f"sram=({cfg.arch_spec.ifmap_sram_kb}/{cfg.arch_spec.filter_sram_kb}/{cfg.arch_spec.ofmap_sram_kb})KB | "
        f"layout={cfg.layout}"
    )


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
    array_h: int
    array_w: int
    bandwidth: int
    ifmap_sram_kb: int
    filter_sram_kb: int
    ofmap_sram_kb: int
    ifmap_custom_layout: bool
    filter_custom_layout: bool


@dataclass(frozen=True)
class LayoutSpec:
    path: Path
    label: str


@dataclass(frozen=True)
class SweepConfig:
    mnk: MnkShape
    tile_shape: TileShape
    arch_spec: ArchSpec
    layout_spec: LayoutSpec = LayoutSpec(path=DEFAULT_LAYOUT, label="default")
    case_name: str = "default"

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

    @property
    def layout(self) -> str:
        return self.layout_spec.label

    @property
    def layout_path(self) -> Path:
        return self.layout_spec.path

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
    cp = configparser.ConfigParser()
    if not cp.read(cfg_path):
        raise FileNotFoundError(cfg_path)
    arch = cp["architecture_presets"]
    general = cp["general"]
    layout = cp["layout"] if "layout" in cp else {}
    arr_h = arch.getint("ArrayHeight")
    arr_w = arch.getint("ArrayWidth")
    dataflow = arch.get("Dataflow")
    return ArchSpec(
        cfg=cfg_path,
        run_name=general.get("run_name", fallback=cfg_path.stem),
        label=f"{arr_h}x{arr_w}_{dataflow}",
        dataflow=dataflow,
        array_h=arr_h,
        array_w=arr_w,
        bandwidth=arch.getint("Bandwidth", fallback=0),
        ifmap_sram_kb=arch.getint("IfmapSramSzkB", fallback=0),
        filter_sram_kb=arch.getint("FilterSramSzkB", fallback=0),
        ofmap_sram_kb=arch.getint("OfmapSramSzkB", fallback=0),
        ifmap_custom_layout=str(layout.get("IfmapCustomLayout", "False")).lower() == "true",
        filter_custom_layout=str(layout.get("FilterCustomLayout", "False")).lower() == "true",
    )


def load_layout_spec(path: Path | None) -> LayoutSpec:
    layout_path = DEFAULT_LAYOUT if path is None else (path if path.is_absolute() else REPO / path)
    return LayoutSpec(path=layout_path, label=layout_path.stem)


def build_configs(
    mnks: tuple[MnkShape, ...],
    tiles: tuple[TileShape, ...],
    arch_cfgs: tuple[Path, ...],
    layouts: tuple[Path, ...] | None = None,
    case_name: str = "default",
) -> tuple[SweepConfig, ...]:
    archs = tuple(load_arch_spec(path) for path in arch_cfgs)
    layout_specs = tuple(load_layout_spec(path) for path in layouts) if layouts else (load_layout_spec(None),)
    return tuple(
        SweepConfig(mnk=mnk, tile_shape=tile, arch_spec=arch, layout_spec=layout, case_name=case_name)
        for mnk in mnks
        for tile in tiles
        for arch in archs
        for layout in layout_specs
    )


DEFAULT_CONFIGS = build_configs(
    mnks=(MnkShape(32, 32, 64),),
    tiles=(TileShape(8, 8, 8), TileShape(16, 16, 16)),
    arch_cfgs=(DEFAULT_ARCH_CFG,),
)


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _empty_row(
    cfg: SweepConfig,
    status: Status,
    *,
    input_mlir: Path | None = None,
    tiled_mlir: Path | None = None,
    topology: Path | None = None,
    sim_dir: Path | None = None,
    compute_report: Path | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    error_stage: str = "",
    error_message: str = "",
) -> dict:
    return {
        "workload": cfg.workload,
        "m": cfg.mnk.m,
        "n": cfg.mnk.n,
        "k": cfg.mnk.k,
        "tile": cfg.tile,
        "tile_m": cfg.tile_shape.m,
        "tile_n": cfg.tile_shape.n,
        "tile_k": cfg.tile_shape.k,
        "case_name": cfg.case_name,
        "arch": cfg.arch,
        "arch_cfg": _display_path(cfg.arch_cfg),
        "array_h": cfg.arch_spec.array_h,
        "array_w": cfg.arch_spec.array_w,
        "bandwidth": cfg.arch_spec.bandwidth,
        "ifmap_sram_kb": cfg.arch_spec.ifmap_sram_kb,
        "filter_sram_kb": cfg.arch_spec.filter_sram_kb,
        "ofmap_sram_kb": cfg.arch_spec.ofmap_sram_kb,
        "dataflow": cfg.dataflow,
        "layout": cfg.layout,
        "layout_path": _display_path(cfg.layout_path),
        "ifmap_custom_layout": cfg.arch_spec.ifmap_custom_layout,
        "filter_custom_layout": cfg.arch_spec.filter_custom_layout,
        "input_mlir_path": _display_path(input_mlir),
        "tiled_mlir_path": _display_path(tiled_mlir),
        "topology_path": _display_path(topology),
        "sim_dir": _display_path(sim_dir),
        "compute_report_path": _display_path(compute_report),
        "stdout_path": _display_path(stdout_path),
        "stderr_path": _display_path(stderr_path),
        "error_stage": error_stage,
        "error_message": error_message[:1000],
        "n_tiles": 0,
        "compute_cycles": 0,
        "total_cycles": 0,
        "reuse_aware_cycles": 0.0,
        "reuse_aware_cycles_est": 0.0,
        "reuse_model": "",
        "reuse_model_calibrated": False,
        "reuse_fold_fraction": 0.0,
        "mean_overall_util_pct": 0.0,
        "mean_mapping_eff_pct": 0.0,
        "mean_compute_util_pct": 0.0,
        "stall": 0,
        "memory_overhead_cycles": 0,
        "memory_overhead_ratio": 0.0,
        "logical_tiles": 0,
        "simulated_tiles": 0,
        "raw_topology_rows": 0,
        "topology_mode": "",
        "cache_key": "",
        "cache_status": "",
        "status": status,
    }


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("->", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _generate_and_tile(cfg: SweepConfig, input_mlir: Path, tiled_mlir: Path, stdout_path: Path, stderr_path: Path) -> tuple[Status | None, str]:
    try:
        input_mlir.parent.mkdir(parents=True, exist_ok=True)
        input_mlir.write_text(render_merged_mlir(cfg.to_mnk_tile_spec()), encoding="utf-8")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            run_iree_transform(input_mlir, tiled_mlir, stdout=out, stderr=err)
    except ValueError as exc:
        return "invalid_config", str(exc)
    except FileNotFoundError as exc:
        return "iree_fail", str(exc)
    except subprocess.CalledProcessError as exc:
        details = ""
        if stderr_path.exists():
            details = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        return "iree_fail", details or str(exc)
    return None, ""


def _dim_chunk_counts(length: int, tile: int) -> list[tuple[int, int]]:
    """Return ``[(actual_tile_size, multiplicity), ...]`` for one tiled dim.

    This is equivalent to iterating ``range(0, length, tile)`` and taking
    ``min(tile, remaining)``, but it is O(1) for divisible large LLM GEMMs.
    """
    full, rem = divmod(length, tile)
    chunks: list[tuple[int, int]] = []
    if full:
        chunks.append((tile, full))
    if rem:
        chunks.append((rem, 1))
    return chunks


def _tile_group_counts(cfg: SweepConfig) -> list[tuple[tuple[int, int, int], int]]:
    """Return grouped full-topology tile shapes and multiplicities.

    This represents the same MNK tiling space as ``write_topology_from_spec``
    without materializing every tile. Divisible TPUv2 and LLM GEMM cases collapse
    to one group. Non-divisible edge cases produce at most eight groups for GEMM:
    interior plus edge/corner groups across M/N/K.
    """
    spec = cfg.to_mnk_tile_spec()
    groups: list[tuple[tuple[int, int, int], int]] = []
    for m, m_count in _dim_chunk_counts(spec.m, spec.tile_m):
        for n, n_count in _dim_chunk_counts(spec.n, spec.tile_n):
            for k, k_count in _dim_chunk_counts(spec.k, spec.tile_k):
                groups.append(((m, n, k), m_count * n_count * k_count))
    return sorted(groups, key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[0]))


def _write_grouped_full_topology(cfg: SweepConfig, out_csv: Path) -> list[int]:
    groups = _tile_group_counts(cfg)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Layer Name, M, N, K,"]
    for idx, ((m, n, k), count) in enumerate(groups):
        lines.append(f"Group_{idx:03d}_x{count}, {m}, {n}, {k},")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [count for _, count in groups]


def _expected_simulated_tiles(cfg: SweepConfig, topology_mode: TopologyMode) -> int:
    if topology_mode == "grouped_full":
        return len(_tile_group_counts(cfg))
    return cfg.to_mnk_tile_spec().expected_tiles


def _expected_group_counts(cfg: SweepConfig, topology_mode: TopologyMode) -> list[int] | None:
    if topology_mode == "grouped_full":
        return [count for _, count in _tile_group_counts(cfg)]
    return None


def _emit_topology(cfg: SweepConfig, out_csv: Path, *, topology_mode: TopologyMode = "raw") -> tuple[int, int, int, list[int] | None, Status | None, str]:
    """Write SCALE-Sim topology.

    Returns ``(simulated_rows, logical_tiles, raw_topology_rows, group_counts,
    status, message)``.

    ``raw`` writes every full-topology tile row and sends all of them to
    SCALE-Sim. ``grouped_full`` still derives the complete full-topology tiling,
    but sends only unique GEMM tile shapes to SCALE-Sim and records each group
    multiplicity. Additive metrics are reconstructed by multiplying each group
    result by its multiplicity; utilization percentages are multiplicity-weighted.
    """
    try:
        spec = cfg.to_mnk_tile_spec()
        logical_tiles = spec.expected_tiles
        if topology_mode == "grouped_full":
            group_counts = _write_grouped_full_topology(cfg, out_csv)
            simulated_tiles = len(group_counts)
        else:
            simulated_tiles = write_topology_from_spec(spec, out_csv)
            group_counts = None
    except Exception as exc:
        return 0, 0, 0, None, "csv_drift", str(exc)
    return simulated_tiles, logical_tiles, logical_tiles, group_counts, None, ""

def _is_divisible_tiling(cfg: SweepConfig) -> bool:
    return (
        cfg.mnk.m % cfg.tile_shape.m == 0
        and cfg.mnk.n % cfg.tile_shape.n == 0
        and cfg.mnk.k % cfg.tile_shape.k == 0
    )


def _scalesim(
    cfg: SweepConfig,
    topology: Path,
    sim_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    verbose: bool = False,
    heartbeat_sec: int = 30,
) -> tuple[Path | None, Status | None, str]:
    """Run SCALE-Sim and return its COMPUTE_REPORT path.

    Do not use ``contextlib.redirect_stdout`` / ``redirect_stderr`` here.
    Those APIs mutate process-global ``sys.stdout`` and ``sys.stderr`` and are
    not safe under ThreadPoolExecutor. The previous version redirected each
    SCALE-Sim call into per-case files; when multiple workers overlapped, one
    worker could restore or close a stream while another worker still pointed at
    it, causing ``ValueError: I/O operation on closed file`` and ``lost
    sys.stderr`` after all cases had already completed.

    SCALE-Sim is now called in-process without global stream redirection. We
    still create per-case stdout/stderr files with explanatory breadcrumbs and
    capture Python exceptions there. This preserves thread parallelism and
    avoids corrupting the parent's standard streams.
    """
    if not cfg.arch_cfg.exists():
        return None, "sim_fail", f"architecture config not found: {cfg.arch_cfg}"
    if not cfg.layout_path.exists():
        return None, "sim_fail", f"layout CSV not found: {cfg.layout_path}"
    if not topology.exists():
        return None, "sim_fail", f"topology CSV not found: {topology}"

    sim_dir.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(
        "SCALE-Sim executed in-process without redirecting sys.stdout/sys.stderr.\n"
        "This avoids thread-unsafe global stdio redirection during parallel sweeps.\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")

    try:
        from scalesim.scale_sim import scalesim

        sim = scalesim(
            save_disk_space=True,
            verbose=False,
            config=str(cfg.arch_cfg),
            topology=str(topology),
            layout=str(cfg.layout_path),
            input_type_gemm=True,
        )
        stop_heartbeat = threading.Event()
        sim_start = time.perf_counter()

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(max(1, heartbeat_sec)):
                elapsed = time.perf_counter() - sim_start
                _log(
                    f"SIM-RUNNING elapsed={elapsed:.1f}s {_cfg_label(cfg)} "
                    f"stdout={_display_path(stdout_path)} stderr={_display_path(stderr_path)}",
                    enabled=verbose,
                )

        heartbeat_thread = None
        if verbose and heartbeat_sec > 0:
            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()
        try:
            sim.run_scale(top_path=str(sim_dir))
        finally:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2)
    except Exception as exc:
        with stderr_path.open("a", encoding="utf-8") as err:
            err.write("\n--- Python exception ---\n")
            traceback.print_exc(file=err)
        return None, "sim_fail", str(exc)

    report = sim_dir / cfg.arch_run_name / "COMPUTE_REPORT.csv"
    if not report.exists():
        return None, "sim_fail", f"SCALE-Sim did not emit {report}"
    return report, None, ""

def _build_ok_row(
    cfg: SweepConfig,
    n_tiles_expected: int,
    report_path: Path,
    *,
    logical_tiles: int | None = None,
    cache_key: str = "",
    cache_status: str = "",
    topology_mode: str = "raw",
    raw_topology_rows: int | None = None,
    group_counts: list[int] | None = None,
) -> dict:
    records = parse_compute_report(report_path)
    if len(records) != n_tiles_expected:
        raise ValueError(
            f"{cfg.workload}/{cfg.tile}/{cfg.arch}: SCALE-Sim emitted {len(records)} rows, "
            f"expected {n_tiles_expected}"
        )
    logical_tiles = n_tiles_expected if logical_tiles is None else logical_tiles
    raw_topology_rows = logical_tiles if raw_topology_rows is None else raw_topology_rows
    if group_counts is None:
        weights = [1] * len(records)
    else:
        if len(group_counts) != len(records):
            raise ValueError(
                f"{cfg.workload}/{cfg.tile}/{cfg.arch}: group_counts has {len(group_counts)} entries, "
                f"but SCALE-Sim emitted {len(records)} rows"
            )
        weights = [int(x) for x in group_counts]
    weight_sum = sum(weights) or len(records)

    compute_cycles = sum(r.total_cycles_compute * w for r, w in zip(records, weights))
    total_cycles = sum(r.total_cycles_incl_prefetch * w for r, w in zip(records, weights))
    stall_cycles = sum(r.stall_cycles * w for r, w in zip(records, weights))
    memory_overhead_cycles = max(0, total_cycles - compute_cycles)
    memory_overhead_ratio = (memory_overhead_cycles / total_cycles) if total_cycles else 0.0
    mean_overall = sum(r.overall_util_pct * w for r, w in zip(records, weights)) / weight_sum
    mean_mapping = sum(r.mapping_eff_pct * w for r, w in zip(records, weights)) / weight_sum
    mean_compute = sum(r.compute_util_pct * w for r, w in zip(records, weights)) / weight_sum
    from tools.reuse_model import DEFAULT_FOLD_FRACTION_WS, REUSE_MODEL_NAME

    reuse_sum = _weighted_reuse_aware_sum(
        records,
        weights,
        dataflow=cfg.dataflow,
        fold_fraction=DEFAULT_FOLD_FRACTION_WS,
    )

    row = _empty_row(cfg, "ok", compute_report=report_path)
    row.update(
        {
            "n_tiles": int(logical_tiles),
            "compute_cycles": int(round(compute_cycles)),
            "total_cycles": int(round(total_cycles)),
            "reuse_aware_cycles": float(reuse_sum),
            "reuse_aware_cycles_est": float(reuse_sum),
            "reuse_model": REUSE_MODEL_NAME + ("+grouped_full_weighted" if topology_mode == "grouped_full" else ""),
            "reuse_model_calibrated": False,
            "reuse_fold_fraction": DEFAULT_FOLD_FRACTION_WS,
            "mean_overall_util_pct": mean_overall,
            "mean_mapping_eff_pct": mean_mapping,
            "mean_compute_util_pct": mean_compute,
            "stall": int(round(stall_cycles)),
            "memory_overhead_cycles": int(round(memory_overhead_cycles)),
            "memory_overhead_ratio": float(memory_overhead_ratio),
            "logical_tiles": int(logical_tiles),
            "simulated_tiles": int(n_tiles_expected),
            "raw_topology_rows": int(raw_topology_rows),
            "topology_mode": topology_mode,
            "cache_key": cache_key,
            "cache_status": cache_status,
        }
    )
    return row



def _weighted_reuse_aware_sum(
    records: list[TileRecord],
    weights: list[int],
    *,
    dataflow: str,
    fold_fraction: float,
) -> float:
    """Return a reuse-aware estimate that respects grouped topology weights.

    `reuse_correction(records)` is order based: the first logical tile pays full
    prefetch and later logical tiles get the fold discount.  In grouped_full
    mode one SCALE-Sim row can represent many identical logical tiles, so simply
    multiplying the first corrected row by its group count would charge all
    logical tiles as if they were first.
    """
    if len(records) != len(weights):
        raise ValueError("records and weights must have the same length")
    if not 0.0 <= fold_fraction <= 1.0:
        raise ValueError(f"fold_fraction must be in [0, 1], got {fold_fraction}")

    if dataflow.lower() != "ws":
        return float(sum(r.total_cycles_incl_prefetch * max(0, int(w)) for r, w in zip(records, weights)))

    total = 0.0
    seen_logical_tile = False
    for r, raw_w in zip(records, weights):
        w = max(0, int(raw_w))
        if w == 0:
            continue
        prefetch = r.total_cycles_incl_prefetch - r.total_cycles_compute
        folded = float(r.total_cycles_compute) + (1.0 - fold_fraction) * prefetch
        if not seen_logical_tile:
            total += float(r.total_cycles_incl_prefetch)
            total += max(0, w - 1) * folded
            seen_logical_tile = True
        else:
            total += w * folded
    return total

def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "default"


def _hash_json(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _file_fingerprint(path: Path) -> str:
    """Return a stable fingerprint for a file without depending on its absolute path."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return f"missing:{path.name}"
    return hashlib.sha256(data).hexdigest()[:20]


def _topology_cache_key(cfg: SweepConfig, *, topology_mode: TopologyMode) -> str:
    # Topology generated from MNK/tile is independent of case name and arch cfg.
    return _hash_json(
        {
            "kind": "gemm_topology_v2",
            "mnk": [cfg.mnk.m, cfg.mnk.n, cfg.mnk.k],
            "tile": [cfg.tile_shape.m, cfg.tile_shape.n, cfg.tile_shape.k],
            "topology_mode": topology_mode,
        }
    )


def _scalesim_cache_key(cfg: SweepConfig, *, topology_mode: TopologyMode, skip_iree: bool) -> str:
    # Deliberately ignore case_name and generated config run_name so identical
    # experiments from different sweep groups share one SCALE-Sim execution.
    return _hash_json(
        {
            "kind": "scalesim_gemm_v2",
            "mnk": [cfg.mnk.m, cfg.mnk.n, cfg.mnk.k],
            "tile": [cfg.tile_shape.m, cfg.tile_shape.n, cfg.tile_shape.k],
            "topology_mode": topology_mode,
            "skip_iree": skip_iree,
            "arch": {
                "dataflow": cfg.dataflow,
                "array_h": cfg.arch_spec.array_h,
                "array_w": cfg.arch_spec.array_w,
                "bandwidth": cfg.arch_spec.bandwidth,
                "ifmap_sram_kb": cfg.arch_spec.ifmap_sram_kb,
                "filter_sram_kb": cfg.arch_spec.filter_sram_kb,
                "ofmap_sram_kb": cfg.arch_spec.ofmap_sram_kb,
                "ifmap_custom_layout": cfg.arch_spec.ifmap_custom_layout,
                "filter_custom_layout": cfg.arch_spec.filter_custom_layout,
            },
            "layout_fingerprint": _file_fingerprint(cfg.layout_path),
            "input_type": "gemm",
        }
    )


_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _cache_lock(key: str) -> threading.Lock:
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _copy_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def _find_cached_compute_report(sim_dir: Path, preferred_run_name: str) -> Path | None:
    preferred = sim_dir / preferred_run_name / "COMPUTE_REPORT.csv"
    if preferred.exists():
        return preferred
    matches = sorted(sim_dir.glob("*/COMPUTE_REPORT.csv"))
    return matches[0] if matches else None


def run_dir_for_config(cfg: SweepConfig, root: Path = SWEEP_DIR) -> Path:
    """Build a collision-resistant artifact directory for one sweep config.

    cfg.arch is intentionally not enough: two SCALE-Sim config files may share
    the same array/dataflow label while differing in SRAM/bandwidth/run_name.
    """
    return (
        root
        / _safe_path_component(cfg.workload)
        / _safe_path_component(cfg.case_name)
        / _safe_path_component(cfg.arch_run_name)
        / _safe_path_component(cfg.arch)
        / _safe_path_component(cfg.dataflow)
        / _safe_path_component(cfg.layout)
        / _safe_path_component(cfg.tile)
    )


def run_config(
    cfg: SweepConfig,
    root: Path = SWEEP_DIR,
    *,
    resume: bool = False,
    skip_iree: bool = False,
    topology_mode: TopologyMode = "raw",
    cache_root: Path | None = DEFAULT_CACHE_ROOT,
    verbose: bool = False,
    heartbeat_sec: int = 30,
) -> dict:
    if topology_mode not in ("raw", "grouped_full"):
        raise ValueError(f"unknown topology_mode: {topology_mode!r}")

    run_dir = run_dir_for_config(cfg, root=root)
    input_mlir = run_dir / "input_with_schedule.mlir"
    tiled = run_dir / "tiled.mlir"
    iree_stdout = run_dir / "iree_stdout.txt"
    iree_stderr = run_dir / "iree_stderr.txt"

    topology_key = _topology_cache_key(cfg, topology_mode=topology_mode)
    scalesim_key = _scalesim_cache_key(cfg, topology_mode=topology_mode, skip_iree=skip_iree)
    use_global_cache = cache_root is not None
    cache_base = cache_root if cache_root is not None else run_dir / ".local_cache"
    topology_cache_dir = cache_base / "topology" / topology_key
    sim_cache_dir = cache_base / "scalesim" / scalesim_key
    topology = topology_cache_dir / "topology.csv" if use_global_cache else run_dir / "topology.csv"
    sim_dir = sim_cache_dir / "sim" if use_global_cache else run_dir / "sim"
    sim_stdout = sim_cache_dir / "sim_stdout.txt" if use_global_cache else run_dir / "sim_stdout.txt"
    sim_stderr = sim_cache_dir / "sim_stderr.txt" if use_global_cache else run_dir / "sim_stderr.txt"
    metadata_path = sim_cache_dir / "metadata.json" if use_global_cache else run_dir / "metadata.json"

    common_paths = dict(input_mlir=input_mlir, tiled_mlir=tiled, topology=topology, sim_dir=sim_dir)
    expected = cfg.to_mnk_tile_spec().expected_tiles
    group_counts = _expected_group_counts(cfg, topology_mode)
    simulated_expected = _expected_simulated_tiles(cfg, topology_mode)

    def _paths_update(row: dict, report: Path | None = None) -> dict:
        row.update(
            {
                "input_mlir_path": _display_path(input_mlir),
                "tiled_mlir_path": _display_path(tiled),
                "topology_path": _display_path(topology),
                "sim_dir": _display_path(sim_dir),
                "compute_report_path": _display_path(report),
                "stdout_path": _display_path(sim_stdout),
                "stderr_path": _display_path(sim_stderr),
            }
        )
        return row

    existing_report = _find_cached_compute_report(sim_dir, cfg.arch_run_name)
    if resume and existing_report is not None:
        _log(f"CACHE-HIT scalesim key={scalesim_key} {_cfg_label(cfg)}", enabled=verbose)
        try:
            return _paths_update(
                _build_ok_row(
                    cfg,
                    simulated_expected,
                    existing_report,
                    logical_tiles=expected,
                    raw_topology_rows=expected,
                    group_counts=group_counts,
                    cache_key=scalesim_key,
                    cache_status="hit",
                    topology_mode=topology_mode,
                ),
                existing_report,
            )
        except Exception as exc:
            _log(f"CACHE-STALE key={scalesim_key} {_cfg_label(cfg)} :: {exc}", enabled=verbose)

    if skip_iree:
        _log(f"SKIP-IREE {_cfg_label(cfg)}", enabled=verbose)
        try:
            input_mlir.parent.mkdir(parents=True, exist_ok=True)
            if not input_mlir.exists():
                input_mlir.write_text(render_merged_mlir(cfg.to_mnk_tile_spec()), encoding="utf-8")
            tiled.parent.mkdir(parents=True, exist_ok=True)
            if not tiled.exists():
                tiled.write_text("// skipped iree-opt; topology was generated from MNK tile spec\n", encoding="utf-8")
        except ValueError as exc:
            return _empty_row(
                cfg,
                "invalid_config",
                **common_paths,
                stdout_path=iree_stdout,
                stderr_path=iree_stderr,
                error_stage="generate-mlir",
                error_message=str(exc),
            )
    else:
        _log(f"IREE  {_cfg_label(cfg)}", enabled=verbose)
        status, message = _generate_and_tile(cfg, input_mlir, tiled, iree_stdout, iree_stderr)
        if status:
            return _empty_row(
                cfg,
                status,
                **common_paths,
                stdout_path=iree_stdout,
                stderr_path=iree_stderr,
                error_stage="iree-opt",
                error_message=message,
            )

    lock_key = f"sim:{scalesim_key}"
    with _cache_lock(lock_key):
        existing_report = _find_cached_compute_report(sim_dir, cfg.arch_run_name)
        if existing_report is not None:
            _log(f"CACHE-HIT scalesim key={scalesim_key} {_cfg_label(cfg)}", enabled=verbose)
            try:
                return _paths_update(
                    _build_ok_row(
                        cfg,
                        simulated_expected,
                        existing_report,
                        logical_tiles=expected,
                        raw_topology_rows=expected,
                        group_counts=group_counts,
                        cache_key=scalesim_key,
                        cache_status="hit-after-wait",
                        topology_mode=topology_mode,
                    ),
                    existing_report,
                )
            except Exception as exc:
                _log(f"CACHE-STALE key={scalesim_key} {_cfg_label(cfg)} :: {exc}", enabled=verbose)

        if topology.exists():
            try:
                n_tiles = sum(1 for _ in topology.open("r", encoding="utf-8")) - 1
                logical_tiles = expected
                raw_topology_rows = expected
                if n_tiles <= 0:
                    raise ValueError("cached topology has no rows")
                _log(f"CACHE-HIT topology key={topology_key} rows={n_tiles} mode={topology_mode} {_cfg_label(cfg)}", enabled=verbose)
            except Exception:
                topology.unlink(missing_ok=True)
                n_tiles = logical_tiles = raw_topology_rows = 0
        else:
            n_tiles = logical_tiles = raw_topology_rows = 0

        if not topology.exists():
            _log(f"TOPO  key={topology_key} mode={topology_mode} {_cfg_label(cfg)}", enabled=verbose)
            n_tiles, logical_tiles, raw_topology_rows, emitted_group_counts, status, message = _emit_topology(cfg, topology, topology_mode=topology_mode)
            if status:
                return _empty_row(
                    cfg,
                    status,
                    **common_paths,
                    error_stage="topology",
                    error_message=message,
                )
            if emitted_group_counts is not None:
                group_counts = emitted_group_counts
            _write_metadata(
                topology_cache_dir / "metadata.json",
                {
                    "topology_key": topology_key,
                    "mnk": [cfg.mnk.m, cfg.mnk.n, cfg.mnk.k],
                    "tile": [cfg.tile_shape.m, cfg.tile_shape.n, cfg.tile_shape.k],
                    "topology_mode": topology_mode,
                    "simulated_tiles": n_tiles,
                    "logical_tiles": logical_tiles,
                    "raw_topology_rows": raw_topology_rows,
                    "group_counts": group_counts,
                },
            )

        if topology_mode == "grouped_full":
            _log(
                f"SIM   grouped_full groups={n_tiles} raw_rows={raw_topology_rows} logical_tiles={logical_tiles} "
                f"cache_key={scalesim_key} {_cfg_label(cfg)}",
                enabled=verbose,
            )
        else:
            _log(f"SIM   raw rows={n_tiles} cache_key={scalesim_key} {_cfg_label(cfg)}", enabled=verbose)

        report, status, message = _scalesim(
            cfg,
            topology,
            sim_dir,
            sim_stdout,
            sim_stderr,
            verbose=verbose,
            heartbeat_sec=heartbeat_sec,
        )
        if status or report is None:
            return _empty_row(
                cfg,
                status or "sim_fail",
                **common_paths,
                stdout_path=sim_stdout,
                stderr_path=sim_stderr,
                error_stage="scalesim",
                error_message=message,
            )
        _write_metadata(
            metadata_path,
            {
                "scalesim_key": scalesim_key,
                "topology_key": topology_key,
                "case_name_first_writer": cfg.case_name,
                "mnk": [cfg.mnk.m, cfg.mnk.n, cfg.mnk.k],
                "tile": [cfg.tile_shape.m, cfg.tile_shape.n, cfg.tile_shape.k],
                "arch": {
                    "dataflow": cfg.dataflow,
                    "array_h": cfg.arch_spec.array_h,
                    "array_w": cfg.arch_spec.array_w,
                    "bandwidth": cfg.arch_spec.bandwidth,
                    "ifmap_sram_kb": cfg.arch_spec.ifmap_sram_kb,
                    "filter_sram_kb": cfg.arch_spec.filter_sram_kb,
                    "ofmap_sram_kb": cfg.arch_spec.ofmap_sram_kb,
                },
                "topology_mode": topology_mode,
                "simulated_tiles": n_tiles,
                "logical_tiles": logical_tiles,
                "raw_topology_rows": raw_topology_rows,
                "group_counts": group_counts,
                "compute_report": _display_path(report),
            },
        )

    try:
        return _paths_update(
            _build_ok_row(
                cfg,
                n_tiles,
                report,
                logical_tiles=logical_tiles,
                raw_topology_rows=raw_topology_rows,
                group_counts=group_counts,
                cache_key=scalesim_key,
                cache_status="miss",
                topology_mode=topology_mode,
            ),
            report,
        )
    except ValueError as exc:
        return _empty_row(
            cfg,
            "csv_drift",
            **common_paths,
            compute_report=report,
            stdout_path=sim_stdout,
            stderr_path=sim_stderr,
            error_stage="aggregate",
            error_message=str(exc),
        )

def _run_config_worker(args: tuple[int, int, SweepConfig, Path, bool, bool, TopologyMode, Path | None, bool, int]) -> tuple[int, dict, float]:
    idx, total, cfg, root, resume, skip_iree, topology_mode, cache_root, verbose, heartbeat_sec = args
    start = time.perf_counter()
    _log(f"START {idx + 1:>3}/{total:<3} {_cfg_label(cfg)}", enabled=verbose)
    try:
        row = run_config(cfg, root=root, resume=resume, skip_iree=skip_iree, topology_mode=topology_mode, cache_root=cache_root, verbose=verbose, heartbeat_sec=heartbeat_sec)
        elapsed = time.perf_counter() - start
        _log(f"DONE  {idx + 1:>3}/{total:<3} status={row.get('status')} elapsed={elapsed:.1f}s {_cfg_label(cfg)}", enabled=verbose)
        return idx, row, elapsed
    except BaseException as exc:  # keep one bad case from killing the whole sweep
        row = _empty_row(
            cfg,
            "sim_fail",
            error_stage="worker",
            error_message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        elapsed = time.perf_counter() - start
        _log(f"FAIL  {idx + 1:>3}/{total:<3} elapsed={elapsed:.1f}s {_cfg_label(cfg)} :: {type(exc).__name__}: {exc}", enabled=verbose)
        return idx, row, elapsed


def _raise_if_failed(cfg: SweepConfig, row: dict) -> None:
    if row.get("status") != "ok":
        raise RuntimeError(
            f"sweep failed at {cfg.case_name}/{cfg.tile}/{cfg.layout}: "
            f"{row.get('error_stage')}: {row.get('error_message')}"
        )


def _run_parallel_configs(
    *,
    configs: tuple[SweepConfig, ...],
    output_root: Path,
    jobs: int,
    resume: bool,
    skip_iree: bool,
    topology_mode: TopologyMode,
    cache_root: Path | None,
    fail_fast: bool,
    parallel_backend: ParallelBackend,
    verbose: bool,
    heartbeat_sec: int,
) -> list[dict]:
    rows: list[dict | None] = [None] * len(configs)
    total = len(configs)
    worker_args = [(idx, total, cfg, output_root, resume, skip_iree, topology_mode, cache_root, verbose, heartbeat_sec) for idx, cfg in enumerate(configs)]
    _log(f"parallel sweep start: cases={total}, jobs={jobs}, backend={parallel_backend}, resume={resume}, skip_iree={skip_iree}, topology_mode={topology_mode}, cache_root={_display_path(cache_root)}", enabled=verbose)
    executor_cls = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor

    try:
        with executor_cls(max_workers=jobs) as executor:
            futures = {executor.submit(_run_config_worker, arg): arg[0] for arg in worker_args}
            done = 0
            ok = 0
            failed = 0
            start_all = time.perf_counter()
            for future in as_completed(futures):
                idx, row, elapsed = future.result()
                rows[idx] = row
                done += 1
                if row.get("status") == "ok":
                    ok += 1
                else:
                    failed += 1
                total_elapsed = time.perf_counter() - start_all
                rate = done / total_elapsed if total_elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                _log(
                    f"PROGRESS {done}/{total} ok={ok} failed={failed} "
                    f"last={elapsed:.1f}s elapsed={total_elapsed:.1f}s eta={eta:.1f}s",
                    enabled=verbose,
                )
                if fail_fast:
                    _raise_if_failed(configs[idx], row)
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "parallel process pool crashed. This usually means too many SCALE-Sim "
            "workers were launched for the available memory, or a native dependency "
            "terminated a child process. Retry with TPUV2_BACKEND=thread, or reduce "
            "TPUV2_JOBS to 4 or 8. Existing completed cases can be reused with "
            "TPUV2_RESUME=1."
        ) from exc

    missing = [i for i, row in enumerate(rows) if row is None]
    if missing:
        raise RuntimeError(f"parallel sweep ended without rows for indices: {missing[:10]}")
    return [row for row in rows if row is not None]


def run_sweep(
    configs: tuple[SweepConfig, ...] = DEFAULT_CONFIGS,
    out_path: Path = RESULTS_PARQUET,
    output_root: Path = SWEEP_DIR,
    clean: bool = True,
    csv_out: Path | None = None,
    fail_fast: bool = False,
    require_any_ok: bool = False,
    jobs: int = 1,
    resume: bool = False,
    skip_iree: bool = False,
    topology_mode: TopologyMode = "raw",
    cache_root: Path | None = DEFAULT_CACHE_ROOT,
    parallel_backend: ParallelBackend = "thread",
    verbose: bool = True,
    heartbeat_sec: int = 30,
) -> pd.DataFrame:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(configs)
    _log(f"sweep start: cases={total}, jobs={jobs}, backend={parallel_backend}, resume={resume}, skip_iree={skip_iree}, topology_mode={topology_mode}, cache_root={_display_path(cache_root)}, clean={clean}", enabled=verbose)
    if jobs <= 1:
        rows = []
        ok = 0
        failed = 0
        start_all = time.perf_counter()
        for idx_cfg, cfg in enumerate(configs):
            idx, row, elapsed = _run_config_worker((idx_cfg, total, cfg, output_root, resume, skip_iree, topology_mode, cache_root, verbose, heartbeat_sec))
            rows.append(row)
            if row.get("status") == "ok":
                ok += 1
            else:
                failed += 1
            done = len(rows)
            total_elapsed = time.perf_counter() - start_all
            rate = done / total_elapsed if total_elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            _log(f"PROGRESS {done}/{total} ok={ok} failed={failed} last={elapsed:.1f}s elapsed={total_elapsed:.1f}s eta={eta:.1f}s", enabled=verbose)
            if fail_fast:
                _raise_if_failed(cfg, row)
    else:
        rows = _run_parallel_configs(
            configs=configs,
            output_root=output_root,
            jobs=jobs,
            resume=resume,
            skip_iree=skip_iree,
            topology_mode=topology_mode,
            cache_root=cache_root,
            fail_fast=fail_fast,
            parallel_backend=parallel_backend,
            verbose=verbose,
            heartbeat_sec=heartbeat_sec,
        )

    df = pd.DataFrame(rows, columns=list(PARQUET_COLUMNS))
    if require_any_ok and not df.empty and not (df["status"] == "ok").any():
        status_counts = df["status"].value_counts(dropna=False).to_dict()
        raise RuntimeError(f"all sweep cases failed; no valid experiment result was produced: {status_counts}")
    write_dataframe_outputs(df, out_path, csv_out)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a general MatMul MNK/tile/HW sweep")
    ap.add_argument("--mnk", type=parse_mnk, action="append", default=None, help="MNK shape, e.g. 32x32x64")
    ap.add_argument("--tiles", type=parse_tile, nargs="+", default=None, help="tile shapes, e.g. 8x8x8 16x16x16")
    ap.add_argument("--arch-cfg", type=Path, nargs="+", default=None, help="SCALE-Sim config paths")
    ap.add_argument("--layout", type=Path, action="append", default=None, help="SCALE-Sim layout CSV path. May be passed multiple times.")
    ap.add_argument("--case-name", default="default", help="label to store in the result row")
    ap.add_argument("--output", type=Path, default=RESULTS_PARQUET)
    ap.add_argument("--csv-output", type=Path, help="optional CSV summary output")
    ap.add_argument("--output-root", type=Path, default=SWEEP_DIR)
    ap.add_argument("--no-clean", action="store_true", help="keep existing output root")
    ap.add_argument("--fail-fast", action="store_true", help="stop on the first non-ok sweep row")
    ap.add_argument("--require-any-ok", action="store_true", help="fail if every sweep row ends with a non-ok status")
    ap.add_argument("--jobs", type=int, default=1, help="parallel SCALE-Sim workers; use 1 for sequential execution")
    ap.add_argument("--parallel-backend", choices=("thread", "process"), default="thread", help="thread is safer inside Docker; process may be faster but uses more memory")
    ap.add_argument("--resume", action="store_true", help="reuse existing COMPUTE_REPORT.csv files when present")
    ap.add_argument("--skip-iree", action="store_true", help="skip iree-opt and generate topology directly from the MNK tile spec")
    ap.add_argument("--topology-mode", choices=("raw", "grouped_full"), default="raw", help="raw sends every full-topology row to SCALE-Sim; grouped_full groups identical full-topology rows and weights aggregate metrics")
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT, help="global topology/SCALE-Sim cache directory; use --no-cache to disable")
    ap.add_argument("--no-cache", action="store_true", help="disable global cache and use per-case output directories only")
    ap.add_argument("--quiet", action="store_true", help="suppress per-case progress logs")
    ap.add_argument("--heartbeat-sec", type=int, default=30, help="print SIM-RUNNING heartbeat every N seconds while a SCALE-Sim case is running; use 0 to disable")
    args = ap.parse_args()

    mnks = tuple(args.mnk) if args.mnk else (MnkShape(32, 32, 64),)
    tiles = tuple(args.tiles) if args.tiles else (TileShape(8, 8, 8), TileShape(16, 16, 16))
    arch_cfgs = tuple(args.arch_cfg) if args.arch_cfg else (DEFAULT_ARCH_CFG,)
    layouts = tuple(args.layout) if args.layout else None
    configs = build_configs(mnks=mnks, tiles=tiles, arch_cfgs=arch_cfgs, layouts=layouts, case_name=args.case_name)

    df = run_sweep(
        configs=configs,
        out_path=args.output,
        output_root=args.output_root,
        clean=not args.no_clean,
        csv_out=args.csv_output,
        fail_fast=args.fail_fast,
        require_any_ok=args.require_any_ok,
        jobs=max(1, args.jobs),
        resume=args.resume,
        skip_iree=args.skip_iree,
        topology_mode=args.topology_mode,
        cache_root=None if args.no_cache else args.cache_root,
        parallel_backend=args.parallel_backend,
        verbose=not args.quiet,
        heartbeat_sec=max(0, args.heartbeat_sec),
    )
    print(f"\nwrote {args.output} ({len(df)} rows x {len(df.columns)} cols)")
    if args.csv_output:
        print(f"wrote {args.csv_output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
