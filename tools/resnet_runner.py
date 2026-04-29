"""Run a SCALE-Sim ResNet convolution topology and summarize layer results.

This is a baseline path for real-model experiments before full IREE import and
Transform Dialect tiling are available for ResNet.  It runs the vendored
SCALE-Sim topology directly, then annotates each layer with the estimated
internal fold count used by SCALE-Sim's array mapping model.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scalesim.scale_config import scale_config
from scalesim.scale_sim import scalesim

from tools.aggregator import parse_compute_report

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TOPOLOGY = REPO / "SCALE-Sim" / "topologies" / "conv_nets" / "Resnet18.csv"
DEFAULT_LAYOUT = REPO / "SCALE-Sim" / "layouts" / "conv_nets" / "resnet18.csv"
DEFAULT_ARCH_CFG = REPO / "SCALE-Sim" / "configs" / "tpuv2.cfg"
OUTPUT_ROOT = REPO / "outputs" / "resnet18_tpuv2_baseline"
RESULTS_PARQUET = REPO / "results" / "resnet18_tpuv2_baseline.parquet"
DEFAULT_SMOKE_LAYERS = ("Conv3_1a", "Conv4_1a", "Conv5_1a", "FC")

PARQUET_COLUMNS = (
    "model",
    "layer_id",
    "layer_name",
    "arch",
    "dataflow",
    "ifmap_h",
    "ifmap_w",
    "filter_h",
    "filter_w",
    "channels",
    "num_filters",
    "stride",
    "gemm_m",
    "gemm_n",
    "gemm_k",
    "s_r",
    "s_c",
    "t",
    "row_fold",
    "col_fold",
    "internal_folds",
    "compute_cycles",
    "total_cycles",
    "stall_cycles",
    "mapping_eff_pct",
    "compute_util_pct",
    "status",
)


@dataclass(frozen=True)
class ConvLayer:
    layer_name: str
    ifmap_h: int
    ifmap_w: int
    filter_h: int
    filter_w: int
    channels: int
    num_filters: int
    stride: int


def _strip_cells(row: list[str]) -> list[str]:
    return [c.strip() for c in row if c.strip() != ""]


def parse_conv_topology(path: Path) -> list[ConvLayer]:
    layers: list[ConvLayer] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            cells = _strip_cells(row)
            if not cells:
                continue
            if len(cells) < 8:
                raise ValueError(f"conv topology row too short: {cells!r}")
            layers.append(
                ConvLayer(
                    layer_name=cells[0],
                    ifmap_h=int(cells[1]),
                    ifmap_w=int(cells[2]),
                    filter_h=int(cells[3]),
                    filter_w=int(cells[4]),
                    channels=int(cells[5]),
                    num_filters=int(cells[6]),
                    stride=int(cells[7]),
                )
            )
    return layers


def _write_subset_csv(src: Path, dst: Path, layer_names: tuple[str, ...]) -> None:
    wanted = set(layer_names)
    with src.open(newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = [row for row in rows[1:] if _strip_cells(row) and _strip_cells(row)[0] in wanted]
    found = {_strip_cells(row)[0] for row in body}
    missing = [name for name in layer_names if name not in found]
    if missing:
        raise ValueError(f"layers not found in {src}: {missing}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


def _ofmap_dim(ifmap: int, filt: int, stride: int) -> int:
    return int(math.ceil((ifmap - filt + stride) / stride))


def equivalent_gemm(layer: ConvLayer) -> tuple[int, int, int]:
    ofmap_h = _ofmap_dim(layer.ifmap_h, layer.filter_h, layer.stride)
    ofmap_w = _ofmap_dim(layer.ifmap_w, layer.filter_w, layer.stride)
    m = ofmap_h * ofmap_w
    n = layer.num_filters
    k = layer.filter_h * layer.filter_w * layer.channels
    return m, n, k


def spatiotemporal_dims(
    layer: ConvLayer, dataflow: str
) -> tuple[int, int, int]:
    m, n, k = equivalent_gemm(layer)
    df = dataflow.lower()
    if df == "ws":
        return k, n, m
    if df == "os":
        return m, n, k
    if df == "is":
        return k, m, n
    raise ValueError(f"unsupported dataflow: {dataflow!r}")


def internal_folds(
    layer: ConvLayer, dataflow: str, array_h: int, array_w: int
) -> tuple[int, int, int, int, int, int]:
    s_r, s_c, t = spatiotemporal_dims(layer, dataflow)
    row_fold = math.ceil(s_r / array_h)
    col_fold = math.ceil(s_c / array_w)
    return s_r, s_c, t, row_fold, col_fold, row_fold * col_fold


def _load_arch(config_path: Path) -> tuple[str, int, int, str]:
    cfg = scale_config()
    cfg.read_conf_file(str(config_path))
    arr_h, arr_w = cfg.get_array_dims()
    dataflow = cfg.get_dataflow()
    return f"{arr_h}x{arr_w}_{dataflow}", arr_h, arr_w, dataflow


def _run_scalesim(
    topology: Path,
    layout: Path,
    arch_cfg: Path,
    output_root: Path,
    clean: bool,
    verbose: bool,
) -> Path:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sim = scalesim(
        save_disk_space=True,
        verbose=verbose,
        config=str(arch_cfg),
        topology=str(topology),
        layout=str(layout),
        input_type_gemm=False,
    )
    sim.run_scale(top_path=str(output_root))

    cfg = scale_config()
    cfg.read_conf_file(str(arch_cfg))
    report = output_root / cfg.get_run_name() / "COMPUTE_REPORT.csv"
    if not report.exists():
        raise RuntimeError(f"SCALE-Sim did not emit {report}")
    return report


def build_layer_dataframe(
    model: str,
    topology: Path,
    arch_cfg: Path,
    compute_report: Path,
) -> pd.DataFrame:
    arch, array_h, array_w, dataflow = _load_arch(arch_cfg)
    layers = parse_conv_topology(topology)
    records = parse_compute_report(compute_report)
    if len(layers) != len(records):
        raise ValueError(
            f"topology/report layer mismatch: {len(layers)} layers vs {len(records)} rows"
        )

    rows = []
    for layer, rec in zip(layers, records):
        gemm_m, gemm_n, gemm_k = equivalent_gemm(layer)
        s_r, s_c, t, row_fold, col_fold, folds = internal_folds(
            layer, dataflow, array_h, array_w
        )
        rows.append(
            {
                "model": model,
                "layer_id": rec.layer_id,
                "layer_name": layer.layer_name,
                "arch": arch,
                "dataflow": dataflow,
                "ifmap_h": layer.ifmap_h,
                "ifmap_w": layer.ifmap_w,
                "filter_h": layer.filter_h,
                "filter_w": layer.filter_w,
                "channels": layer.channels,
                "num_filters": layer.num_filters,
                "stride": layer.stride,
                "gemm_m": gemm_m,
                "gemm_n": gemm_n,
                "gemm_k": gemm_k,
                "s_r": s_r,
                "s_c": s_c,
                "t": t,
                "row_fold": row_fold,
                "col_fold": col_fold,
                "internal_folds": folds,
                "compute_cycles": rec.total_cycles_compute,
                "total_cycles": rec.total_cycles_incl_prefetch,
                "stall_cycles": rec.stall_cycles,
                "mapping_eff_pct": rec.mapping_eff_pct,
                "compute_util_pct": rec.compute_util_pct,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows, columns=list(PARQUET_COLUMNS))


def run_resnet_baseline(
    topology: Path = DEFAULT_TOPOLOGY,
    layout: Path = DEFAULT_LAYOUT,
    arch_cfg: Path = DEFAULT_ARCH_CFG,
    output_root: Path = OUTPUT_ROOT,
    results_path: Path = RESULTS_PARQUET,
    clean: bool = True,
    layers: tuple[str, ...] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    sim_topology = topology
    sim_layout = layout
    if layers:
        subset_dir = output_root / "_subset_inputs"
        sim_topology = subset_dir / topology.name
        sim_layout = subset_dir / layout.name
        _write_subset_csv(topology, sim_topology, layers)
        _write_subset_csv(layout, sim_layout, layers)

    report = _run_scalesim(
        sim_topology, sim_layout, arch_cfg, output_root, clean=False, verbose=verbose
    )
    df = build_layer_dataframe(
        model=topology.stem,
        topology=sim_topology,
        arch_cfg=arch_cfg,
        compute_report=report,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ResNet topology baseline on SCALE-Sim")
    ap.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--arch-cfg", type=Path, default=DEFAULT_ARCH_CFG)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS_PARQUET)
    ap.add_argument(
        "--layers",
        nargs="+",
        help="optional layer-name subset, e.g. Conv3_1a Conv4_1a FC",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=f"run representative subset: {' '.join(DEFAULT_SMOKE_LAYERS)}",
    )
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="show SCALE-Sim per-layer progress")
    args = ap.parse_args()
    layers = tuple(args.layers) if args.layers else None
    if args.smoke:
        layers = DEFAULT_SMOKE_LAYERS

    df = run_resnet_baseline(
        topology=args.topology,
        layout=args.layout,
        arch_cfg=args.arch_cfg,
        output_root=args.output_root,
        results_path=args.results,
        clean=not args.no_clean,
        layers=layers,
        verbose=args.verbose,
    )
    print(f"wrote {args.results} ({len(df)} layers)")
    print(
        df[
            [
                "layer_id",
                "layer_name",
                "gemm_m",
                "gemm_n",
                "gemm_k",
                "internal_folds",
                "compute_cycles",
                "total_cycles",
                "mapping_eff_pct",
            ]
        ].to_string(index=False)
    )
    print(f"model compute_cycles={int(df['compute_cycles'].sum())}")
    print(f"model total_cycles={int(df['total_cycles'].sum())}")


if __name__ == "__main__":
    main()
