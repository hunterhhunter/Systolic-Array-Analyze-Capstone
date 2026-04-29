"""IREE tile SCALE-Sim ResNet/Conv topology rows and emit Conv topology tiles.

This is the first Conv/ResNet path:

    SCALE-Sim Conv topology rows
      -> generated linalg.conv_2d_nhwc_hwcf MLIR + Transform schedule
      -> iree-opt tiled MLIR
      -> per-IREE-tile SCALE-Sim Conv topology CSV
      -> optional SCALE-Sim run + aggregate summary

Edge tiles are emitted as smaller SCALE-Sim topology rows. This models the
common systolic-array behavior where the remaining PE lanes are left idle for
the final partial tile.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scalesim.scale_config import scale_config
from scalesim.scale_sim import scalesim

from tools.aggregator import summarize_compute_report
from tools.resnet_runner import ConvLayer, parse_conv_topology

REPO = Path(__file__).resolve().parent.parent


def default_iree_opt() -> Path:
    if os.environ.get("IREE_OPT"):
        return Path(os.environ["IREE_OPT"]).expanduser()
    repo_venv_tool = REPO / ".venv" / "bin" / "iree-opt"
    if repo_venv_tool.exists():
        return repo_venv_tool
    path_tool = shutil.which("iree-opt")
    return Path(path_tool) if path_tool else repo_venv_tool


IREE_OPT = default_iree_opt()
DEFAULT_TOPOLOGY = REPO / "SCALE-Sim" / "topologies" / "conv_nets" / "Resnet18.csv"
DEFAULT_ARCH_CFG = REPO / "SCALE-Sim" / "configs" / "tpuv2.cfg"
DEFAULT_LAYOUT = REPO / "SCALE-Sim" / "layouts" / "conv_nets" / "test.csv"
DEFAULT_OUTPUT_ROOT = REPO / "outputs" / "resnet18_iree_tiles"
DEFAULT_RESULTS = REPO / "results" / "resnet18_iree_tiles.parquet"
DEFAULT_SMOKE_LAYERS = ("Conv3_1a", "Conv4_1a", "Conv5_1a")


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def _display_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ConvTile:
    oh: int
    ow: int
    oc: int

    @property
    def label(self) -> str:
        return f"oh{self.oh}_ow{self.ow}_oc{self.oc}"


def _ofmap_dim(ifmap: int, filt: int, stride: int) -> int:
    return int(math.ceil((ifmap - filt + stride) / stride))


def ofmap_hw(layer: ConvLayer) -> tuple[int, int]:
    return (
        _ofmap_dim(layer.ifmap_h, layer.filter_h, layer.stride),
        _ofmap_dim(layer.ifmap_w, layer.filter_w, layer.stride),
    )


def effective_mlir_ifmap_hw(layer: ConvLayer) -> tuple[int, int]:
    """Input shape required by padding-free linalg.conv for SCALE-Sim's output size.

    Some ResNet topology rows imply padded stride-2 convolutions. SCALE-Sim stores
    only the logical IFMAP size and computes output with a ceil-style formula.
    linalg.conv_2d_nhwc_hwcf has no padding attribute, so the generated MLIR uses
    the effective padded size needed to produce the same OH/OW.
    """
    oh, ow = ofmap_hw(layer)
    return (
        max(layer.ifmap_h, (oh - 1) * layer.stride + layer.filter_h),
        max(layer.ifmap_w, (ow - 1) * layer.stride + layer.filter_w),
    )


def _sanitize_symbol(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not out or out[0].isdigit():
        out = f"conv_{out}"
    return out


def _select_layers(
    topology: Path,
    layer_names: tuple[str, ...] | None,
    skip_non_divisible: bool,
    tile: ConvTile,
) -> list[ConvLayer]:
    if skip_non_divisible:
        print("--skip-non-divisible is ignored: edge tiles are supported")
    layers = parse_conv_topology(topology)
    if layer_names:
        wanted = set(layer_names)
        layers = [layer for layer in layers if layer.layer_name in wanted]
        found = {layer.layer_name for layer in layers}
        missing = [name for name in layer_names if name not in found]
        if missing:
            raise ValueError(f"layers not found in {topology}: {missing}")

    if not layers:
        raise ValueError("no layers selected")
    return layers


def render_conv_module(layers: list[ConvLayer], tile: ConvTile) -> str:
    funcs = []
    for layer in layers:
        oh, ow = ofmap_hw(layer)
        mlir_ifmap_h, mlir_ifmap_w = effective_mlir_ifmap_hw(layer)
        fn = _sanitize_symbol(layer.layer_name)
        funcs.append(
            f"""  func.func @{fn}(
      %input: tensor<1x{mlir_ifmap_h}x{mlir_ifmap_w}x{layer.channels}xf32>,
      %filter: tensor<{layer.filter_h}x{layer.filter_w}x{layer.channels}x{layer.num_filters}xf32>) -> tensor<1x{oh}x{ow}x{layer.num_filters}xf32> {{
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x{oh}x{ow}x{layer.num_filters}xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x{oh}x{ow}x{layer.num_filters}xf32>) -> tensor<1x{oh}x{ow}x{layer.num_filters}xf32>
    %result = linalg.conv_2d_nhwc_hwcf
        {{dilations = dense<1> : vector<2xi64>, strides = dense<{layer.stride}> : vector<2xi64>}}
        ins(%input, %filter : tensor<1x{mlir_ifmap_h}x{mlir_ifmap_w}x{layer.channels}xf32>, tensor<{layer.filter_h}x{layer.filter_w}x{layer.channels}x{layer.num_filters}xf32>)
        outs(%filled : tensor<1x{oh}x{ow}x{layer.num_filters}xf32>) -> tensor<1x{oh}x{ow}x{layer.num_filters}xf32>
    return %result : tensor<1x{oh}x{ow}x{layer.num_filters}xf32>
  }}"""
        )

    return f"""// Auto-generated ResNet/Conv Transform Dialect tiling module.
// Tile output loops: OH={tile.oh}, OW={tile.ow}, OC={tile.oc}

module attributes {{ transform.with_named_sequence }} {{

{chr(10).join(funcs)}

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %conv = transform.structured.match ops{{["linalg.conv_2d_nhwc_hwcf"]}} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %conv tile_sizes [0, {tile.oh}, {tile.ow}, {tile.oc}, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }}
}}
"""


def run_iree_transform(input_mlir: Path, tiled_mlir: Path) -> None:
    if not IREE_OPT.exists():
        raise FileNotFoundError(
            f"IREE_OPT={IREE_OPT} not found. Run `make env` or set IREE_OPT=/path/to/iree-opt."
        )
    tiled_mlir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(IREE_OPT),
            str(input_mlir),
            "--pass-pipeline=builtin.module(transform-interpreter)",
            "-o",
            str(tiled_mlir),
        ],
        check=True,
    )


def emit_conv_topology(layers: list[ConvLayer], tile: ConvTile) -> str:
    lines = [
        "Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,"
    ]
    for layer in layers:
        oh, ow = ofmap_hw(layer)
        for oh0 in range(0, oh, tile.oh):
            actual_oh = min(tile.oh, oh - oh0)
            ifmap_tile_h = (actual_oh - 1) * layer.stride + layer.filter_h
            for ow0 in range(0, ow, tile.ow):
                actual_ow = min(tile.ow, ow - ow0)
                ifmap_tile_w = (actual_ow - 1) * layer.stride + layer.filter_w
                for oc0 in range(0, layer.num_filters, tile.oc):
                    actual_oc = min(tile.oc, layer.num_filters - oc0)
                    name = f"{layer.layer_name}_oh{oh0:03d}_ow{ow0:03d}_oc{oc0:04d}"
                    lines.append(
                        f"{name}, {ifmap_tile_h}, {ifmap_tile_w}, {layer.filter_h}, "
                        f"{layer.filter_w}, {layer.channels}, {actual_oc}, {layer.stride},"
                    )
    return "\n".join(lines) + "\n"


def write_conv_topology(layers: list[ConvLayer], tile: ConvTile, output: Path) -> int:
    text = emit_conv_topology(layers, tile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return max(0, len(text.splitlines()) - 1)


def _run_scalesim(topology: Path, arch_cfg: Path, output_root: Path, verbose: bool) -> Path:
    sim = scalesim(
        save_disk_space=True,
        verbose=verbose,
        config=str(arch_cfg),
        topology=str(topology),
        layout=str(DEFAULT_LAYOUT),
        input_type_gemm=False,
    )
    sim.run_scale(top_path=str(output_root))
    cfg = scale_config()
    cfg.read_conf_file(str(arch_cfg))
    report = output_root / cfg.get_run_name() / "COMPUTE_REPORT.csv"
    if not report.exists():
        raise RuntimeError(f"SCALE-Sim did not emit {report}")
    return report


def run_resnet_iree_tiles(
    topology: Path = DEFAULT_TOPOLOGY,
    arch_cfg: Path = DEFAULT_ARCH_CFG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    results_path: Path = DEFAULT_RESULTS,
    tile: ConvTile = ConvTile(7, 7, 128),
    layer_names: tuple[str, ...] | None = DEFAULT_SMOKE_LAYERS,
    skip_non_divisible: bool = False,
    run_scalesim: bool = False,
    clean: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    topology = _repo_path(topology)
    arch_cfg = _repo_path(arch_cfg)
    output_root = _repo_path(output_root)
    results_path = _repo_path(results_path)

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layers = _select_layers(topology, layer_names, skip_non_divisible, tile)
    input_mlir = output_root / "input_with_schedule.mlir"
    tiled_mlir = output_root / "tiled.mlir"
    topology_csv = output_root / "topology.csv"

    input_mlir.write_text(render_conv_module(layers, tile))
    run_iree_transform(input_mlir, tiled_mlir)
    n_tiles = write_conv_topology(layers, tile, topology_csv)

    rows = [
        {
            "model": topology.stem,
            "tile_oh": tile.oh,
            "tile_ow": tile.ow,
            "tile_oc": tile.oc,
            "n_source_layers": len(layers),
            "n_tiles": n_tiles,
            "input_mlir": _display_path(input_mlir),
            "tiled_mlir": _display_path(tiled_mlir),
            "topology_csv": _display_path(topology_csv),
            "status": "tiled",
        }
    ]

    if run_scalesim:
        report = _run_scalesim(topology_csv, arch_cfg, output_root / "sim", verbose=verbose)
        summary = summarize_compute_report(
            report,
            dataflow=_load_dataflow(arch_cfg),
            workload=f"{topology.stem}_iree_tiles",
            arch=arch_cfg.stem,
        )
        rows[0].update(summary)
        rows[0]["status"] = "ok"

    df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_path, index=False)
    csv_path = results_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return df


def _load_dataflow(arch_cfg: Path) -> str:
    cfg = scale_config()
    cfg.read_conf_file(str(arch_cfg))
    return cfg.get_dataflow()


def main() -> None:
    ap = argparse.ArgumentParser(description="IREE-tile ResNet/Conv topology rows")
    ap.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    ap.add_argument("--arch-cfg", type=Path, default=DEFAULT_ARCH_CFG)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--tile-oh", type=int, default=7)
    ap.add_argument("--tile-ow", type=int, default=7)
    ap.add_argument("--tile-oc", type=int, default=128)
    ap.add_argument("--layers", nargs="+")
    ap.add_argument("--smoke", action="store_true", help=f"use {' '.join(DEFAULT_SMOKE_LAYERS)}")
    ap.add_argument("--full", action="store_true", help="select all topology rows")
    ap.add_argument("--skip-non-divisible", action="store_true")
    ap.add_argument("--run-scalesim", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    layer_names: tuple[str, ...] | None
    if args.full:
        layer_names = None
    elif args.layers:
        layer_names = tuple(args.layers)
    else:
        layer_names = DEFAULT_SMOKE_LAYERS if args.smoke or not args.layers else None

    df = run_resnet_iree_tiles(
        topology=args.topology,
        arch_cfg=args.arch_cfg,
        output_root=args.output_root,
        results_path=args.results,
        tile=ConvTile(args.tile_oh, args.tile_ow, args.tile_oc),
        layer_names=layer_names,
        skip_non_divisible=args.skip_non_divisible,
        run_scalesim=args.run_scalesim,
        clean=not args.no_clean,
        verbose=args.verbose,
    )
    print(f"wrote {args.results}")
    print(f"wrote {args.results.with_suffix('.csv')}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
