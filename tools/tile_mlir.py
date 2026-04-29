"""Public CLI: linalg MLIR -> IREE-tiled MLIR + tile manifest."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from tools.iree_tile_input import TileConfig, tile_input
from tools.linalg_mlir import (
    Kind,
    display_path,
    ops_to_json,
    parse_linalg_ops,
    parse_torch_onnx_conv_ops,
    render_conv2d_linalg_module,
    repo_path,
    write_manifest,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO / "outputs" / "experiments"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text).strip("._-") or "model"


def _tile_label(kind: Kind, args: argparse.Namespace) -> str:
    if kind == "matmul":
        return f"{args.tile_m}x{args.tile_n}x{args.tile_k}"
    return f"{args.tile_oh}x{args.tile_ow}x{args.tile_oc}"


def default_run_dir(input_path: Path, kind: Kind, args: argparse.Namespace) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{_safe_name(input_path.stem)}_{kind}_tile_{_tile_label(kind, args)}"


def build_tile_config(kind: Kind, args: argparse.Namespace) -> TileConfig:
    if kind == "matmul":
        tile = (args.tile_m, args.tile_n, args.tile_k)
        if any(x <= 0 for x in tile):
            raise ValueError("matmul tile dimensions must be positive")
        return TileConfig(ops=("linalg.matmul",), tile_sizes=tile, num_loops=3)

    tile = (args.tile_oh, args.tile_ow, args.tile_oc)
    if any(x <= 0 for x in tile):
        raise ValueError("conv2d tile dimensions must be positive")
    return TileConfig(
        ops=("linalg.conv_2d_nhwc_hwcf",),
        tile_sizes=(0, args.tile_oh, args.tile_ow, args.tile_oc, 0, 0, 0),
        num_loops=3,
    )


def manifest_tile(kind: Kind, args: argparse.Namespace) -> dict[str, int]:
    if kind == "matmul":
        return {"m": args.tile_m, "n": args.tile_n, "k": args.tile_k}
    return {"oh": args.tile_oh, "ow": args.tile_ow, "oc": args.tile_oc}


def build_manifest(
    *,
    kind: Kind,
    source_mlir: Path,
    input_with_schedule: Path,
    tiled_mlir: Path | None,
    manifest_path: Path,
    tile_config: TileConfig,
    tile: dict[str, int],
    ops: list[Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "source_mlir": display_path(source_mlir),
        "input_with_schedule": display_path(input_with_schedule),
        "tiled_mlir": display_path(tiled_mlir) if tiled_mlir else None,
        "manifest": display_path(manifest_path),
        "tile": tile,
        "tile_sizes": list(tile_config.tile_sizes),
        "ops": ops_to_json(ops),
    }


def run_tile_mlir(
    input_path: Path,
    kind: Kind,
    run_dir: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_mlir = repo_path(input_path)
    run_dir = repo_path(run_dir) if run_dir else default_run_dir(source_mlir, kind, args)
    manifest_path = run_dir / "tile_manifest.json"
    text = source_mlir.read_text()
    bridge_mlir: Path | None = None
    try:
        ops = parse_linalg_ops(text, kind)
        tile_source_mlir = source_mlir
    except ValueError:
        if kind != "conv2d":
            raise
        ops = parse_torch_onnx_conv_ops(text)
        if not ops:
            raise
        run_dir.mkdir(parents=True, exist_ok=True)
        bridge_mlir = run_dir / "linalg_bridge.mlir"
        bridge_mlir.write_text(render_conv2d_linalg_module(ops))
        tile_source_mlir = bridge_mlir

    tile_config = build_tile_config(kind, args)
    input_with_schedule, tiled_mlir = tile_input(
        tile_source_mlir,
        tile_config,
        out_dir=run_dir,
        input_with_schedule=run_dir / "input_with_schedule.mlir",
        tiled_mlir=run_dir / "tiled.mlir",
        no_run=args.no_run,
    )
    manifest = build_manifest(
        kind=kind,
        source_mlir=tile_source_mlir,
        input_with_schedule=input_with_schedule,
        tiled_mlir=tiled_mlir,
        manifest_path=manifest_path,
        tile_config=tile_config,
        tile=manifest_tile(kind, args),
        ops=ops,
    )
    if bridge_mlir:
        manifest["original_source_mlir"] = display_path(source_mlir)
        manifest["linalg_bridge_mlir"] = display_path(bridge_mlir)
    write_manifest(manifest, manifest_path)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile static-shape linalg MLIR with IREE")
    ap.add_argument("--input", type=Path, required=True, help="linalg MLIR input")
    ap.add_argument("--kind", choices=("matmul", "conv2d"), required=True)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--tile-m", type=int, default=8)
    ap.add_argument("--tile-n", type=int, default=8)
    ap.add_argument("--tile-k", type=int, default=8)
    ap.add_argument("--tile-oh", type=int, default=8)
    ap.add_argument("--tile-ow", type=int, default=8)
    ap.add_argument("--tile-oc", type=int, default=96)
    ap.add_argument("--no-run", action="store_true", help="write schedule/manifest but skip iree-opt")
    args = ap.parse_args()

    manifest = run_tile_mlir(args.input, args.kind, args.run_dir, args)
    print(f"wrote {manifest['input_with_schedule']}")
    if manifest["tiled_mlir"]:
        print(f"wrote {manifest['tiled_mlir']}")
    else:
        print("skipped iree-opt (--no-run)")
    print(f"wrote {manifest['manifest']} ({len(manifest['ops'])} ops)")


if __name__ == "__main__":
    main()
