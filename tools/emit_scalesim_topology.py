"""Public CLI: tile manifest -> SCALE-Sim topology CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.linalg_mlir import parse_linalg_ops, read_manifest, repo_path

REPO = Path(__file__).resolve().parent.parent


def _manifest_ops(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    ops = manifest.get("ops")
    if ops:
        return list(ops)

    source = manifest.get("source_mlir")
    if not source:
        raise ValueError("manifest must contain ops or source_mlir")
    parsed = parse_linalg_ops(repo_path(source).read_text(), manifest["kind"])
    return [
        {"name": op.name, "kind": op.kind, "shape": op.shape}
        for op in parsed
    ]


def emit_matmul_topology(manifest: dict[str, Any]) -> str:
    tile = manifest["tile"]
    tile_m, tile_n, tile_k = int(tile["m"]), int(tile["n"]), int(tile["k"])
    lines = ["Layer Name, M, N, K,"]
    for op in _manifest_ops(manifest):
        shape = op["shape"]
        idx = 0
        for m0 in range(0, int(shape["m"]), tile_m):
            m = min(tile_m, int(shape["m"]) - m0)
            for n0 in range(0, int(shape["n"]), tile_n):
                n = min(tile_n, int(shape["n"]) - n0)
                for k0 in range(0, int(shape["k"]), tile_k):
                    k = min(tile_k, int(shape["k"]) - k0)
                    name = f"{op['name']}_m{m0:04d}_n{n0:04d}_k{k0:04d}_{idx:03d}"
                    lines.append(f"{name}, {m}, {n}, {k},")
                    idx += 1
    return "\n".join(lines) + "\n"


def emit_conv2d_topology(manifest: dict[str, Any]) -> str:
    tile = manifest["tile"]
    tile_oh, tile_ow, tile_oc = int(tile["oh"]), int(tile["ow"]), int(tile["oc"])
    lines = [
        "Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,"
    ]
    for op in _manifest_ops(manifest):
        shape = op["shape"]
        stride_h, stride_w = int(shape["stride_h"]), int(shape["stride_w"])
        dilation_h, dilation_w = int(shape["dilation_h"]), int(shape["dilation_w"])
        if stride_h != stride_w:
            raise ValueError(f"SCALE-Sim topology supports one stride, got {op['name']}: {stride_h}x{stride_w}")
        if dilation_h != 1 or dilation_w != 1:
            raise ValueError(f"SCALE-Sim topology does not model dilation, got {op['name']}: {dilation_h}x{dilation_w}")

        ofmap_h, ofmap_w = int(shape["ofmap_h"]), int(shape["ofmap_w"])
        num_filters = int(shape["num_filters"])
        filter_h, filter_w = int(shape["filter_h"]), int(shape["filter_w"])
        channels = int(shape["channels"])
        for oh0 in range(0, ofmap_h, tile_oh):
            actual_oh = min(tile_oh, ofmap_h - oh0)
            ifmap_h = (actual_oh - 1) * stride_h + filter_h
            for ow0 in range(0, ofmap_w, tile_ow):
                actual_ow = min(tile_ow, ofmap_w - ow0)
                ifmap_w = (actual_ow - 1) * stride_w + filter_w
                for oc0 in range(0, num_filters, tile_oc):
                    actual_oc = min(tile_oc, num_filters - oc0)
                    name = f"{op['name']}_oh{oh0:03d}_ow{ow0:03d}_oc{oc0:04d}"
                    lines.append(
                        f"{name}, {ifmap_h}, {ifmap_w}, {filter_h}, {filter_w}, "
                        f"{channels}, {actual_oc}, {stride_h},"
                    )
    return "\n".join(lines) + "\n"


def emit_topology(manifest: dict[str, Any]) -> str:
    kind = manifest["kind"]
    if kind == "matmul":
        return emit_matmul_topology(manifest)
    if kind == "conv2d":
        return emit_conv2d_topology(manifest)
    raise ValueError(f"unsupported manifest kind: {kind}")


def write_topology(manifest_path: Path, output_path: Path) -> int:
    manifest = read_manifest(manifest_path)
    text = emit_topology(manifest)
    output_path = repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return max(0, len(text.splitlines()) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit SCALE-Sim topology from tile_manifest.json")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="default: <manifest-dir>/topology.csv")
    args = ap.parse_args()

    manifest_path = repo_path(args.manifest)
    output = repo_path(args.output) if args.output else manifest_path.parent / "topology.csv"
    n_tiles = write_topology(manifest_path, output)
    print(f"wrote {output} ({n_tiles} tiles)")


if __name__ == "__main__":
    main()
