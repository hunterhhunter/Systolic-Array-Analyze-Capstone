"""Generate and tile an MNK MatMul with IREE Transform Dialect.

This is the direct CLI entry point for custom GEMM experiments:

    python -m tools.iree_tile_mnk --m 32 --n 32 --k 64 \
      --tile-m 8 --tile-n 8 --tile-k 8 --emit-topology

It writes a merged payload+schedule MLIR file, runs iree-opt's
transform-interpreter, and can optionally emit a SCALE-Sim GEMM topology CSV
from the tiled MLIR.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class MnkTileSpec:
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    name: str = "matmul_mnk"

    @property
    def tile_label(self) -> str:
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}"

    @property
    def shape_label(self) -> str:
        return f"m{self.m}_n{self.n}_k{self.k}"

    @property
    def expected_tiles(self) -> int:
        return (
            math.ceil(self.m / self.tile_m)
            * math.ceil(self.n / self.tile_n)
            * math.ceil(self.k / self.tile_k)
        )


def _sanitize_symbol(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not out or out[0].isdigit():
        out = f"matmul_{out}"
    return out


def validate_spec(spec: MnkTileSpec) -> None:
    fields = {
        "m": spec.m,
        "n": spec.n,
        "k": spec.k,
        "tile_m": spec.tile_m,
        "tile_n": spec.tile_n,
        "tile_k": spec.tile_k,
    }
    bad = [name for name, value in fields.items() if value <= 0]
    if bad:
        raise ValueError(f"all MNK and tile dimensions must be positive: {bad}")


def render_merged_mlir(spec: MnkTileSpec) -> str:
    validate_spec(spec)
    fn = _sanitize_symbol(spec.name)
    return f"""// Auto-generated MNK MatMul + Transform Dialect schedule.
// Shape: A[{spec.m}x{spec.k}] * B[{spec.k}x{spec.n}] -> C[{spec.m}x{spec.n}]
// Tile: [M={spec.tile_m}, N={spec.tile_n}, K={spec.tile_k}]
// Expected full tiles: {spec.expected_tiles}

module attributes {{ transform.with_named_sequence }} {{

  func.func @{fn}(%A: tensor<{spec.m}x{spec.k}xf32>, %B: tensor<{spec.k}x{spec.n}xf32>) -> tensor<{spec.m}x{spec.n}xf32> {{
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<{spec.m}x{spec.n}xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<{spec.m}x{spec.n}xf32>) -> tensor<{spec.m}x{spec.n}xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<{spec.m}x{spec.k}xf32>, tensor<{spec.k}x{spec.n}xf32>)
        outs(%filled : tensor<{spec.m}x{spec.n}xf32>) -> tensor<{spec.m}x{spec.n}xf32>
    return %result : tensor<{spec.m}x{spec.n}xf32>
  }}

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %matmul = transform.structured.match ops{{["linalg.matmul"]}} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [{spec.tile_m}, {spec.tile_n}, {spec.tile_k}]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }}
}}
"""


def run_iree_transform(
    input_mlir: Path,
    tiled_mlir: Path,
    iree_opt: Path = IREE_OPT,
    *,
    stdout=None,
    stderr=None,
) -> None:
    if not iree_opt.exists():
        raise FileNotFoundError(
            f"IREE_OPT={iree_opt} not found. Run inside the Docker dev container, run `make env`, "
            "or set IREE_OPT=/path/to/iree-opt."
        )
    tiled_mlir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(iree_opt),
            str(input_mlir),
            "--pass-pipeline=builtin.module(transform-interpreter)",
            "-o",
            str(tiled_mlir),
        ],
        check=True,
        stdout=stdout,
        stderr=stderr,
    )


def tiles_to_gemm_csv_from_spec(spec: MnkTileSpec, *, max_tiles: int | None = None) -> str:
    validate_spec(spec)
    lines = ["Layer Name, M, N, K,"]
    idx = 0
    for m0 in range(0, spec.m, spec.tile_m):
        m = min(spec.tile_m, spec.m - m0)
        for n0 in range(0, spec.n, spec.tile_n):
            n = min(spec.tile_n, spec.n - n0)
            for k0 in range(0, spec.k, spec.tile_k):
                if max_tiles is not None and idx >= max_tiles:
                    return "\n".join(lines) + "\n"
                k = min(spec.tile_k, spec.k - k0)
                lines.append(f"Tile_{idx:03d}, {m}, {n}, {k},")
                idx += 1
    return "\n".join(lines) + "\n"


def write_topology_from_spec(spec: MnkTileSpec, topology_csv: Path, *, max_tiles: int | None = None) -> int:
    topology_csv.parent.mkdir(parents=True, exist_ok=True)
    topology_csv.write_text(tiles_to_gemm_csv_from_spec(spec, max_tiles=max_tiles))
    return spec.expected_tiles if max_tiles is None else min(spec.expected_tiles, max_tiles)


def default_out_dir(spec: MnkTileSpec) -> Path:
    return REPO / "outputs" / "mnk" / f"{spec.shape_label}_tile_{spec.tile_label}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and IREE-tile an MNK MatMul")
    ap.add_argument("--m", type=int, required=True, help="GEMM M dimension")
    ap.add_argument("--n", type=int, required=True, help="GEMM N dimension")
    ap.add_argument("--k", type=int, required=True, help="GEMM K dimension")
    ap.add_argument("--tile-m", type=int, required=True)
    ap.add_argument("--tile-n", type=int, required=True)
    ap.add_argument("--tile-k", type=int, required=True)
    ap.add_argument("--name", default="matmul_mnk", help="MLIR function symbol")
    ap.add_argument("--out-dir", type=Path, help="output directory")
    ap.add_argument("--input-mlir", type=Path, help="override generated input MLIR path")
    ap.add_argument("--tiled-mlir", type=Path, help="override tiled MLIR output path")
    ap.add_argument("--topology", type=Path, help="override topology CSV output path")
    ap.add_argument("--emit-topology", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="only write input MLIR")
    args = ap.parse_args()

    spec = MnkTileSpec(
        m=args.m,
        n=args.n,
        k=args.k,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        name=args.name,
    )
    out_dir = args.out_dir or default_out_dir(spec)
    input_mlir = args.input_mlir or out_dir / "input_with_schedule.mlir"
    tiled_mlir = args.tiled_mlir or out_dir / "tiled.mlir"
    topology_csv = args.topology or out_dir / "topology.csv"

    input_mlir.parent.mkdir(parents=True, exist_ok=True)
    input_mlir.write_text(render_merged_mlir(spec))
    print(f"wrote {input_mlir}")

    if args.no_run:
        return

    run_iree_transform(input_mlir, tiled_mlir)
    print(f"wrote {tiled_mlir}")

    if args.emit_topology:
        n_tiles = write_topology_from_spec(spec, topology_csv)
        print(f"wrote {topology_csv} ({n_tiles} tiles)")


if __name__ == "__main__":
    main()
