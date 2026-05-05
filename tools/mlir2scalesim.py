"""Tiled MLIR → SCALE-Sim GEMM topology CSV.

Two-tier parser per plan §4-3-1:
  1) iree.compiler.ir.Module.parse() + walk() — primary, type-safe IR traversal
  2) text-regex fallback — engaged only if IR-binding parse raises

Phase 3 scope: walking_skeleton-style matmul (3-level scf.for + innermost linalg.matmul,
f32, GEMM). Conv emitter 및 non-f32 dtype 지원은 Phase 3 후반/Phase 4에서 확장.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import NamedTuple


class TileShape(NamedTuple):
    M: int
    N: int
    K: int


class TiledMatmul(NamedTuple):
    tile: TileShape
    loop_counts: tuple[int, ...]
    total_tiles: int


_INDEX_CONST_RE = re.compile(r"^\s*(-?\d+)\s*:\s*index\s*$")

_REGEX_CONST_INDEX_RE = re.compile(r"(%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index")
_REGEX_SCF_FOR_RE = re.compile(r"scf\.for\s+%\w+\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)")
_REGEX_LINALG_MATMUL_RE = re.compile(
    r"linalg\.matmul\s+ins\(\s*[^:]+:\s*tensor<(\d+)x(\d+)xf32>\s*,\s*"
    r"tensor<(\d+)x(\d+)xf32>\s*\)\s*outs\(\s*[^:]+:\s*tensor<(\d+)x(\d+)xf32>\s*\)"
)


def _ceil_loop_count(lb: int, ub: int, step: int) -> int:
    """Return the exact scf.for iteration count for constant positive steps.

    scf.for executes lb, lb+step, ... while the induction variable is < ub.
    This is ceil((ub - lb) / step), clamped at zero for empty ranges.
    """
    if step <= 0:
        raise ValueError(f"non-positive scf.for step: {step}")
    extent = ub - lb
    if extent <= 0:
        return 0
    return (extent + step - 1) // step


def _build(tile: TileShape, loop_counts: list[int]) -> TiledMatmul:
    if not loop_counts:
        raise ValueError("no scf.for loops found — input may not be a tiled matmul")
    total = 1
    for c in loop_counts:
        total *= c
    return TiledMatmul(tile=tile, loop_counts=tuple(loop_counts), total_tiles=total)


def _parse_via_ir(text: str) -> TiledMatmul:
    from iree.compiler.ir import Context, Module, RankedTensorType, WalkResult

    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        mod = Module.parse(text)

        const_by_op: dict = {}
        scf_fors: list = []
        matmuls: list = []

        def cb(op):
            name = op.operation.name
            if name == "arith.constant":
                m = _INDEX_CONST_RE.match(str(op.operation.attributes["value"]))
                if m:
                    const_by_op[op.operation] = int(m.group(1))
            elif name == "scf.for":
                scf_fors.append(op.operation)
            elif name == "linalg.matmul":
                matmuls.append(op.operation)
            return WalkResult.ADVANCE

        mod.operation.walk(cb)

        if not matmuls:
            raise ValueError("no linalg.matmul found")

        loop_counts: list[int] = []
        for fo in scf_fors:
            owners = [fo.operands[i].owner for i in range(3)]
            missing = [role for role, o in zip(("lb", "ub", "step"), owners)
                       if o not in const_by_op]
            if missing:
                raise ValueError(
                    f"scf.for bound not an index arith.constant: {missing}"
                )
            lb, ub, step = (const_by_op[o] for o in owners)
            loop_counts.append(_ceil_loop_count(lb, ub, step))

        mm = matmuls[0]
        a = list(RankedTensorType(mm.operands[0].type).shape)
        b = list(RankedTensorType(mm.operands[1].type).shape)
        c = list(RankedTensorType(mm.operands[2].type).shape)
        if not (len(a) == len(b) == len(c) == 2):
            raise ValueError(f"non-2D matmul operand shapes: A={a} B={b} C={c}")
        M, K1 = a
        K2, N = b
        Mo, No = c
        if K1 != K2 or M != Mo or N != No:
            raise ValueError(f"inconsistent matmul shapes: A={a} B={b} C={c}")

        return _build(TileShape(M=M, N=N, K=K1), loop_counts)


def _parse_via_regex(text: str) -> TiledMatmul:
    consts = {name: int(val) for name, val in _REGEX_CONST_INDEX_RE.findall(text)}

    loop_counts: list[int] = []
    for lb, ub, step in _REGEX_SCF_FOR_RE.findall(text):
        lb_v, ub_v, step_v = consts[lb], consts[ub], consts[step]
        loop_counts.append(_ceil_loop_count(lb_v, ub_v, step_v))

    mm = _REGEX_LINALG_MATMUL_RE.search(text)
    if not mm:
        raise ValueError("no linalg.matmul found (regex)")

    m_a, k_a, k_b, n_b, m_o, n_o = map(int, mm.groups())
    if k_a != k_b or m_a != m_o or n_b != n_o:
        raise ValueError(
            f"inconsistent matmul shapes (regex): A={m_a}x{k_a} B={k_b}x{n_b} C={m_o}x{n_o}"
        )

    return _build(TileShape(M=m_a, N=n_b, K=k_a), loop_counts)


def parse_tiled_matmul(mlir_text: str) -> TiledMatmul:
    """Primary IR-binding parse, regex fallback only when binding itself fails."""
    try:
        return _parse_via_ir(mlir_text)
    except Exception as e:
        warnings.warn(
            f"IR-binding parse failed ({type(e).__name__}: {e}); using regex fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        return _parse_via_regex(mlir_text)


def tiles_to_gemm_csv(parsed: TiledMatmul) -> str:
    lines = ["Layer Name, M, N, K,"]
    for i in range(parsed.total_tiles):
        lines.append(f"Tile_{i:03d}, {parsed.tile.M}, {parsed.tile.N}, {parsed.tile.K},")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="tiled MLIR → SCALE-Sim GEMM topology CSV")
    ap.add_argument("mlir_path", type=Path, help="tiled MLIR (iree-opt output)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output CSV path")
    ap.add_argument("--kind", choices=["gemm"], default="gemm")
    ap.add_argument(
        "--parser",
        choices=["auto", "ir", "regex"],
        default="auto",
        help="force parsing path (default: auto = IR binding with regex fallback)",
    )
    args = ap.parse_args()

    text = args.mlir_path.read_text()
    if args.parser == "ir":
        parsed = _parse_via_ir(text)
    elif args.parser == "regex":
        parsed = _parse_via_regex(text)
    else:
        parsed = parse_tiled_matmul(text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tiles_to_gemm_csv(parsed))
    print(
        f"wrote {parsed.total_tiles} tiles "
        f"({parsed.tile.M}x{parsed.tile.N}x{parsed.tile.K}) "
        f"from loops {parsed.loop_counts} → {args.output}"
    )


if __name__ == "__main__":
    main()
