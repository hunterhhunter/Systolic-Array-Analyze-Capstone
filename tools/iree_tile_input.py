"""Tile a user-provided MLIR/ONNX input with IREE Transform Dialect.

This CLI is meant for shared model artifacts:

    python -m tools.iree_tile_input --input models/resnet18_linalg.mlir \
      --preset conv2d-nhwc-hwcf --tile-oh 8 --tile-ow 8 --tile-oc 96

The stable path is MLIR that already contains named linalg ops. ONNX is
supported only when an `iree-import-onnx` executable is available locally.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IREE_IMPORT_ONNX = os.environ.get("IREE_IMPORT_ONNX", "iree-import-onnx")


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
class TileConfig:
    ops: tuple[str, ...]
    tile_sizes: tuple[int, ...]
    num_loops: int

    @property
    def tile_label(self) -> str:
        return "x".join(str(size) for size in self.tile_sizes)


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def _parse_tile_sizes(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in re.split(r"[x,]", text) if part.strip()]
    if not parts:
        raise ValueError("--tile-sizes must contain at least one integer")
    sizes = tuple(int(part) for part in parts)
    if any(size < 0 for size in sizes):
        raise ValueError("tile sizes must be non-negative")
    if not any(size > 0 for size in sizes):
        raise ValueError("at least one tile size must be positive")
    return sizes


def build_tile_config(args: argparse.Namespace) -> TileConfig:
    if args.preset == "matmul":
        sizes = (args.tile_m, args.tile_n, args.tile_k)
        ops = ("linalg.matmul",)
    elif args.preset == "batch-matmul":
        sizes = (0, args.tile_m, args.tile_n, args.tile_k)
        ops = ("linalg.batch_matmul",)
    elif args.preset == "conv2d-nhwc-hwcf":
        sizes = (0, args.tile_oh, args.tile_ow, args.tile_oc, 0, 0, 0)
        ops = ("linalg.conv_2d_nhwc_hwcf",)
    elif args.preset == "custom":
        if not args.op:
            raise ValueError("--op is required with --preset custom")
        if not args.tile_sizes:
            raise ValueError("--tile-sizes is required with --preset custom")
        sizes = _parse_tile_sizes(args.tile_sizes)
        ops = tuple(args.op)
    else:
        raise ValueError(f"unknown preset: {args.preset}")

    if any(size < 0 for size in sizes):
        raise ValueError("tile sizes must be non-negative")
    if not any(size > 0 for size in sizes):
        raise ValueError("at least one tile size must be positive")

    num_loops = args.num_loops if args.num_loops is not None else sum(size > 0 for size in sizes)
    if num_loops <= 0:
        raise ValueError("--num-loops must be positive")
    return TileConfig(ops=ops, tile_sizes=sizes, num_loops=num_loops)


def render_transform_schedule(config: TileConfig) -> str:
    ops_literal = ", ".join(f'"{op}"' for op in config.ops)
    tile_literal = ", ".join(str(size) for size in config.tile_sizes)
    loop_types = ", ".join(["!transform.any_op"] * (config.num_loops + 1))
    return f"""  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %target = transform.structured.match ops{{[{ops_literal}]}} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:{config.num_loops} = transform.structured.tile_using_for %target tile_sizes [{tile_literal}]
      : (!transform.any_op) -> ({loop_types})
    transform.yield
  }}
"""


def _add_transform_attr_to_module_header(text: str) -> str:
    if "transform.with_named_sequence" in text[:512]:
        return text

    module_match = re.search(r"(^\s*module\b[^\n]*\{)", text, flags=re.MULTILINE)
    if not module_match:
        return f"module attributes {{ transform.with_named_sequence }} {{\n{text.rstrip()}\n}}\n"

    header = module_match.group(1)
    if "attributes" not in header:
        new_header = re.sub(
            r"\bmodule\b\s*\{",
            "module attributes { transform.with_named_sequence } {",
            header,
            count=1,
        )
        return text[: module_match.start(1)] + new_header + text[module_match.end(1) :]

    attr_match = re.match(r"(\s*module\b\s+attributes\s*\{)(.*)(\}\s*\{)", header)
    if attr_match:
        prefix, attrs, suffix = attr_match.groups()
        sep = ", " if attrs.strip() else " "
        new_header = f"{prefix}{attrs}{sep}transform.with_named_sequence {suffix}"
        return text[: module_match.start(1)] + new_header + text[module_match.end(1) :]

    return f"module attributes {{ transform.with_named_sequence }} {{\n{text.rstrip()}\n}}\n"


def attach_transform_schedule(payload_mlir: str, config: TileConfig) -> str:
    if "@__transform_main" in payload_mlir:
        raise ValueError("input MLIR already contains @__transform_main")

    text = _add_transform_attr_to_module_header(payload_mlir.rstrip())
    insert_at = text.rfind("}")
    if insert_at < 0:
        schedule = render_transform_schedule(config)
        return f"module attributes {{ transform.with_named_sequence }} {{\n{text}\n\n{schedule}}}\n"

    schedule = render_transform_schedule(config)
    return f"{text[:insert_at].rstrip()}\n\n{schedule}{text[insert_at:]}\n"


def import_onnx(input_path: Path, imported_mlir: Path, importer: str = IREE_IMPORT_ONNX) -> None:
    importer_path = shutil.which(importer) if not Path(importer).exists() else importer
    if not importer_path:
        raise FileNotFoundError(
            "ONNX input requires iree-import-onnx. Set IREE_IMPORT_ONNX or provide MLIR directly."
        )
    imported_mlir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(importer_path), str(input_path), "-o", str(imported_mlir)], check=True)


def run_iree_transform(input_mlir: Path, tiled_mlir: Path, iree_opt: Path = IREE_OPT) -> None:
    if not iree_opt.exists():
        raise FileNotFoundError(
            f"IREE_OPT={iree_opt} not found. Run `make env` or set IREE_OPT=/path/to/iree-opt."
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
    )


def default_out_dir(input_path: Path, config: TileConfig) -> Path:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", input_path.stem)
    return REPO / "outputs" / "iree_input_tiles" / f"{name}_tile_{config.tile_label}"


def tile_input(
    input_path: Path,
    config: TileConfig,
    out_dir: Path | None = None,
    imported_mlir: Path | None = None,
    input_with_schedule: Path | None = None,
    tiled_mlir: Path | None = None,
    no_run: bool = False,
) -> tuple[Path, Path | None]:
    input_path = _repo_path(input_path)
    out_dir = _repo_path(out_dir) if out_dir else default_out_dir(input_path, config)
    imported_mlir = _repo_path(imported_mlir) if imported_mlir else out_dir / "imported.mlir"
    input_with_schedule = (
        _repo_path(input_with_schedule)
        if input_with_schedule
        else out_dir / "input_with_schedule.mlir"
    )
    tiled_mlir = _repo_path(tiled_mlir) if tiled_mlir else out_dir / "tiled.mlir"

    if input_path.suffix.lower() == ".onnx":
        import_onnx(input_path, imported_mlir)
        payload_path = imported_mlir
    else:
        payload_path = input_path

    payload = payload_path.read_text()
    input_with_schedule.parent.mkdir(parents=True, exist_ok=True)
    input_with_schedule.write_text(attach_transform_schedule(payload, config))

    if no_run:
        return input_with_schedule, None

    run_iree_transform(input_with_schedule, tiled_mlir)
    return input_with_schedule, tiled_mlir


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile a user-provided MLIR/ONNX input with IREE")
    ap.add_argument("--input", type=Path, required=True, help="MLIR input, or ONNX if importer exists")
    ap.add_argument(
        "--preset",
        choices=("conv2d-nhwc-hwcf", "matmul", "batch-matmul", "custom"),
        default="conv2d-nhwc-hwcf",
    )
    ap.add_argument("--tile-oh", type=int, default=8)
    ap.add_argument("--tile-ow", type=int, default=8)
    ap.add_argument("--tile-oc", type=int, default=96)
    ap.add_argument("--tile-m", type=int, default=8)
    ap.add_argument("--tile-n", type=int, default=8)
    ap.add_argument("--tile-k", type=int, default=8)
    ap.add_argument("--op", action="append", help="custom linalg op name, repeatable")
    ap.add_argument("--tile-sizes", help="custom tile sizes, e.g. 0,8,8,96,0,0,0")
    ap.add_argument("--num-loops", type=int, help="override number of loop handles returned")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--imported-mlir", type=Path)
    ap.add_argument("--input-with-schedule", type=Path)
    ap.add_argument("--tiled-mlir", type=Path)
    ap.add_argument("--no-run", action="store_true", help="only write input_with_schedule.mlir")
    args = ap.parse_args()

    config = build_tile_config(args)
    scheduled, tiled = tile_input(
        input_path=args.input,
        config=config,
        out_dir=args.out_dir,
        imported_mlir=args.imported_mlir,
        input_with_schedule=args.input_with_schedule,
        tiled_mlir=args.tiled_mlir,
        no_run=args.no_run,
    )
    print(f"wrote {scheduled}")
    if tiled:
        print(f"wrote {tiled}")
    else:
        print("skipped iree-opt (--no-run)")


if __name__ == "__main__":
    main()
