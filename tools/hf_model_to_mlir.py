"""Download a HuggingFace Transformers model and emit shape-only linalg MLIR.

The emitted MLIR is a tiling/simulation artifact, not a numerically equivalent
full-model lowering. It records Conv2d and Linear module shapes from a real
downloaded model, then emits independent linalg functions that IREE can tile.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.iree_tile_input import TileConfig, tile_input

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)


@dataclass(frozen=True)
class LayerRecord:
    index: int
    name: str
    kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    groups: int = 1


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def _safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    return out.strip("._-") or "model"


def _sanitize_symbol(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not out or out[0].isdigit():
        out = f"layer_{out}"
    return out


def _as_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    if isinstance(value, list):
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return (value, value)
    return (0, 0)


def parse_input_shape(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None:
        return None
    parts = [part.strip() for part in re.split(r"[x,]", text) if part.strip()]
    if len(parts) != 4:
        raise ValueError("--input-shape must be N,C,H,W, e.g. 1,3,224,224")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError("--input-shape dimensions must be positive")
    return shape  # type: ignore[return-value]


def infer_default_input_shape(model: Any) -> tuple[int, int, int, int]:
    config = getattr(model, "config", None)
    channels = int(getattr(config, "num_channels", DEFAULT_INPUT_SHAPE[1]))
    image_size = getattr(config, "image_size", None)
    if isinstance(image_size, int):
        return (1, channels, image_size, image_size)
    if isinstance(image_size, (tuple, list)) and len(image_size) >= 2:
        return (1, channels, int(image_size[0]), int(image_size[1]))
    return DEFAULT_INPUT_SHAPE


def load_transformers_model(
    model_id: str,
    *,
    task: str,
    revision: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
    trust_remote_code: bool,
    from_tf: bool,
    from_flax: bool,
) -> Any:
    try:
        from transformers import AutoModel, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace conversion requires optional deps. Install with: "
            "uv pip install --python .venv/bin/python -r requirements-model.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "from_tf": from_tf,
        "from_flax": from_flax,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if task == "image-classification":
        return AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
    if task == "auto-model":
        return AutoModel.from_pretrained(model_id, **kwargs)

    try:
        return AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
    except Exception:
        return AutoModel.from_pretrained(model_id, **kwargs)


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _first_tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = _tensor_shape(value)
    if shape:
        return shape
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = _first_tensor_shape(item)
            if shape:
                return shape
    return None


def collect_layer_records(
    model: Any,
    input_shape: tuple[int, int, int, int],
    *,
    device: str = "cpu",
    max_layers: int | None = None,
) -> list[LayerRecord]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace conversion requires torch. Install requirements-model.txt."
        ) from exc

    records: list[LayerRecord] = []
    handles = []

    def make_hook(name: str, module: Any):
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if max_layers is not None and len(records) >= max_layers:
                return
            input_shape_seen = _first_tensor_shape(inputs)
            output_shape_seen = _first_tensor_shape(output)
            weight_shape = _tensor_shape(getattr(module, "weight", None))
            if not input_shape_seen or not output_shape_seen or not weight_shape:
                return

            kind = "conv2d" if isinstance(module, torch.nn.Conv2d) else "linear"
            records.append(
                LayerRecord(
                    index=len(records),
                    name=name,
                    kind=kind,
                    input_shape=input_shape_seen,
                    output_shape=output_shape_seen,
                    weight_shape=weight_shape,
                    stride=_as_pair(getattr(module, "stride", (1, 1))),
                    dilation=_as_pair(getattr(module, "dilation", (1, 1))),
                    padding=_as_pair(getattr(module, "padding", (0, 0))),
                    groups=int(getattr(module, "groups", 1)),
                )
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    model.to(device)
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    try:
        with torch.no_grad():
            try:
                model(pixel_values=dummy)
            except TypeError:
                model(dummy)
    finally:
        for handle in handles:
            handle.remove()

    return records


def _conv_effective_input_hw(record: LayerRecord) -> tuple[int, int]:
    if len(record.output_shape) != 4:
        raise ValueError(f"conv layer output must be NCHW: {record.name}")
    out_h, out_w = record.output_shape[2], record.output_shape[3]
    kernel_h, kernel_w = record.weight_shape[2], record.weight_shape[3]
    stride_h, stride_w = record.stride
    dilation_h, dilation_w = record.dilation
    eff_h = (out_h - 1) * stride_h + dilation_h * (kernel_h - 1) + 1
    eff_w = (out_w - 1) * stride_w + dilation_w * (kernel_w - 1) + 1
    return eff_h, eff_w


def render_layer_mlir(records: list[LayerRecord], *, include_linear: bool = True) -> str:
    funcs: list[str] = []
    skipped: list[str] = []

    for record in records:
        symbol = _sanitize_symbol(f"{record.index:03d}_{record.name}")
        if record.kind == "conv2d":
            if record.groups != 1:
                skipped.append(
                    f"// skipped grouped conv {record.name}: groups={record.groups}"
                )
                continue
            if len(record.input_shape) != 4 or len(record.output_shape) != 4:
                skipped.append(f"// skipped conv {record.name}: non-NCHW shape")
                continue

            batch = record.output_shape[0]
            channels = record.input_shape[1]
            out_h, out_w, out_channels = (
                record.output_shape[2],
                record.output_shape[3],
                record.output_shape[1],
            )
            kernel_h, kernel_w = record.weight_shape[2], record.weight_shape[3]
            eff_h, eff_w = _conv_effective_input_hw(record)
            stride_h, stride_w = record.stride
            dilation_h, dilation_w = record.dilation
            funcs.append(
                f"""  // source: {record.name}, NCHW input={record.input_shape}, output={record.output_shape}, padding={record.padding}
  func.func @{symbol}(
      %input: tensor<{batch}x{eff_h}x{eff_w}x{channels}xf32>,
      %filter: tensor<{kernel_h}x{kernel_w}x{channels}x{out_channels}xf32>) -> tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32> {{
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32>) -> tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32>
    %result = linalg.conv_2d_nhwc_hwcf
        {{dilations = dense<[{dilation_h}, {dilation_w}]> : vector<2xi64>, strides = dense<[{stride_h}, {stride_w}]> : vector<2xi64>}}
        ins(%input, %filter : tensor<{batch}x{eff_h}x{eff_w}x{channels}xf32>, tensor<{kernel_h}x{kernel_w}x{channels}x{out_channels}xf32>)
        outs(%filled : tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32>) -> tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32>
    return %result : tensor<{batch}x{out_h}x{out_w}x{out_channels}xf32>
  }}"""
            )
            continue

        if record.kind == "linear" and include_linear:
            in_features = record.weight_shape[1]
            out_features = record.weight_shape[0]
            funcs.append(
                f"""  // source: {record.name}, input={record.input_shape}, output={record.output_shape}
  func.func @{symbol}(
      %input: tensor<1x{in_features}xf32>,
      %weight: tensor<{in_features}x{out_features}xf32>) -> tensor<1x{out_features}xf32> {{
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x{out_features}xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x{out_features}xf32>) -> tensor<1x{out_features}xf32>
    %result = linalg.matmul
        ins(%input, %weight : tensor<1x{in_features}xf32>, tensor<{in_features}x{out_features}xf32>)
        outs(%filled : tensor<1x{out_features}xf32>) -> tensor<1x{out_features}xf32>
    return %result : tensor<1x{out_features}xf32>
  }}"""
            )

    body = "\n\n".join([*skipped, *funcs])
    return f"""// Auto-generated shape-only linalg MLIR from HuggingFace module shapes.
// Conv tensors are emitted in NHWC/HWCF form for linalg.conv_2d_nhwc_hwcf.
// Linear tensors are emitted as 1xK by KxN linalg.matmul.

module {{

{body}

}}
"""


def write_summary_csv(records: list[LayerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "name",
        "kind",
        "input_shape",
        "output_shape",
        "weight_shape",
        "stride",
        "dilation",
        "padding",
        "groups",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "index": record.index,
                    "name": record.name,
                    "kind": record.kind,
                    "input_shape": "x".join(str(dim) for dim in record.input_shape),
                    "output_shape": "x".join(str(dim) for dim in record.output_shape),
                    "weight_shape": "x".join(str(dim) for dim in record.weight_shape),
                    "stride": "x".join(str(dim) for dim in record.stride),
                    "dilation": "x".join(str(dim) for dim in record.dilation),
                    "padding": "x".join(str(dim) for dim in record.padding),
                    "groups": record.groups,
                }
            )


def convert_hf_model(
    model_id: str,
    *,
    output_dir: Path | None = None,
    mlir_output: Path | None = None,
    summary_csv: Path | None = None,
    input_shape: tuple[int, int, int, int] | None = None,
    task: str = "auto",
    revision: str | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    from_tf: bool = False,
    from_flax: bool = False,
    device: str = "cpu",
    max_layers: int | None = None,
    include_linear: bool = True,
) -> tuple[Path, Path, list[LayerRecord]]:
    output_dir = _repo_path(output_dir) if output_dir else REPO / "models" / "hf" / _safe_name(model_id)
    mlir_output = _repo_path(mlir_output) if mlir_output else output_dir / "model_linalg.mlir"
    summary_csv = _repo_path(summary_csv) if summary_csv else output_dir / "layers.csv"
    cache_dir = _repo_path(cache_dir) if cache_dir else None

    model = load_transformers_model(
        model_id,
        task=task,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        from_tf=from_tf,
        from_flax=from_flax,
    )
    shape = input_shape or infer_default_input_shape(model)
    records = collect_layer_records(model, shape, device=device, max_layers=max_layers)
    if not records:
        raise RuntimeError("no Conv2d/Linear layers were observed during dummy forward")

    mlir_output.parent.mkdir(parents=True, exist_ok=True)
    mlir_output.write_text(render_layer_mlir(records, include_linear=include_linear))
    write_summary_csv(records, summary_csv)
    return mlir_output, summary_csv, records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download a HuggingFace model and emit shape-only linalg MLIR"
    )
    ap.add_argument("--model-id", required=True, help="e.g. microsoft/resnet-50")
    ap.add_argument("--task", choices=("auto", "image-classification", "auto-model"), default="auto")
    ap.add_argument("--revision")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--from-tf", action="store_true", help="load TensorFlow .h5 weights if needed")
    ap.add_argument("--from-flax", action="store_true")
    ap.add_argument("--input-shape", help="dummy input N,C,H,W, default from config or 1,3,224,224")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-layers", type=int)
    ap.add_argument("--no-linear", action="store_true", help="omit Linear layers from emitted MLIR")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--mlir-output", type=Path)
    ap.add_argument("--summary-csv", type=Path)
    ap.add_argument("--tile", action="store_true", help="also run IREE transform tiling on conv ops")
    ap.add_argument("--tile-oh", type=int, default=8)
    ap.add_argument("--tile-ow", type=int, default=8)
    ap.add_argument("--tile-oc", type=int, default=96)
    ap.add_argument("--tile-out-dir", type=Path)
    args = ap.parse_args()

    mlir_output, summary_csv, records = convert_hf_model(
        args.model_id,
        output_dir=args.output_dir,
        mlir_output=args.mlir_output,
        summary_csv=args.summary_csv,
        input_shape=parse_input_shape(args.input_shape),
        task=args.task,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        from_tf=args.from_tf,
        from_flax=args.from_flax,
        device=args.device,
        max_layers=args.max_layers,
        include_linear=not args.no_linear,
    )

    print(f"wrote {mlir_output}")
    print(f"wrote {summary_csv}")
    print(f"observed {len(records)} Conv2d/Linear layers")

    if args.tile:
        scheduled, tiled = tile_input(
            mlir_output,
            TileConfig(
                ops=("linalg.conv_2d_nhwc_hwcf",),
                tile_sizes=(0, args.tile_oh, args.tile_ow, args.tile_oc, 0, 0, 0),
                num_loops=3,
            ),
            out_dir=args.tile_out_dir,
        )
        print(f"wrote {scheduled}")
        print(f"wrote {tiled}")


if __name__ == "__main__":
    main()
