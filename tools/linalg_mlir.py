"""Small static-shape linalg MLIR helpers for experiment scripts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

REPO = Path(__file__).resolve().parent.parent
Kind = Literal["matmul", "conv2d"]


@dataclass(frozen=True)
class LinalgOp:
    name: str
    kind: Kind
    shape: dict[str, Any]


def repo_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def display_path(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text())


def write_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def ops_to_json(ops: list[LinalgOp]) -> list[dict[str, Any]]:
    return [asdict(op) for op in ops]


def _nearest_func_name(text: str, offset: int, fallback: str) -> str:
    matches = list(re.finditer(r"func\.func\s+@([A-Za-z_.$][A-Za-z0-9_.$-]*)", text[:offset]))
    if not matches:
        return fallback
    return re.sub(r"[^A-Za-z0-9_]", "_", matches[-1].group(1))


_MATMUL_RE = re.compile(
    r"linalg\.matmul\s+ins\(\s*[^:]+:\s*tensor<(\d+)x(\d+)xf32>\s*,\s*"
    r"tensor<(\d+)x(\d+)xf32>\s*\)\s*outs\(\s*[^:]+:\s*"
    r"tensor<(\d+)x(\d+)xf32>\s*\)",
    re.DOTALL,
)

_CONV_RE = re.compile(
    r"linalg\.conv_2d_nhwc_hwcf(?P<attrs>.*?)"
    r"ins\(\s*[^:]+:\s*tensor<(?P<batch>\d+)x(?P<ifmap_h>\d+)x(?P<ifmap_w>\d+)x(?P<channels>\d+)xf32>\s*,\s*"
    r"tensor<(?P<filter_h>\d+)x(?P<filter_w>\d+)x(?P<filter_channels>\d+)x(?P<num_filters>\d+)xf32>\s*\)\s*"
    r"outs\(\s*[^:]+:\s*tensor<(?P<out_batch>\d+)x(?P<ofmap_h>\d+)x(?P<ofmap_w>\d+)x(?P<out_filters>\d+)xf32>\s*\)",
    re.DOTALL,
)


def _dense_pair(attrs: str, name: str, default: tuple[int, int]) -> tuple[int, int]:
    match = re.search(rf"{name}\s*=\s*dense<([^>]+)>", attrs)
    if not match:
        return default
    raw = match.group(1).strip()
    vals = [int(x) for x in re.findall(r"-?\d+", raw)]
    if len(vals) == 1:
        return (vals[0], vals[0])
    if len(vals) >= 2:
        return (vals[0], vals[1])
    return default


def _torch_attr_int(attrs: str, name: str, default: int = 0) -> int:
    match = re.search(rf"torch\.onnx\.{name}\s*=\s*(-?\d+)\s*:\s*si64", attrs)
    if not match:
        return default
    return int(match.group(1))


def _torch_attr_pair(attrs: str, name: str, default: tuple[int, int]) -> tuple[int, int]:
    match = re.search(rf"torch\.onnx\.{name}\s*=\s*\[([^\]]+)\]", attrs)
    if not match:
        return default
    raw = match.group(1)
    vals = [int(x) for x in re.findall(r"(-?\d+)\s*:\s*si\d+", raw)]
    if not vals:
        vals = [int(x) for x in re.findall(r"-?\d+", raw)]
    if len(vals) == 1:
        return (vals[0], vals[0])
    if len(vals) >= 2:
        return (vals[0], vals[1])
    return default


def parse_matmul_ops(text: str) -> list[LinalgOp]:
    ops: list[LinalgOp] = []
    used_names: dict[str, int] = {}
    for idx, match in enumerate(_MATMUL_RE.finditer(text)):
        m_a, k_a, k_b, n_b, m_o, n_o = map(int, match.groups())
        if k_a != k_b or m_a != m_o or n_b != n_o:
            raise ValueError(
                f"inconsistent matmul shapes: A={m_a}x{k_a} B={k_b}x{n_b} C={m_o}x{n_o}"
            )
        base = _nearest_func_name(text, match.start(), f"matmul_{idx:03d}")
        serial = used_names.get(base, 0)
        used_names[base] = serial + 1
        name = base if serial == 0 else f"{base}_{serial:03d}"
        ops.append(
            LinalgOp(
                name=name,
                kind="matmul",
                shape={"m": m_a, "n": n_b, "k": k_a},
            )
        )
    return ops


_TORCH_ONNX_CONV_RE = re.compile(
    r"torch\.operator\s+\"onnx\.Conv\"\([^)]*\)\s*\{(?P<attrs>.*?)\}\s*:\s*"
    r"\(!torch\.vtensor<\[(?P<batch>\d+),(?P<channels>\d+),(?P<ifmap_h>\d+),(?P<ifmap_w>\d+)\],f32>\s*,\s*"
    r"!torch\.vtensor<\[(?P<num_filters>\d+),(?P<filter_channels>\d+),(?P<filter_h>\d+),(?P<filter_w>\d+)\],f32>"
    r"(?:\s*,\s*!torch\.vtensor<\[(?P<bias>\d+)\],f32>)?\)\s*->\s*"
    r"!torch\.vtensor<\[(?P<out_batch>\d+),(?P<out_filters>\d+),(?P<ofmap_h>\d+),(?P<ofmap_w>\d+)\],f32>",
    re.DOTALL,
)


def parse_torch_onnx_conv_ops(text: str) -> list[LinalgOp]:
    """Parse IREE ONNX-importer torch.operator "onnx.Conv" ops.

    This is a bridge for real ONNX-imported artifacts. It extracts static NCHW
    Conv shapes and records them in the same shape schema used by the linalg
    topology emitter.
    """
    ops: list[LinalgOp] = []
    for idx, match in enumerate(_TORCH_ONNX_CONV_RE.finditer(text)):
        values = {
            key: int(match.group(key))
            for key in (
                "batch",
                "channels",
                "ifmap_h",
                "ifmap_w",
                "num_filters",
                "filter_channels",
                "filter_h",
                "filter_w",
                "out_batch",
                "out_filters",
                "ofmap_h",
                "ofmap_w",
            )
        }
        if values["channels"] != values["filter_channels"]:
            raise ValueError(f"onnx.Conv channels mismatch in op {idx}: {values}")
        if values["batch"] != values["out_batch"]:
            raise ValueError(f"onnx.Conv batch mismatch in op {idx}: {values}")
        if values["num_filters"] != values["out_filters"]:
            raise ValueError(f"onnx.Conv output filters mismatch in op {idx}: {values}")

        attrs = match.group("attrs")
        stride = _torch_attr_pair(attrs, "strides", (1, 1))
        dilation = _torch_attr_pair(attrs, "dilations", (1, 1))
        pads = re.search(r"torch\.onnx\.pads\s*=\s*\[([^\]]+)\]", attrs)
        pad_vals = (
            [int(x) for x in re.findall(r"(-?\d+)\s*:\s*si\d+", pads.group(1))]
            if pads
            else [0, 0, 0, 0]
        )
        if pads and not pad_vals:
            pad_vals = [int(x) for x in re.findall(r"-?\d+", pads.group(1))]
        group = _torch_attr_int(attrs, "group", 1)
        base = _nearest_func_name(text, match.start(), "main_graph")
        name = f"{base}_onnx_conv_{idx:03d}"
        ops.append(
            LinalgOp(
                name=name,
                kind="conv2d",
                shape={
                    "batch": values["batch"],
                    "ifmap_h": values["ifmap_h"],
                    "ifmap_w": values["ifmap_w"],
                    "filter_h": values["filter_h"],
                    "filter_w": values["filter_w"],
                    "channels": values["channels"],
                    "num_filters": values["num_filters"],
                    "ofmap_h": values["ofmap_h"],
                    "ofmap_w": values["ofmap_w"],
                    "stride_h": stride[0],
                    "stride_w": stride[1],
                    "dilation_h": dilation[0],
                    "dilation_w": dilation[1],
                    "pad_top": pad_vals[0] if len(pad_vals) > 0 else 0,
                    "pad_left": pad_vals[1] if len(pad_vals) > 1 else 0,
                    "pad_bottom": pad_vals[2] if len(pad_vals) > 2 else 0,
                    "pad_right": pad_vals[3] if len(pad_vals) > 3 else 0,
                    "group": group,
                    "source_dialect": "torch_onnx",
                },
            )
        )
    return ops


def parse_conv2d_ops(text: str) -> list[LinalgOp]:
    ops: list[LinalgOp] = []
    used_names: dict[str, int] = {}
    for idx, match in enumerate(_CONV_RE.finditer(text)):
        values = {
            key: int(match.group(key))
            for key in (
                "batch",
                "ifmap_h",
                "ifmap_w",
                "channels",
                "filter_h",
                "filter_w",
                "filter_channels",
                "num_filters",
                "out_batch",
                "ofmap_h",
                "ofmap_w",
                "out_filters",
            )
        }
        if values["channels"] != values["filter_channels"]:
            raise ValueError(f"conv channels mismatch in op {idx}: {values}")
        if values["batch"] != values["out_batch"]:
            raise ValueError(f"conv batch mismatch in op {idx}: {values}")
        if values["num_filters"] != values["out_filters"]:
            raise ValueError(f"conv output filters mismatch in op {idx}: {values}")

        attrs = match.group("attrs")
        stride = _dense_pair(attrs, "strides", (1, 1))
        dilation = _dense_pair(attrs, "dilations", (1, 1))
        base = _nearest_func_name(text, match.start(), f"conv2d_{idx:03d}")
        serial = used_names.get(base, 0)
        used_names[base] = serial + 1
        name = base if serial == 0 else f"{base}_{serial:03d}"
        ops.append(
            LinalgOp(
                name=name,
                kind="conv2d",
                shape={
                    "batch": values["batch"],
                    "ifmap_h": values["ifmap_h"],
                    "ifmap_w": values["ifmap_w"],
                    "filter_h": values["filter_h"],
                    "filter_w": values["filter_w"],
                    "channels": values["channels"],
                    "num_filters": values["num_filters"],
                    "ofmap_h": values["ofmap_h"],
                    "ofmap_w": values["ofmap_w"],
                    "stride_h": stride[0],
                    "stride_w": stride[1],
                    "dilation_h": dilation[0],
                    "dilation_w": dilation[1],
                },
            )
        )
    return ops


def render_conv2d_linalg_module(ops: list[LinalgOp]) -> str:
    funcs: list[str] = []
    for op in ops:
        shape = op.shape
        group = int(shape.get("group", 1))
        if group != 1:
            raise ValueError(f"grouped conv is not supported for linalg bridge: {op.name}, group={group}")

        batch = int(shape["batch"])
        channels = int(shape["channels"])
        filter_h = int(shape["filter_h"])
        filter_w = int(shape["filter_w"])
        num_filters = int(shape["num_filters"])
        ofmap_h = int(shape["ofmap_h"])
        ofmap_w = int(shape["ofmap_w"])
        stride_h = int(shape["stride_h"])
        stride_w = int(shape["stride_w"])
        dilation_h = int(shape["dilation_h"])
        dilation_w = int(shape["dilation_w"])
        effective_h = (ofmap_h - 1) * stride_h + dilation_h * (filter_h - 1) + 1
        effective_w = (ofmap_w - 1) * stride_w + dilation_w * (filter_w - 1) + 1
        symbol = re.sub(r"[^A-Za-z0-9_]", "_", op.name)
        if not symbol or symbol[0].isdigit():
            symbol = f"conv_{symbol}"
        funcs.append(
            f"""  // source: {op.name}, original_ifmap={shape.get('ifmap_h')}x{shape.get('ifmap_w')}, pads={shape.get('pad_top', 0)}x{shape.get('pad_left', 0)}x{shape.get('pad_bottom', 0)}x{shape.get('pad_right', 0)}
  func.func @{symbol}(
      %input: tensor<{batch}x{effective_h}x{effective_w}x{channels}xf32>,
      %filter: tensor<{filter_h}x{filter_w}x{channels}x{num_filters}xf32>) -> tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32> {{
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32>) -> tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32>
    %result = linalg.conv_2d_nhwc_hwcf
        {{dilations = dense<[{dilation_h}, {dilation_w}]> : vector<2xi64>, strides = dense<[{stride_h}, {stride_w}]> : vector<2xi64>}}
        ins(%input, %filter : tensor<{batch}x{effective_h}x{effective_w}x{channels}xf32>, tensor<{filter_h}x{filter_w}x{channels}x{num_filters}xf32>)
        outs(%filled : tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32>) -> tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32>
    return %result : tensor<{batch}x{ofmap_h}x{ofmap_w}x{num_filters}xf32>
  }}"""
        )

    return f"""// Auto-generated linalg bridge from IREE ONNX-imported torch.operator Conv ops.
// The bridge is shape-only: NCHW/OIHW ONNX Conv shapes are emitted as NHWC/HWCF linalg conv ops.

module {{

{chr(10).join(funcs)}

}}
"""


def parse_linalg_ops(text: str, kind: Kind) -> list[LinalgOp]:
    if kind == "matmul":
        ops = parse_matmul_ops(text)
    elif kind == "conv2d":
        ops = parse_conv2d_ops(text)
    else:
        raise ValueError(f"unsupported kind: {kind}")
    if not ops:
        raise ValueError(f"no supported linalg.{kind} ops found")
    return ops
