"""Generic user-provided MLIR tiling CLI helpers."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tools.iree_tile_input import (
    TileConfig,
    _parse_tile_sizes,
    attach_transform_schedule,
    render_transform_schedule,
)


def test_parse_tile_sizes_accepts_commas_and_x_separator():
    assert _parse_tile_sizes("0,8,8,96,0,0,0") == (0, 8, 8, 96, 0, 0, 0)
    assert _parse_tile_sizes("8x16x32") == (8, 16, 32)


def test_parse_tile_sizes_rejects_all_zero():
    with pytest.raises(ValueError, match="at least one"):
        _parse_tile_sizes("0,0,0")


def test_render_transform_schedule_for_conv_preset_shape():
    text = render_transform_schedule(
        TileConfig(
            ops=("linalg.conv_2d_nhwc_hwcf",),
            tile_sizes=(0, 8, 8, 96, 0, 0, 0),
            num_loops=3,
        )
    )
    assert 'ops{["linalg.conv_2d_nhwc_hwcf"]}' in text
    assert "tile_sizes [0, 8, 8, 96, 0, 0, 0]" in text
    assert "%loops:3" in text


def test_attach_transform_schedule_adds_module_attr_and_sequence():
    payload = """module {
  func.func @main(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8x8xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%filled : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
"""
    text = attach_transform_schedule(
        payload,
        TileConfig(ops=("linalg.matmul",), tile_sizes=(4, 4, 4), num_loops=3),
    )
    assert "module attributes { transform.with_named_sequence }" in text
    assert "transform.named_sequence @__transform_main" in text
    assert "tile_sizes [4, 4, 4]" in text


def test_cli_writes_scheduled_input_without_running_iree(tmp_path):
    payload = tmp_path / "matmul.mlir"
    payload.write_text(
        """module {
  func.func @main(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8x8xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%filled : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
"""
    )
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.iree_tile_input",
            "--input",
            str(payload),
            "--preset",
            "matmul",
            "--tile-m",
            "4",
            "--tile-n",
            "4",
            "--tile-k",
            "4",
            "--out-dir",
            str(out_dir),
            "--no-run",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"iree_tile_input failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    scheduled = out_dir / "input_with_schedule.mlir"
    assert scheduled.exists()
    assert "tile_sizes [4, 4, 4]" in scheduled.read_text()
