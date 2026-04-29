"""Public linalg MLIR experiment flow tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tools.emit_scalesim_topology import emit_topology
from tools.linalg_mlir import parse_linalg_ops, parse_torch_onnx_conv_ops

REPO = Path(__file__).resolve().parent.parent


MATMUL_MLIR = """module {
  func.func @edge_matmul(%A: tensor<30x10xf32>, %B: tensor<10x18xf32>) -> tensor<30x18xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<30x18xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<30x18xf32>) -> tensor<30x18xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<30x10xf32>, tensor<10x18xf32>)
        outs(%filled : tensor<30x18xf32>) -> tensor<30x18xf32>
    return %result : tensor<30x18xf32>
  }
}
"""


CONV_MLIR = """module {
  func.func @edge_conv(
      %input: tensor<1x7x7x3xf32>,
      %filter: tensor<3x3x3x10xf32>) -> tensor<1x5x5x10xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x5x5x10xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x5x5x10xf32>) -> tensor<1x5x5x10xf32>
    %result = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<[1, 1]> : vector<2xi64>, strides = dense<[1, 1]> : vector<2xi64>}
        ins(%input, %filter : tensor<1x7x7x3xf32>, tensor<3x3x3x10xf32>)
        outs(%filled : tensor<1x5x5x10xf32>) -> tensor<1x5x5x10xf32>
    return %result : tensor<1x5x5x10xf32>
  }
}
"""

TORCH_ONNX_CONV_MLIR = """module {
  func.func @main_graph(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,64,112,112],f32> {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0> : tensor<64x3x7x7xf32>} : () -> !torch.vtensor<[64,3,7,7],f32>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0> : tensor<64xf32>} : () -> !torch.vtensor<[64],f32>
    %2 = torch.operator "onnx.Conv"(%arg0, %0, %1) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [7 : si64, 7 : si64], torch.onnx.pads = [3 : si64, 3 : si64, 3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>) -> !torch.vtensor<[1,64,112,112],f32>
    return %2 : !torch.vtensor<[1,64,112,112],f32>
  }
}
"""


def _iree_opt_available() -> bool:
    if os.environ.get("IREE_OPT"):
        return Path(os.environ["IREE_OPT"]).exists()
    repo_venv_tool = REPO / ".venv" / "bin" / "iree-opt"
    return repo_venv_tool.exists() or bool(shutil.which("iree-opt"))


def _scalesim_available() -> bool:
    try:
        import scalesim  # noqa: F401
    except ImportError:
        return False
    return True


def test_static_linalg_parser_finds_matmul_and_conv_shapes():
    matmul = parse_linalg_ops(MATMUL_MLIR, "matmul")[0]
    assert matmul.name == "edge_matmul"
    assert matmul.shape == {"m": 30, "n": 18, "k": 10}

    conv = parse_linalg_ops(CONV_MLIR, "conv2d")[0]
    assert conv.name == "edge_conv"
    assert conv.shape["ofmap_h"] == 5
    assert conv.shape["num_filters"] == 10
    assert conv.shape["stride_h"] == 1


def test_torch_onnx_conv_parser_ignores_si64_type_suffixes():
    conv = parse_torch_onnx_conv_ops(TORCH_ONNX_CONV_MLIR)[0]
    assert conv.name == "main_graph_onnx_conv_000"
    assert conv.shape["stride_h"] == 2
    assert conv.shape["stride_w"] == 2
    assert conv.shape["dilation_h"] == 1
    assert conv.shape["pad_right"] == 3


def test_tile_mlir_cli_writes_stable_manifest_without_iree(tmp_path):
    mlir = tmp_path / "edge_matmul.mlir"
    mlir.write_text(MATMUL_MLIR)
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.tile_mlir",
            "--input",
            str(mlir),
            "--kind",
            "matmul",
            "--tile-m",
            "8",
            "--tile-n",
            "8",
            "--tile-k",
            "4",
            "--run-dir",
            str(run_dir),
            "--no-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"tile_mlir failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    manifest = json.loads((run_dir / "tile_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "matmul"
    assert manifest["tile"] == {"m": 8, "n": 8, "k": 4}
    assert manifest["ops"][0]["shape"] == {"m": 30, "n": 18, "k": 10}
    assert (run_dir / "input_with_schedule.mlir").exists()
    assert manifest["tiled_mlir"] is None


def test_emit_topology_handles_matmul_edge_tiles():
    manifest = {
        "kind": "matmul",
        "tile": {"m": 8, "n": 8, "k": 4},
        "ops": [
            {
                "name": "edge_matmul",
                "kind": "matmul",
                "shape": {"m": 30, "n": 18, "k": 10},
            }
        ],
    }
    rows = emit_topology(manifest).splitlines()
    assert len(rows) == 37
    assert rows[1] == "edge_matmul_m0000_n0000_k0000_000, 8, 8, 4,"
    assert rows[-1] == "edge_matmul_m0024_n0016_k0008_035, 6, 2, 2,"


def test_emit_topology_handles_conv2d_edge_tiles():
    manifest = {
        "kind": "conv2d",
        "tile": {"oh": 3, "ow": 4, "oc": 6},
        "ops": [
            {
                "name": "edge_conv",
                "kind": "conv2d",
                "shape": parse_linalg_ops(CONV_MLIR, "conv2d")[0].shape,
            }
        ],
    }
    rows = emit_topology(manifest).splitlines()
    assert len(rows) == 9
    assert rows[1] == "edge_conv_oh000_ow000_oc0000, 5, 6, 3, 3, 3, 6, 1,"
    assert rows[-1] == "edge_conv_oh003_ow004_oc0006, 4, 3, 3, 3, 3, 4, 1,"


def test_public_conv2d_tile_to_topology_smoke(tmp_path):
    if not _iree_opt_available():
        pytest.skip("IREE_OPT not available")

    mlir = tmp_path / "conv.mlir"
    mlir.write_text(CONV_MLIR)
    run_dir = tmp_path / "conv_run"
    commands = [
        [
            sys.executable,
            "-m",
            "tools.tile_mlir",
            "--input",
            str(mlir),
            "--kind",
            "conv2d",
            "--tile-oh",
            "3",
            "--tile-ow",
            "4",
            "--tile-oc",
            "6",
            "--run-dir",
            str(run_dir),
        ],
        [
            sys.executable,
            "-m",
            "tools.emit_scalesim_topology",
            "--manifest",
            str(run_dir / "tile_manifest.json"),
            "--output",
            str(run_dir / "topology.csv"),
        ],
    ]
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"{cmd} failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    assert (run_dir / "tiled.mlir").exists()
    assert len((run_dir / "topology.csv").read_text().splitlines()) == 9


def test_public_matmul_flow_smoke(tmp_path):
    if not _iree_opt_available():
        pytest.skip("IREE_OPT not available")
    if not _scalesim_available():
        pytest.skip("scalesim package not installed")

    mlir = tmp_path / "matmul.mlir"
    mlir.write_text(MATMUL_MLIR)
    run_dir = tmp_path / "run"
    results = tmp_path / "result.parquet"

    commands = [
        [
            sys.executable,
            "-m",
            "tools.tile_mlir",
            "--input",
            str(mlir),
            "--kind",
            "matmul",
            "--tile-m",
            "8",
            "--tile-n",
            "8",
            "--tile-k",
            "4",
            "--run-dir",
            str(run_dir),
        ],
        [
            sys.executable,
            "-m",
            "tools.emit_scalesim_topology",
            "--manifest",
            str(run_dir / "tile_manifest.json"),
            "--output",
            str(run_dir / "topology.csv"),
        ],
        [
            sys.executable,
            "-m",
            "tools.run_experiment",
            "--run-dir",
            str(run_dir),
            "--kind",
            "matmul",
            "--arch-cfg",
            "SCALE-Sim/configs/tpuv2.cfg",
            "--results",
            str(results),
        ],
    ]
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"{cmd} failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    df = pd.read_parquet(results)
    assert df.iloc[0]["status"] == "ok"
    assert df.iloc[0]["n_tiles"] == 36
    assert df.iloc[0]["compute_cycles"] > 0
    assert results.with_suffix(".csv").exists()
