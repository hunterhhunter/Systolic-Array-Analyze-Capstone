"""Custom MNK IREE tiling CLI helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.iree_tile_mnk import (
    MnkTileSpec,
    render_merged_mlir,
    tiles_to_gemm_csv_from_spec,
    validate_spec,
)
from tools.mlir2scalesim import parse_tiled_matmul


def _iree_opt_available() -> bool:
    if os.environ.get("IREE_OPT"):
        return Path(os.environ["IREE_OPT"]).exists()
    repo_venv_tool = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "iree-opt"
    return repo_venv_tool.exists() or bool(shutil.which("iree-opt"))


def test_render_merged_mlir_uses_custom_mnk_and_tile():
    text = render_merged_mlir(
        MnkTileSpec(m=64, n=32, k=128, tile_m=16, tile_n=8, tile_k=32)
    )
    assert "tensor<64x128xf32>" in text
    assert "tensor<128x32xf32>" in text
    assert "tensor<64x32xf32>" in text
    assert "tile_sizes [16, 8, 32]" in text
    assert "Expected full tiles: 64" in text


def test_non_divisible_edge_tiles_emit_smaller_rows():
    spec = MnkTileSpec(m=30, n=18, k=10, tile_m=8, tile_n=8, tile_k=4)
    validate_spec(spec)
    assert spec.expected_tiles == 36
    rows = tiles_to_gemm_csv_from_spec(spec).splitlines()
    assert len(rows) == 37
    assert rows[1] == "Tile_000, 8, 8, 4,"
    assert rows[-1] == "Tile_035, 6, 2, 2,"


def test_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="must be positive"):
        validate_spec(MnkTileSpec(m=32, n=0, k=64, tile_m=8, tile_n=8, tile_k=8))


def test_cli_generates_tiled_mlir_and_topology(tmp_path):
    if not _iree_opt_available():
        pytest.skip("IREE_OPT not available")

    out_dir = tmp_path / "mnk"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.iree_tile_mnk",
            "--m",
            "32",
            "--n",
            "32",
            "--k",
            "64",
            "--tile-m",
            "8",
            "--tile-n",
            "8",
            "--tile-k",
            "8",
            "--out-dir",
            str(out_dir),
            "--emit-topology",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"iree_tile_mnk failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    tiled = out_dir / "tiled.mlir"
    topology = out_dir / "topology.csv"
    assert (out_dir / "input_with_schedule.mlir").exists()
    assert tiled.exists()
    assert topology.exists()
    assert parse_tiled_matmul(tiled.read_text()).total_tiles == 128
    assert len(topology.read_text().splitlines()) == 129
