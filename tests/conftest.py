"""Shared fixtures: paths and golden artifact reads for Phase 3 tests."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def tiled_mlir_text() -> str:
    return (REPO / "mlir" / "tiled_outputs" / "matmul_tiled.mlir").read_text()


@pytest.fixture(scope="session")
def golden_gemm_csv_text() -> str:
    return (
        REPO / "SCALE-Sim" / "topologies" / "walking_skeleton"
        / "matmul_tiled_8x8x8_full.csv"
    ).read_text()
