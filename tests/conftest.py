"""Shared fixtures: paths and golden artifact reads for tests."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


def find_tiled_mlir() -> Path:
    candidates = [
        # Legacy golden path
        REPO / "mlir" / "tiled_outputs" / "matmul_tiled.mlir",

        # Current demo output path
        REPO
        / "outputs"
        / "demo"
        / "matmul_m32_n32_k64"
        / "8x8_ws"
        / "ws"
        / "8x8x8"
        / "tiled.mlir",

        # Current sweep output path
        REPO
        / "outputs"
        / "small_sweep"
        / "matmul_m32_n32_k64"
        / "8x8_ws"
        / "ws"
        / "8x8x8"
        / "tiled.mlir",

        # Manual experiment output path
        REPO
        / "outputs"
        / "experiments"
        / "matmul_8x8x8"
        / "tiled.mlir",

        REPO
        / "outputs"
        / "experiments"
        / "matmul_small_matmul"
        / "tiled.mlir",
    ]

    for path in candidates:
        if path.exists():
            return path

    pytest.skip(
        "tiled.mlir not found. Run `make demo`, `make sweep`, "
        "or `make tile RUN_DIR=outputs/experiments/matmul_8x8x8` first."
    )


def find_walking_compute_report() -> Path:
    candidates = [
        # Legacy golden path
        REPO
        / "outputs"
        / "walking_skeleton_full"
        / "walking_8x8_ws"
        / "COMPUTE_REPORT.csv",

        # Current demo output path
        REPO
        / "outputs"
        / "demo"
        / "matmul_m32_n32_k64"
        / "8x8_ws"
        / "ws"
        / "8x8x8"
        / "sim"
        / "walking_8x8_ws"
        / "COMPUTE_REPORT.csv",

        # Current sweep output path
        REPO
        / "outputs"
        / "small_sweep"
        / "matmul_m32_n32_k64"
        / "8x8_ws"
        / "ws"
        / "8x8x8"
        / "sim"
        / "walking_8x8_ws"
        / "COMPUTE_REPORT.csv",

        # Manual experiment output paths
        REPO
        / "outputs"
        / "experiments"
        / "matmul_8x8x8"
        / "sim"
        / "scale_example_TPUv2"
        / "COMPUTE_REPORT.csv",

        REPO
        / "outputs"
        / "experiments"
        / "matmul_small_matmul"
        / "sim"
        / "scale_example_TPUv2"
        / "COMPUTE_REPORT.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    pytest.skip(
        "COMPUTE_REPORT.csv not found. Run `make demo`, `make sweep`, "
        "or `make experiment` first."
    )


@pytest.fixture(scope="session")
def tiled_mlir_text() -> str:
    return find_tiled_mlir().read_text()


@pytest.fixture(scope="session")
def walking_compute_report() -> Path:
    return find_walking_compute_report()


@pytest.fixture(scope="session")
def golden_gemm_csv_text() -> str:
    return (
        REPO
        / "SCALE-Sim"
        / "topologies"
        / "walking_skeleton"
        / "matmul_tiled_8x8x8_full.csv"
    ).read_text()