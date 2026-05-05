"""Shared fixtures: paths and golden artifact reads for tests."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _walking_paths(base: Path, artifact: str) -> list[Path]:
    """Return current and legacy walking-skeleton artifact paths."""
    suffix = [
        "matmul_m32_n32_k64",
        "default",
        "walking_8x8_ws",
        "8x8_ws",
        "ws",
        "test",
        "8x8x8",
    ]
    legacy_suffix = [
        "matmul_m32_n32_k64",
        "default",
        "8x8_ws",
        "ws",
        "test",
        "8x8x8",
    ]
    if artifact == "report":
        tail = ["sim", "walking_8x8_ws", "COMPUTE_REPORT.csv"]
    elif artifact == "tiled":
        tail = ["tiled.mlir"]
    else:
        raise ValueError(artifact)
    return [
        base.joinpath(*suffix, *tail),
        base.joinpath(*legacy_suffix, *tail),
    ]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


def find_tiled_mlir() -> Path:
    candidates = [
        # Legacy golden path
        REPO / "mlir" / "tiled_outputs" / "matmul_tiled.mlir",
        *_walking_paths(REPO / "outputs" / "demo", "tiled"),
        *_walking_paths(REPO / "outputs" / "small_sweep", "tiled"),
        # Manual experiment output paths
        REPO / "outputs" / "experiments" / "matmul_8x8x8" / "tiled.mlir",
        REPO / "outputs" / "experiments" / "matmul_small_matmul" / "tiled.mlir",
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
        REPO / "outputs" / "walking_skeleton_full" / "walking_8x8_ws" / "COMPUTE_REPORT.csv",
        *_walking_paths(REPO / "outputs" / "demo", "report"),
        *_walking_paths(REPO / "outputs" / "small_sweep", "report"),
        # Intentionally do not fall back to legacy outputs/experiments paths here.
        # Those reports may have been generated with different SCALE-Sim/topology
        # settings and do not match the walking-skeleton golden invariants.
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
    return find_tiled_mlir().read_text(encoding="utf-8")


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
    ).read_text(encoding="utf-8")
