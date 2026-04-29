"""make demo / tools.demo end-to-end smoke (master_plan §4 Phase 3 #6).

mlir2scalesim → iree-opt → SCALE-Sim → aggregator → results/demo.parquet 까지
직선 호출이 끊김 없이 results/demo.parquet 을 정확한 스키마로 생성하는지 검증.
sweep_runner / joblib / cache는 Phase 4 — 본 테스트의 책임 범위 밖.

환경 의존: IREE_OPT 바이너리 + SCALE-Sim editable 설치. 둘 다 없으면 skip.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest


REPO = Path(__file__).resolve().parent.parent
RESULTS_PARQUET = REPO / "results" / "demo.parquet"

EXPECTED_COLUMNS = [
    "workload",
    "m",
    "n",
    "k",
    "tile",
    "tile_m",
    "tile_n",
    "tile_k",
    "arch",
    "arch_cfg",
    "dataflow",
    "n_tiles",
    "compute_cycles",
    "total_cycles",
    "reuse_aware_cycles",
    "mean_overall_util_pct",
    "mean_mapping_eff_pct",
    "mean_compute_util_pct",
    "stall",
    "status",
]
EXPECTED_TILES = {"8x8x8", "16x16x16"}


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


@pytest.fixture(scope="module")
def demo_parquet() -> Path:
    if not _iree_opt_available():
        pytest.skip("IREE_OPT not available")
    if not _scalesim_available():
        pytest.skip("scalesim package not installed (`make env`)")
    if RESULTS_PARQUET.exists():
        RESULTS_PARQUET.unlink()
    proc = subprocess.run(
        [sys.executable, "-m", "tools.demo"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"`tools.demo` exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    assert RESULTS_PARQUET.exists(), "demo.parquet not written"
    return RESULTS_PARQUET


def test_parquet_columns_match_schema(demo_parquet):
    table = pq.read_table(demo_parquet)
    assert list(table.column_names) == EXPECTED_COLUMNS


def test_parquet_has_two_tile_rows(demo_parquet):
    df = pq.read_table(demo_parquet).to_pandas()
    assert len(df) == 2
    assert set(df["tile"]) == EXPECTED_TILES


def test_walking_skeleton_8x8x8_row_matches_baseline(demo_parquet):
    df = pq.read_table(demo_parquet).to_pandas()
    row = df[df["tile"] == "8x8x8"].iloc[0]
    assert row["n_tiles"] == 128
    assert row["compute_cycles"] == 3712
    assert row["total_cycles"] == 13056
    assert row["mean_mapping_eff_pct"] == 100.0
    assert row["stall"] == 0
    assert row["status"] == "ok"


def test_16x16x16_row_has_fewer_tiles_higher_util(demo_parquet):
    df = pq.read_table(demo_parquet).to_pandas()
    big = df[df["tile"] == "16x16x16"].iloc[0]
    small = df[df["tile"] == "8x8x8"].iloc[0]
    # 16-tile 타일은 16개, 8-tile 타일은 128개
    assert big["n_tiles"] < small["n_tiles"]
    # 큰 타일은 array warm-up overhead가 더 적게 누적 → overall util 더 높음
    assert big["mean_overall_util_pct"] > small["mean_overall_util_pct"]


def test_total_cycles_strictly_greater_than_compute(demo_parquet):
    """L4 invariant: incl. prefetch ≥ compute (모든 row)."""
    df = pq.read_table(demo_parquet).to_pandas()
    assert (df["total_cycles"] > df["compute_cycles"]).all()
