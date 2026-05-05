"""Phase 4 small sweep runner regression tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import sweep_runner
from tools.result_schema import SWEEP_COLUMNS

REPO = Path(__file__).resolve().parent.parent
RESULTS_PARQUET = REPO / "results" / "results.parquet"

EXPECTED_COLUMNS = list(SWEEP_COLUMNS)



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


def test_default_configs_are_the_phase4_small_grid():
    assert [cfg.tile for cfg in sweep_runner.DEFAULT_CONFIGS] == ["8x8x8", "16x16x16"]
    assert {cfg.workload for cfg in sweep_runner.DEFAULT_CONFIGS} == {
        "matmul_m32_n32_k64"
    }
    assert {cfg.arch for cfg in sweep_runner.DEFAULT_CONFIGS} == {"8x8_ws"}
    assert {cfg.dataflow for cfg in sweep_runner.DEFAULT_CONFIGS} == {"ws"}
    assert {cfg.layout for cfg in sweep_runner.DEFAULT_CONFIGS} == {"test"}


def test_empty_failure_row_preserves_schema():
    row = sweep_runner._empty_row(sweep_runner.DEFAULT_CONFIGS[0], "iree_fail")
    assert list(row) == EXPECTED_COLUMNS
    assert row["status"] == "iree_fail"
    assert row["n_tiles"] == 0
    assert row["compute_cycles"] == 0
    assert row["total_cycles"] == 0
    assert row["reuse_aware_cycles"] == 0.0
    assert row["reuse_aware_cycles_est"] == 0.0
    assert row["reuse_model"] == ""
    assert row["reuse_fold_fraction"] == 0.0
    assert row["stall"] == 0
    assert row["logical_tiles"] == 0
    assert row["simulated_tiles"] == 0
    assert row["raw_topology_rows"] == 0
    assert row["topology_mode"] == ""
    assert row["cache_key"] == ""
    assert row["cache_status"] == ""


@pytest.fixture(scope="module")
def sweep_parquet() -> Path:
    if not _iree_opt_available():
        pytest.skip("IREE_OPT not available")
    if not _scalesim_available():
        pytest.skip("scalesim package not installed (`make env`)")
    if RESULTS_PARQUET.exists():
        RESULTS_PARQUET.unlink()
    proc = subprocess.run(
        [sys.executable, "-m", "tools.sweep_runner"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"`tools.sweep_runner` exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    assert RESULTS_PARQUET.exists(), "results.parquet not written"
    return RESULTS_PARQUET


def test_sweep_parquet_columns_match_schema(sweep_parquet):
    import pyarrow.parquet as pq

    table = pq.read_table(sweep_parquet)
    assert list(table.column_names) == EXPECTED_COLUMNS


def test_sweep_has_two_ok_tile_rows(sweep_parquet):
    import pyarrow.parquet as pq

    df = pq.read_table(sweep_parquet).to_pandas()
    assert len(df) == 2
    assert set(df["tile"]) == {"8x8x8", "16x16x16"}
    assert set(df["status"]) == {"ok"}


def test_sweep_8x8x8_row_matches_walking_baseline(sweep_parquet):
    import pyarrow.parquet as pq

    df = pq.read_table(sweep_parquet).to_pandas()
    row = df[df["tile"] == "8x8x8"].iloc[0]
    assert row["n_tiles"] == 128
    assert row["compute_cycles"] == 3712
    assert row["total_cycles"] == 13056
    assert row["mean_mapping_eff_pct"] == 100.0


def test_sweep_total_exposes_prefetch_overhead(sweep_parquet):
    import pyarrow.parquet as pq

    df = pq.read_table(sweep_parquet).to_pandas()
    assert (df["total_cycles"] > df["compute_cycles"]).all()
    assert (df["reuse_aware_cycles"] < df["total_cycles"]).all()



def test_run_dir_uses_arch_run_name_to_avoid_config_collisions(tmp_path):
    cfg = sweep_runner.DEFAULT_CONFIGS[0]
    run_dir = sweep_runner.run_dir_for_config(cfg, root=tmp_path)
    assert cfg.arch_run_name in run_dir.parts
    assert cfg.arch in run_dir.parts
