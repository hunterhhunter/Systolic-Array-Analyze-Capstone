"""Standalone report aggregation/reuse CLIs."""

from pathlib import Path

from tools.aggregator import summarize_compute_report
from tools.reuse_runner import build_reuse_dataframe

REPO = Path(__file__).resolve().parent.parent
WALKING_REPORT = (
    REPO / "outputs" / "walking_skeleton_full" / "walking_8x8_ws" / "COMPUTE_REPORT.csv"
)


def test_summarize_compute_report_is_topology_agnostic():
    row = summarize_compute_report(
        WALKING_REPORT,
        dataflow="ws",
        workload="walking",
        arch="8x8_ws",
    )
    assert row["workload"] == "walking"
    assert row["arch"] == "8x8_ws"
    assert row["n_tiles"] == 128
    assert row["compute_cycles"] == 3712
    assert row["total_cycles"] == 13056
    assert row["stall"] == 0


def test_reuse_runner_builds_per_tile_dataframe():
    df = build_reuse_dataframe(WALKING_REPORT, dataflow="ws")
    assert len(df) == 128
    assert list(df.columns) == [
        "layer_id",
        "compute_cycles",
        "total_cycles",
        "prefetch_cycles",
        "reuse_aware_cycles",
        "overall_util_pct",
        "mapping_eff_pct",
        "compute_util_pct",
    ]
    assert df.iloc[0]["reuse_aware_cycles"] == 102.0
    assert df.iloc[1]["reuse_aware_cycles"] == 65.5
