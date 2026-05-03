"""Standalone report aggregation/reuse CLIs."""

from tools.aggregator import summarize_compute_report
from tools.reuse_runner import build_reuse_dataframe


def test_summarize_compute_report_is_topology_agnostic(walking_compute_report):
    row = summarize_compute_report(
        walking_compute_report,
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


def test_reuse_runner_builds_per_tile_dataframe(walking_compute_report):
    df = build_reuse_dataframe(walking_compute_report, dataflow="ws")
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