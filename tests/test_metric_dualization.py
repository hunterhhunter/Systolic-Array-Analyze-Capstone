"""Dual-metric aggregate 분리 검증 (master_plan §2 핵심 설계, L5).

Phase 6 figure는 compute / total / reuse_aware 3 컬럼을 병기한다.
aggregator.aggregate(kind=...) 가 같은 입력에 대해 두 metric을 절대 섞지 않는지
walking_skeleton 합산 oracle (3,712 / 13,056) 로 lock.
"""

import pytest

from tools.aggregator import aggregate, parse_compute_report, reuse_correction


@pytest.fixture(scope="module")
def walking_records(walking_compute_report):
    return parse_compute_report(walking_compute_report)

def test_walking_compute_sum_is_3712(walking_records):
    out = aggregate(walking_records, metric_kind="compute")
    assert out["sum_cycles"] == 3712


def test_walking_total_sum_is_13056(walking_records):
    out = aggregate(walking_records, metric_kind="total")
    assert out["sum_cycles"] == 13056


def test_compute_and_total_must_differ(walking_records):
    c = aggregate(walking_records, metric_kind="compute")
    t = aggregate(walking_records, metric_kind="total")
    assert c["sum_cycles"] != t["sum_cycles"]
    # prefetch only adds, never subtracts
    assert t["sum_cycles"] > c["sum_cycles"]


def test_total_minus_compute_is_walking_prefetch_overhead(walking_records):
    """Plan L5 anchor: 13056 - 3712 = 9344 → /128 tiles = 73 cycles/tile prefetch."""
    c = aggregate(walking_records, metric_kind="compute")
    t = aggregate(walking_records, metric_kind="total")
    per_tile_overhead = (t["sum_cycles"] - c["sum_cycles"]) / c["n_tiles"]
    assert per_tile_overhead == 73.0


def test_unknown_metric_kind_raises(walking_records):
    with pytest.raises(ValueError):
        aggregate(walking_records, metric_kind="bogus")  # type: ignore[arg-type]


def test_empty_input_raises():
    with pytest.raises(ValueError):
        aggregate([], metric_kind="compute")


def test_n_tiles_and_means(walking_records):
    out = aggregate(walking_records, metric_kind="compute")
    assert out["n_tiles"] == 128
    assert out["sum_stall"] == 0
    assert out["mean_mapping_eff_pct"] == 100.0


def test_reuse_correction_returns_per_tile_floats(walking_records):
    corrected = reuse_correction(walking_records, dataflow="ws")
    assert len(corrected) == 128
    assert all(isinstance(x, float) for x in corrected)
    # First tile pays full prefetch (102), subsequent tiles fold half (29 + 0.5*73 = 65.5).
    assert corrected[0] == 102.0
    assert corrected[1] == pytest.approx(65.5)
