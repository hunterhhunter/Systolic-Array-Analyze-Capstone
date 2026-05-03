"""COMPUTE_REPORT.csv parsing regression (master_plan L4).

Locks: column-1=prefetch+, column-2=compute-only, Util fields are floats.
ex02 walking_skeleton 128-row 보고서를 oracle로 사용해 schema drift 즉시 포착.
"""

from pathlib import Path

import pytest

from tools.aggregator import TileRecord, parse_compute_report


GOLDEN_HEADER = (
    "LayerID, Total Cycles (incl. prefetch), Total Cycles, "
    "Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,\n"
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "COMPUTE_REPORT.csv"
    p.write_text(GOLDEN_HEADER + body)
    return p


def test_parses_single_walking_row(tmp_path):
    p = _write(tmp_path, "0, 102, 29, 0, 27.5862, 100.0, 21.6216,\n")
    recs = parse_compute_report(p)
    assert len(recs) == 1
    r = recs[0]
    assert r.layer_id == 0
    assert r.total_cycles_incl_prefetch == 102
    assert r.total_cycles_compute == 29
    assert r.stall_cycles == 0
    assert r.mapping_eff_pct == 100.0


def test_util_fields_are_floats(tmp_path):
    p = _write(tmp_path, "0, 99, 31, 0, 34.4086, 44.4444, 33.3333,\n")
    r = parse_compute_report(p)[0]
    assert isinstance(r.overall_util_pct, float)
    assert isinstance(r.mapping_eff_pct, float)
    assert isinstance(r.compute_util_pct, float)
    assert r.overall_util_pct == pytest.approx(34.4086, abs=1e-4)
    assert r.compute_util_pct == pytest.approx(33.3333, abs=1e-4)


def test_column1_is_prefetch_plus_column2_is_compute(tmp_path):
    """L4 invariant: prefetch+ ≥ compute always."""
    p = _write(tmp_path, "0, 102, 29, 0, 27.5862, 100.0, 21.6216,\n")
    r = parse_compute_report(p)[0]
    assert r.total_cycles_incl_prefetch >= r.total_cycles_compute
    assert r.total_cycles_incl_prefetch - r.total_cycles_compute == 73


def test_walking_skeleton_full_parses_128_rows(walking_compute_report):
    recs = parse_compute_report(walking_compute_report)
    assert len(recs) == 128
    assert all(r.total_cycles_incl_prefetch == 102 for r in recs)
    assert all(r.total_cycles_compute == 29 for r in recs)
    assert all(r.mapping_eff_pct == 100.0 for r in recs)


def test_rejects_unexpected_header(tmp_path):
    p = tmp_path / "COMPUTE_REPORT.csv"
    p.write_text("garbage,header,row,\n0,1,2,\n")
    with pytest.raises(ValueError):
        parse_compute_report(p)


def test_rejects_short_row(tmp_path):
    p = _write(tmp_path, "0, 102, 29,\n")
    with pytest.raises(ValueError):
        parse_compute_report(p)


def test_skips_blank_rows(tmp_path):
    body = "0, 102, 29, 0, 27.5, 100.0, 21.6,\n\n1, 102, 29, 0, 27.5, 100.0, 21.6,\n"
    p = _write(tmp_path, body)
    assert len(parse_compute_report(p)) == 2


def test_record_is_namedtuple_with_typed_fields():
    r = TileRecord(0, 102, 29, 0, 27.5862, 100.0, 21.6216)
    assert r.layer_id == 0
    assert r.total_cycles_incl_prefetch == 102
    assert r.total_cycles_compute == 29
