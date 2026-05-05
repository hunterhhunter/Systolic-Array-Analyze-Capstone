"""Unit tests for tiled MLIR loop-count handling."""

from tools.mlir2scalesim import _ceil_loop_count


def test_ceil_loop_count_counts_edge_tile_iteration():
    assert _ceil_loop_count(0, 10, 4) == 3
    assert _ceil_loop_count(0, 32, 8) == 4
    assert _ceil_loop_count(8, 8, 4) == 0


def test_ceil_loop_count_rejects_non_positive_step():
    try:
        _ceil_loop_count(0, 10, 0)
    except ValueError as exc:
        assert "non-positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
