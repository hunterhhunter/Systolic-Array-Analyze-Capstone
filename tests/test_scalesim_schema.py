"""GEMM topology CSV schema regression (plan L3).

Locks in: column order `Layer Name, M, N, K,`, trailing comma both in header and rows,
row count == total_tiles, single trailing newline. 이 포맷을 SCALE-Sim topology reader가
기대하므로 drift가 생기면 조용히 파이프라인이 오염되는 L3 버그 재발 방지.
"""

from tools.mlir2scalesim import TileShape, TiledMatmul, tiles_to_gemm_csv


def _make(total_tiles: int, tile: TileShape = TileShape(M=8, N=8, K=8)) -> TiledMatmul:
    return TiledMatmul(tile=tile, loop_counts=(total_tiles,), total_tiles=total_tiles)


def test_header_is_exact():
    csv = tiles_to_gemm_csv(_make(1))
    assert csv.splitlines()[0] == "Layer Name, M, N, K,"


def test_header_has_trailing_comma():
    csv = tiles_to_gemm_csv(_make(1))
    assert csv.splitlines()[0].endswith(",")


def test_each_row_has_trailing_comma():
    csv = tiles_to_gemm_csv(_make(5))
    for line in csv.splitlines()[1:]:
        assert line.endswith(","), f"row missing trailing comma: {line!r}"


def test_row_format_tile_name_and_dims():
    csv = tiles_to_gemm_csv(
        TiledMatmul(tile=TileShape(M=4, N=16, K=32), loop_counts=(2, 3), total_tiles=6)
    )
    lines = csv.splitlines()
    assert lines[1] == "Tile_000, 4, 16, 32,"
    assert lines[6] == "Tile_005, 4, 16, 32,"


def test_row_count_equals_total_tiles_plus_header():
    csv = tiles_to_gemm_csv(_make(128))
    assert len(csv.splitlines()) == 129  # 1 header + 128 rows


def test_trailing_newline():
    csv = tiles_to_gemm_csv(_make(1))
    assert csv.endswith("\n")


def test_tile_index_zero_padded_3_digits():
    csv = tiles_to_gemm_csv(_make(10))
    rows = csv.splitlines()[1:]
    assert rows[0].startswith("Tile_000,")
    assert rows[9].startswith("Tile_009,")
