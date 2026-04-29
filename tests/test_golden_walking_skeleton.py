"""Walking skeleton golden regression (plan §4 Phase 3 gate).

Locks the three parser paths (IR / regex / auto) against the committed golden CSV
produced during Phase 2.5. If mlir2scalesim.py changes break either parser, this
test fires immediately — the whole Phase 3+ pipeline hangs on 128 tile × 8x8x8 parity.

Scope: mlir2scalesim.py byte-exact CSV reproduction only. SCALE-Sim 실측 cycle
재현(13,056 / 3,712)은 aggregator.py 도입 후 확장.
"""

from tools.mlir2scalesim import (
    TileShape,
    _parse_via_ir,
    _parse_via_regex,
    parse_tiled_matmul,
    tiles_to_gemm_csv,
)


def test_ir_parser_reproduces_golden_csv_byte_exact(tiled_mlir_text, golden_gemm_csv_text):
    parsed = _parse_via_ir(tiled_mlir_text)
    assert tiles_to_gemm_csv(parsed) == golden_gemm_csv_text


def test_regex_parser_reproduces_golden_csv_byte_exact(tiled_mlir_text, golden_gemm_csv_text):
    parsed = _parse_via_regex(tiled_mlir_text)
    assert tiles_to_gemm_csv(parsed) == golden_gemm_csv_text


def test_auto_parser_reproduces_golden(tiled_mlir_text, golden_gemm_csv_text):
    parsed = parse_tiled_matmul(tiled_mlir_text)
    assert tiles_to_gemm_csv(parsed) == golden_gemm_csv_text


def test_tile_shape_is_8x8x8(tiled_mlir_text):
    parsed = _parse_via_ir(tiled_mlir_text)
    assert parsed.tile == TileShape(M=8, N=8, K=8)


def test_total_tiles_is_128(tiled_mlir_text):
    parsed = _parse_via_ir(tiled_mlir_text)
    assert parsed.total_tiles == 128


def test_ir_and_regex_parsers_agree(tiled_mlir_text):
    ir = tiles_to_gemm_csv(_parse_via_ir(tiled_mlir_text))
    rx = tiles_to_gemm_csv(_parse_via_regex(tiled_mlir_text))
    assert ir == rx
