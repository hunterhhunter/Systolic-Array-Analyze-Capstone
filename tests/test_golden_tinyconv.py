"""TinyConv 손계산 vs SCALE-Sim 실측 oracle (plan/scalesim_plan.md §2-3, §4).

손계산: cycles_per_tile = K_gemm + ArrayH - 1 = 14, total = 2 × 1 × 14 = 28 cycles.
실측 (committed): outputs/tiny_conv/tiny_conv_3x3/COMPUTE_REPORT.csv → 31 cycles compute.
허용오차 ±5 (pipeline warm-up/cool-down 반영). master_plan §7 Must-have 손계산
3자 검증 항목의 anchor.

본 테스트는 SCALE-Sim 재실행 없이 commit된 보고서를 직접 oracle로 사용한다.
실행 자동화는 Phase 4 sweep_runner에서 통합.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

from tools.aggregator import parse_compute_report

REPO = Path(__file__).resolve().parent.parent
TINYCONV_REPORT = REPO / "outputs" / "tiny_conv" / "tiny_conv_3x3" / "COMPUTE_REPORT.csv"

ORACLE_COMPUTE_CYCLES = 28
TOLERANCE = 5


@pytest.fixture(scope="module")
def tinyconv_record():
    if not TINYCONV_REPORT.exists():
        pytest.skip(
            f"{TINYCONV_REPORT} not present; regenerate via scalesim_plan §3-3"
        )
    recs = parse_compute_report(TINYCONV_REPORT)
    assert len(recs) == 1, f"expected single TinyConv layer, got {len(recs)}"
    return recs[0]


def test_compute_cycles_within_tolerance(tinyconv_record):
    diff = abs(tinyconv_record.total_cycles_compute - ORACLE_COMPUTE_CYCLES)
    assert diff <= TOLERANCE, (
        f"compute cycles {tinyconv_record.total_cycles_compute} "
        f"deviates from oracle {ORACLE_COMPUTE_CYCLES} by {diff} (tol ±{TOLERANCE})"
    )


def test_no_stall_on_in_sram_workload(tinyconv_record):
    assert tinyconv_record.stall_cycles == 0


def test_prefetch_overhead_dominates_compute(tinyconv_record):
    """scalesim_plan §4: 99/31 ≈ 3.19× overhead ratio."""
    ratio = (
        tinyconv_record.total_cycles_incl_prefetch
        / tinyconv_record.total_cycles_compute
    )
    assert ratio > 1.5


def test_compute_util_le_mapping_eff(tinyconv_record):
    """scalesim_plan §4 invariant: compute_util ≤ mapping_eff (PE 유휴 fill 사이클)."""
    assert tinyconv_record.compute_util_pct <= tinyconv_record.mapping_eff_pct + 1e-6


def test_mapping_eff_matches_cycle_weighted_44pct(tinyconv_record):
    """scalesim_plan §4 교훈 #2: mapping_eff = cycle-weighted PE 활용 평균.
    TinyConv 두 M-tile 평균 = (66.67 + 22.22) / 2 = 44.44%.
    """
    assert tinyconv_record.mapping_eff_pct == pytest.approx(44.4444, abs=1e-2)
