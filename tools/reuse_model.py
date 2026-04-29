"""Inter-tile reuse analytic correction (master_plan §3-4 #4, §9 #5, §10 #3).

Phase 3 초안: GEMM 2D / WS만 placeholder 수식.
- 첫 타일은 IFMAP+Filter prefetch를 100% 부담.
- 후속 타일은 인접 row/column 타일과의 공유 footprint(~50%)만큼 prefetch가 fold됨.
실제 보정 계수는 Phase 6에서 측정 데이터로 fit. 본 모듈은 인터페이스를
Phase 4+가 의지할 수 있도록 먼저 잡는 것이 목적.

Conv 2D 확장은 master_plan §10 #3 결정대로 Phase 6 초반.
"""

from __future__ import annotations

from typing import Iterable

from tools.aggregator import TileRecord


_FOLD_FRACTION_WS_DEFAULT = 0.5  # placeholder until Phase 6 fit


def reuse_aware_cycles(
    tile_records: Iterable[TileRecord],
    dataflow: str = "ws",
    fold_fraction: float = _FOLD_FRACTION_WS_DEFAULT,
) -> list[float]:
    """Per-tile reuse-aware cycle estimate.

    cycles_i =
        compute_i                                    if i == 0
        compute_i + (1 - fold_fraction) * prefetch_i otherwise
    where prefetch_i = total_incl_prefetch_i - compute_i.

    Other dataflows (OS/IS) are not modeled yet: pass through total_incl_prefetch.
    """
    if not 0.0 <= fold_fraction <= 1.0:
        raise ValueError(f"fold_fraction must be in [0, 1], got {fold_fraction}")

    recs = list(tile_records)
    if not recs:
        return []

    if dataflow.lower() != "ws":
        return [float(r.total_cycles_incl_prefetch) for r in recs]

    out: list[float] = []
    for i, r in enumerate(recs):
        prefetch = r.total_cycles_incl_prefetch - r.total_cycles_compute
        if i == 0:
            out.append(float(r.total_cycles_incl_prefetch))
        else:
            out.append(float(r.total_cycles_compute) + (1.0 - fold_fraction) * prefetch)
    return out
