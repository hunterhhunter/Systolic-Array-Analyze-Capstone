"""Boundary-case generators for MatMul SCALE-Sim sweeps.

These helpers create reproducible cases for questions such as:
- what happens when a tile is smaller/equal/larger than the systolic array?
- what happens as memory bandwidth becomes tight?
- what happens as SRAM capacity approaches the tile footprint?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

from tools.config_factory import ArchConfigParams, generated_cfg_path, write_arch_cfg
from tools.sweep_runner import (
    DEFAULT_ARCH_CFG,
    DEFAULT_LAYOUT,
    ArchSpec,
    LayoutSpec,
    MnkShape,
    SweepConfig,
    TileShape,
    load_arch_spec,
    load_layout_spec,
)


@dataclass(frozen=True)
class BoundaryCase:
    case_name: str
    mnk: MnkShape
    tile: TileShape
    arch_cfg: Path
    layout: Path


def parse_array(value: str) -> tuple[int, int]:
    raw = value.lower().replace(",", "x").split("x")
    if len(raw) != 2:
        raise ValueError("array must be formatted like 8x8")
    h, w = (int(x) for x in raw)
    if h <= 0 or w <= 0:
        raise ValueError("array dimensions must be positive")
    return h, w


def parse_tile_shape(value: str) -> TileShape:
    raw = value.lower().replace(",", "x").split("x")
    if len(raw) != 3:
        raise ValueError("tile must be formatted like 32x32x32")
    m, n, k = (int(x) for x in raw)
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("tile dimensions must be positive")
    return TileShape(m, n, k)


def tile_smaller_equal_larger(array_h: int, array_w: int, tile_k: int) -> tuple[tuple[str, TileShape], ...]:
    """Return canonical tile-vs-array boundary cases."""

    small_m = max(1, array_h // 2)
    small_n = max(1, array_w // 2)
    return (
        ("tile_smaller_than_array", TileShape(small_m, small_n, tile_k)),
        ("tile_equal_to_array", TileShape(array_h, array_w, tile_k)),
        ("tile_larger_than_array", TileShape(array_h * 2, array_w * 2, tile_k)),
    )


def default_sram_tile(mnk: MnkShape, array_h: int, array_w: int, tile_k: int) -> TileShape:
    """Pick a tile that is large enough to make 1KB SRAM visibly tight.

    The normal array-equal tile (for example 8x8x8 on an 8x8 array) is too small:
    1KB can hold all operands, so 1KB and 64KB produce identical results.  This
    default chooses a larger still-valid tile for the SRAM-only cases.
    """

    return TileShape(
        m=min(mnk.m, max(array_h * 4, array_h)),
        n=min(mnk.n, max(array_w * 4, array_w)),
        k=min(mnk.k, max(tile_k * 4, tile_k)),
    )


def estimate_tile_sram_kb(tile: TileShape, *, safety_factor: int = 2) -> tuple[int, int, int]:
    """Estimate IFMAP/FILTER/OFMAP SRAM capacities for a GEMM tile.

    SCALE-Sim configs are in KB and many existing configs effectively model one
    element as one word/byte. We therefore round up to at least 1KB and add a
    safety factor to avoid failing only because of double-buffer bookkeeping.
    """

    ifmap = max(1, ceil(tile.m * tile.k * safety_factor / 1024))
    filt = max(1, ceil(tile.k * tile.n * safety_factor / 1024))
    ofmap = max(1, ceil(tile.m * tile.n * safety_factor / 1024))
    return ifmap, filt, ofmap


def _has_nondefault_layout(layout_specs: tuple[LayoutSpec, ...]) -> bool:
    default = load_layout_spec(DEFAULT_LAYOUT).path.resolve()
    return any(spec.path.resolve() != default for spec in layout_specs)


def _write_case_cfg(
    *,
    case_name: str,
    array_h: int,
    array_w: int,
    tile: TileShape,
    dataflow: str,
    base_cfg: Path,
    bandwidth: int,
    ifmap_sram_kb: int,
    filter_sram_kb: int,
    ofmap_sram_kb: int,
    use_custom_layouts: bool,
) -> ArchSpec:
    run_name = f"boundary_{case_name}_{array_h}x{array_w}_{tile.label}_{dataflow}"
    cfg_path = write_arch_cfg(
        base_cfg=base_cfg,
        out_cfg=generated_cfg_path(run_name),
        params=ArchConfigParams(
            run_name=run_name,
            array_h=array_h,
            array_w=array_w,
            bandwidth=bandwidth,
            ifmap_sram_kb=ifmap_sram_kb,
            filter_sram_kb=filter_sram_kb,
            ofmap_sram_kb=ofmap_sram_kb,
            dataflow=dataflow,
            ifmap_custom_layout=use_custom_layouts,
            filter_custom_layout=use_custom_layouts,
        ),
    )
    return load_arch_spec(cfg_path)


def build_boundary_configs(
    *,
    mnk: MnkShape = MnkShape(32, 32, 64),
    array_h: int = 8,
    array_w: int = 8,
    base_cfg: Path = DEFAULT_ARCH_CFG,
    layouts: Iterable[Path] = (DEFAULT_LAYOUT,),
    dataflow: str = "ws",
    tile_k: int = 8,
    bandwidths: tuple[int, ...] = (600, 300, 200, 100, 80, 60, 40, 30, 20, 10),
    sram_tile: TileShape | None = None,
    use_custom_layouts: bool | None = None,
) -> tuple[SweepConfig, ...]:
    """Build default boundary sweep configs.

    The returned configs can be passed directly to ``tools.sweep_runner.run_sweep``.
    """

    configs: list[SweepConfig] = []
    layout_specs = tuple(load_layout_spec(path) for path in layouts)
    if use_custom_layouts is None:
        # Keep custom layout mode opt-in.  Passing a different layout CSV to
        # SCALE-Sim is not the same as enabling SCALE-Sim's custom-layout
        # banking mode; the latter has stricter format/topology constraints and
        # can fail for otherwise valid GEMM topologies.
        use_custom_layouts = False

    def append(case_name: str, tile: TileShape, arch: ArchSpec, layout: LayoutSpec) -> None:
        configs.append(
            SweepConfig(
                mnk=mnk,
                tile_shape=tile,
                arch_spec=arch,
                layout_spec=layout,
                case_name=case_name,
            )
        )

    # Tile-vs-array cases with the base bandwidth/SRAM settings.
    for case_name, tile in tile_smaller_equal_larger(array_h, array_w, tile_k):
        arch = _write_case_cfg(
            case_name=case_name,
            array_h=array_h,
            array_w=array_w,
            tile=tile,
            dataflow=dataflow,
            base_cfg=base_cfg,
            bandwidth=10,
            ifmap_sram_kb=64,
            filter_sram_kb=64,
            ofmap_sram_kb=64,
            use_custom_layouts=use_custom_layouts,
        )
        for layout in layout_specs:
            append(case_name, tile, arch, layout)

    # Bandwidth cases keep tile equal to the array and vary only bandwidth.
    equal_tile = TileShape(array_h, array_w, tile_k)
    for bw in bandwidths:
        case_name = "bandwidth_large" if bw == max(bandwidths) else f"bandwidth_{bw}"
        arch = _write_case_cfg(
            case_name=case_name,
            array_h=array_h,
            array_w=array_w,
            tile=equal_tile,
            dataflow=dataflow,
            base_cfg=base_cfg,
            bandwidth=bw,
            ifmap_sram_kb=64,
            filter_sram_kb=64,
            ofmap_sram_kb=64,
            use_custom_layouts=use_custom_layouts,
        )
        for layout in layout_specs:
            append(case_name, equal_tile, arch, layout)

    # SRAM cases use a larger tile by default so that 1KB is not trivially enough.
    sram_tile = sram_tile or default_sram_tile(mnk, array_h, array_w, tile_k)
    fit_ifmap, fit_filter, fit_ofmap = estimate_tile_sram_kb(sram_tile)
    tight_ifmap = max(1, fit_ifmap // 2)
    tight_filter = max(1, fit_filter // 2)
    tight_ofmap = max(1, fit_ofmap // 2)
    sram_cases = (
        ("sram_large", 64, 64, 64),
        ("sram_fit", fit_ifmap, fit_filter, fit_ofmap),
        ("sram_tight", tight_ifmap, tight_filter, tight_ofmap),
    )
    for case_name, ifmap_kb, filter_kb, ofmap_kb in sram_cases:
        arch = _write_case_cfg(
            case_name=case_name,
            array_h=array_h,
            array_w=array_w,
            tile=sram_tile,
            dataflow=dataflow,
            base_cfg=base_cfg,
            bandwidth=10,
            ifmap_sram_kb=ifmap_kb,
            filter_sram_kb=filter_kb,
            ofmap_sram_kb=ofmap_kb,
            use_custom_layouts=use_custom_layouts,
        )
        for layout in layout_specs:
            append(case_name, sram_tile, arch, layout)

    return tuple(configs)
