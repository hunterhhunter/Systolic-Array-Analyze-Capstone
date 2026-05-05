"""CLI for array/bandwidth/SRAM/layout boundary sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tools.boundary_cases import build_boundary_configs, parse_array, parse_tile_shape
from tools.sweep_runner import DEFAULT_ARCH_CFG, DEFAULT_LAYOUT, RESULTS_DIR, SWEEP_DIR, SweepConfig, parse_mnk, run_sweep

DEFAULT_OUT = RESULTS_DIR / "boundary_sweep.parquet"
DEFAULT_ROOT = SWEEP_DIR.parent / "boundary_sweep"
BOUNDARY_GROUPS = ("tile", "bandwidth", "sram")


def _case_group(cfg: SweepConfig) -> str:
    if cfg.case_name.startswith("bandwidth"):
        return "bandwidth"
    if cfg.case_name.startswith("sram"):
        return "sram"
    return "tile"


def _filter_configs(
    configs: tuple[SweepConfig, ...],
    *,
    only: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[SweepConfig, ...]:
    selected = configs
    if only:
        wanted = set(only)
        selected = tuple(cfg for cfg in selected if _case_group(cfg) in wanted)
    if limit is not None:
        selected = selected[:limit]
    return selected


def _configs_preview(configs: tuple[SweepConfig, ...]) -> pd.DataFrame:
    rows = []
    for idx, cfg in enumerate(configs, start=1):
        rows.append(
            {
                "idx": idx,
                "group": _case_group(cfg),
                "case_name": cfg.case_name,
                "mnk": cfg.mnk.label,
                "tile": cfg.tile,
                "array": f"{cfg.arch_spec.array_h}x{cfg.arch_spec.array_w}",
                "bandwidth": cfg.arch_spec.bandwidth,
                "ifmap_sram_kb": cfg.arch_spec.ifmap_sram_kb,
                "filter_sram_kb": cfg.arch_spec.filter_sram_kb,
                "ofmap_sram_kb": cfg.arch_spec.ofmap_sram_kb,
                "layout": cfg.layout,
                "arch_cfg": str(cfg.arch_cfg),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MatMul array/bandwidth/SRAM/layout boundary sweep")
    ap.add_argument("--mnk", type=parse_mnk, default=parse_mnk("32x32x64"))
    ap.add_argument("--array", type=parse_array, default=parse_array("8x8"), help="array shape, e.g. 8x8")
    ap.add_argument("--tile-k", type=int, default=8)
    ap.add_argument(
        "--bandwidth",
        type=int,
        action="append",
        default=None,
        help="bandwidth value to include. May be repeated. Default: 600,300,200,100,80,60,40,30,20,10",
    )
    ap.add_argument(
        "--sram-tile",
        type=parse_tile_shape,
        default=None,
        help="tile used only for SRAM boundary cases, e.g. 32x32x32. Default chooses a larger valid tile.",
    )
    ap.add_argument("--base-cfg", type=Path, default=DEFAULT_ARCH_CFG)
    ap.add_argument("--layout", type=Path, action="append", default=None, help="layout CSV path; may be repeated")
    ap.add_argument("--dataflow", default="ws", choices=("ws", "os", "is"))
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csv-output", type=Path, default=DEFAULT_OUT.with_suffix(".csv"))
    ap.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print generated cases without running IREE/SCALE-Sim")
    ap.add_argument("--limit", type=int, default=None, help="run at most N generated cases after filtering")
    ap.add_argument(
        "--only",
        choices=BOUNDARY_GROUPS,
        action="append",
        default=None,
        help="run only one case group; repeat for multiple groups, e.g. --only tile --only sram",
    )
    ap.add_argument("--fail-fast", action="store_true", help="stop on the first non-ok sweep row")
    layout_group = ap.add_mutually_exclusive_group()
    layout_group.add_argument(
        "--custom-layout",
        action="store_true",
        default=None,
        help="force custom layout flags on in generated SCALE-Sim configs",
    )
    layout_group.add_argument(
        "--no-custom-layout",
        action="store_false",
        dest="custom_layout",
        help="force custom layout flags off in generated SCALE-Sim configs",
    )
    args = ap.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    array_h, array_w = args.array
    layouts = tuple(args.layout) if args.layout else (DEFAULT_LAYOUT,)
    bandwidths = tuple(args.bandwidth) if args.bandwidth else (600, 300, 200, 100, 80, 60, 40, 30, 20, 10)
    configs = build_boundary_configs(
        mnk=args.mnk,
        array_h=array_h,
        array_w=array_w,
        base_cfg=args.base_cfg,
        layouts=layouts,
        dataflow=args.dataflow,
        tile_k=args.tile_k,
        bandwidths=bandwidths,
        sram_tile=args.sram_tile,
        use_custom_layouts=args.custom_layout,
    )
    configs = _filter_configs(configs, only=tuple(args.only) if args.only else None, limit=args.limit)
    if not configs:
        raise SystemExit("no boundary cases selected; check --only/--limit arguments")

    preview = _configs_preview(configs)
    if args.dry_run:
        print(f"dry-run: generated {len(preview)} case(s); no IREE/SCALE-Sim execution")
        print(preview.to_string(index=False))
        return

    df = run_sweep(
        configs=configs,
        out_path=args.output,
        output_root=args.output_root,
        clean=not args.no_clean,
        csv_out=args.csv_output,
        fail_fast=args.fail_fast,
    )
    print(f"wrote {args.output} ({len(df)} rows x {len(df.columns)} cols)")
    print(f"wrote {args.csv_output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
