"""TPUv2 grouped-full experiments for LLM-like GEMM workloads.

The workloads are representative matrix multiplications from transformer blocks.
They are not a full end-to-end LLM runtime model; they are intended to stress the
same large GEMM shapes that dominate prefill/decode compute:

* QKV projection:      [tokens x hidden] @ [hidden x 3*hidden]
* output projection:   [tokens x hidden] @ [hidden x hidden]
* gate/up projection:  [tokens x hidden] @ [hidden x 2*ffn]
* down projection:     [tokens x ffn]    @ [ffn x hidden]

The runner reuses the TPUv2 grouped-full pipeline, so it derives the full tiling
space and then groups identical tile rows before invoking SCALE-Sim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from tools.io_utils import write_dataframe_outputs

import pandas as pd

from tools.sweep_runner import (
    DEFAULT_LAYOUT,
    MnkShape,
    SweepConfig,
    load_arch_spec,
    load_layout_spec,
    run_sweep,
)
from tools.tpuv2_experiment import (
    DEFAULT_TPUV2_CFG,
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLOT_DIR,
    TILE_RELATION_FACTORS,
    _extract_case_tags,
    _make_case,
    _positive_int,
    _safe_float_label,
    _scaled_sram,
    enrich_metrics,
    make_plots,
    parse_array,
    parse_float,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LLM_OUTPUT = REPO / "results" / "llm_tpuv2_experiment.parquet"
DEFAULT_LLM_CSV = REPO / "results" / "llm_tpuv2_experiment.csv"
DEFAULT_LLM_OUTPUT_ROOT = REPO / "outputs" / "llm_tpuv2_experiment"
DEFAULT_LLM_PLOT_DIR = REPO / "results" / "figures" / "llm_tpuv2_experiment"
DEFAULT_LLM_CACHE_ROOT = REPO / "outputs" / "cache" / "llm_tpuv2_experiment"


@dataclass(frozen=True)
class LlmGemm:
    preset: str
    op: str
    m: int
    n: int
    k: int
    description: str

    @property
    def mnk(self) -> MnkShape:
        return MnkShape(self.m, self.n, self.k)

    @property
    def tag(self) -> str:
        return f"{self.preset}-{self.op}".replace("_", "-")


LLM_PRESETS: dict[str, tuple[LlmGemm, ...]] = {
    # LLaMA-2/3 7B-like block, hidden=4096, ffn=11008, prefill sequence 2048.
    "llama7b_prefill_2048": (
        LlmGemm("llama7b_prefill_2048", "qkv", 2048, 12288, 4096, "QKV projection, M=tokens"),
        LlmGemm("llama7b_prefill_2048", "o_proj", 2048, 4096, 4096, "Attention output projection"),
        LlmGemm("llama7b_prefill_2048", "gate_up", 2048, 22016, 4096, "Fused gate/up MLP projection"),
        LlmGemm("llama7b_prefill_2048", "down", 2048, 4096, 11008, "MLP down projection"),
    ),
    # Decode is latency-sensitive and has tiny M. This tends to underutilize large arrays.
    "llama7b_decode_1": (
        LlmGemm("llama7b_decode_1", "qkv", 1, 12288, 4096, "QKV projection for one token"),
        LlmGemm("llama7b_decode_1", "o_proj", 1, 4096, 4096, "Attention output projection for one token"),
        LlmGemm("llama7b_decode_1", "gate_up", 1, 22016, 4096, "Fused gate/up projection for one token"),
        LlmGemm("llama7b_decode_1", "down", 1, 4096, 11008, "MLP down projection for one token"),
    ),
    # Larger model option, hidden=8192, ffn=28672, prefill sequence 2048.
    "llama70b_prefill_2048": (
        LlmGemm("llama70b_prefill_2048", "qkv", 2048, 24576, 8192, "QKV projection"),
        LlmGemm("llama70b_prefill_2048", "o_proj", 2048, 8192, 8192, "Attention output projection"),
        LlmGemm("llama70b_prefill_2048", "gate_up", 2048, 57344, 8192, "Fused gate/up MLP projection"),
        LlmGemm("llama70b_prefill_2048", "down", 2048, 8192, 28672, "MLP down projection"),
    ),
}


def _build_llm_cases(
    *,
    workloads: Iterable[LlmGemm],
    arrays: Iterable[tuple[int, int]],
    tile_relations: Iterable[str],
    bandwidths: Iterable[int],
    sram_scales: Iterable[float],
    base_cfg: Path,
    layouts: Iterable[Path],
    dataflow: str,
    tile_k: int | None,
    mode: str,
    use_custom_layouts: bool,
):
    base_arch = load_arch_spec(base_cfg)
    cases = []
    seen = set()

    def append(*, workload: LlmGemm, array: tuple[int, int], relation: str, bandwidth: int, sram_scale: float, layout: Path) -> None:
        case = _make_case(
            group=f"llm-{workload.tag}",
            base_cfg=base_cfg,
            base_ifmap_sram_kb=base_arch.ifmap_sram_kb,
            base_filter_sram_kb=base_arch.filter_sram_kb,
            base_ofmap_sram_kb=base_arch.ofmap_sram_kb,
            mnk=workload.mnk,
            array=array,
            relation=relation,
            bandwidth=bandwidth,
            sram_scale=sram_scale,
            tile_k=tile_k,
            layout=layout,
            dataflow=dataflow,
            use_custom_layouts=use_custom_layouts,
        )
        key = (workload.tag, case.mnk.label, case.tile.label, case.array_h, case.array_w, case.bandwidth, case.sram_scale, str(case.layout))
        if key not in seen:
            seen.add(key)
            cases.append(case)

    for workload in workloads:
        for array in arrays:
            for layout in layouts:
                if mode == "factorial":
                    for relation in tile_relations:
                        for bandwidth in bandwidths:
                            for sram_scale in sram_scales:
                                append(workload=workload, array=array, relation=relation, bandwidth=bandwidth, sram_scale=sram_scale, layout=layout)
                else:
                    for relation in tile_relations:
                        append(workload=workload, array=array, relation=relation, bandwidth=base_arch.bandwidth, sram_scale=1.0, layout=layout)
                    for bandwidth in bandwidths:
                        append(workload=workload, array=array, relation="equal", bandwidth=bandwidth, sram_scale=1.0, layout=layout)
                    for sram_scale in sram_scales:
                        append(workload=workload, array=array, relation="equal", bandwidth=base_arch.bandwidth, sram_scale=sram_scale, layout=layout)
    return tuple(cases)


def _cases_to_configs(cases) -> tuple[SweepConfig, ...]:
    configs = []
    for case in cases:
        configs.append(
            SweepConfig(
                mnk=case.mnk,
                tile_shape=case.tile,
                arch_spec=load_arch_spec(case.arch_cfg),
                layout_spec=load_layout_spec(case.layout),
                case_name=case.case_name,
            )
        )
    return tuple(configs)


def _preview(cases) -> pd.DataFrame:
    rows = []
    for idx, case in enumerate(cases, start=1):
        rows.append(
            {
                "idx": idx,
                "case_name": case.case_name,
                "mnk": case.mnk.label,
                "array": f"{case.array_h}x{case.array_w}",
                "tile": case.tile.label,
                "bandwidth": case.bandwidth,
                "sram_scale": case.sram_scale,
                "sram_kb": f"{case.ifmap_sram_kb}/{case.filter_sram_kb}/{case.ofmap_sram_kb}",
                "layout": case.layout.stem,
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TPUv2 grouped-full sweeps on LLM-like GEMM workloads")
    ap.add_argument("--preset", choices=tuple(LLM_PRESETS), action="append", default=None, help="LLM workload preset. May be repeated")
    ap.add_argument("--base-cfg", type=Path, default=DEFAULT_TPUV2_CFG)
    ap.add_argument("--array", type=parse_array, action="append", default=None)
    ap.add_argument("--tile-relation", choices=tuple(TILE_RELATION_FACTORS), action="append", default=None)
    ap.add_argument("--tile-k", type=_positive_int, default=None, help="fixed tile K. Default: array width")
    ap.add_argument("--bandwidth", type=_positive_int, action="append", default=None)
    ap.add_argument("--sram-scale", type=parse_float, action="append", default=None)
    ap.add_argument("--layout", type=Path, action="append", default=None)
    ap.add_argument("--dataflow", default="ws", choices=("ws", "os", "is"))
    ap.add_argument("--mode", choices=("one-factor", "factorial"), default="one-factor")
    ap.add_argument("--custom-layout", action="store_true")
    ap.add_argument("--output", type=Path, default=DEFAULT_LLM_OUTPUT)
    ap.add_argument("--csv-output", type=Path, default=DEFAULT_LLM_CSV)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_LLM_OUTPUT_ROOT)
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_LLM_PLOT_DIR)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_LLM_CACHE_ROOT)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--limit", type=_positive_int, default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--jobs", type=_positive_int, default=1)
    ap.add_argument("--parallel-backend", choices=("thread", "process"), default="thread")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-iree", action="store_true")
    ap.add_argument("--topology-mode", choices=("grouped_full", "raw"), default="grouped_full")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--heartbeat-sec", type=int, default=30)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    presets = tuple(args.preset) if args.preset else ("llama7b_prefill_2048",)
    workloads = tuple(w for preset in presets for w in LLM_PRESETS[preset])
    arrays = tuple(args.array) if args.array else ((128, 128), (256, 256))
    tile_relations = tuple(args.tile_relation) if args.tile_relation else ("equal", "larger")
    bandwidths = tuple(args.bandwidth) if args.bandwidth else (1200, 600, 300, 150)
    # Include much smaller SRAM scales by default because large GEMMs often need
    # stronger pressure than 0.25x to show SRAM sensitivity in SCALE-Sim.
    sram_scales = tuple(args.sram_scale) if args.sram_scale else (1.0, 0.25, 0.0625, 0.015625)
    layouts = tuple(args.layout) if args.layout else (DEFAULT_LAYOUT,)

    cases = _build_llm_cases(
        workloads=workloads,
        arrays=arrays,
        tile_relations=tile_relations,
        bandwidths=bandwidths,
        sram_scales=sram_scales,
        base_cfg=args.base_cfg,
        layouts=layouts,
        dataflow=args.dataflow,
        tile_k=args.tile_k,
        mode=args.mode,
        use_custom_layouts=args.custom_layout,
    )
    if args.limit is not None:
        cases = cases[: args.limit]
    preview = _preview(cases)
    if not args.quiet:
        print("LLM TPUv2 experiment configuration:", flush=True)
        print(f"  presets={','.join(presets)} workloads={len(workloads)} cases={len(cases)} mode={args.mode}", flush=True)
        print(f"  jobs={args.jobs} backend={args.parallel_backend} topology_mode={args.topology_mode} skip_iree={args.skip_iree}", flush=True)
        print(preview.head(min(12, len(preview))).to_string(index=False), flush=True)
    if args.dry_run:
        print(f"dry-run: generated {len(cases)} case(s)")
        print(preview.to_string(index=False))
        return

    df = run_sweep(
        configs=_cases_to_configs(cases),
        out_path=args.output,
        output_root=args.output_root,
        clean=(not args.no_clean and not args.resume),
        csv_out=args.csv_output,
        fail_fast=args.fail_fast,
        jobs=args.jobs,
        resume=args.resume,
        skip_iree=args.skip_iree,
        topology_mode=args.topology_mode,
        cache_root=None if args.no_cache else args.cache_root,
        parallel_backend=args.parallel_backend,
        verbose=not args.quiet,
        heartbeat_sec=max(0, args.heartbeat_sec),
    )
    df = _extract_case_tags(df)
    df = enrich_metrics(df)
    write_dataframe_outputs(df, args.output, args.csv_output)
    print(f"wrote {args.output} ({len(df)} rows x {len(df.columns)} cols)")
    print(f"wrote {args.csv_output}")
    print(f"status counts: {df['status'].value_counts(dropna=False).to_dict()}")
    if not args.skip_plots:
        generated = make_plots(df, args.plot_dir)
        if generated:
            print("generated plots/tables:")
            for path in generated:
                print(f"  {path}")


if __name__ == "__main__":
    main()
