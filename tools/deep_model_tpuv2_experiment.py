"""TPUv2 grouped-full sweeps for diverse deep-learning GEMM workloads.

This runner is meant for the next stage after the LLM-only experiments: search
for quantitative tiling-efficiency indicators across SRAM size, cache/bank
bandwidth, tile shape, memory bandwidth, dataflow, and multiple model families.

The workloads are representative GEMM shapes extracted from common DL patterns:
transformer projections, CNN/im2col convolutions, ViT/BERT blocks, MLPs, and
recommendation MLPs.  They are not full framework execution traces.  The goal is
architectural/tile sensitivity screening with the existing SCALE-Sim pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from tools.config_factory import ArchConfigParams, generated_cfg_path, write_arch_cfg
from tools.sweep_runner import (
    DEFAULT_LAYOUT,
    RESULTS_DIR,
    MnkShape,
    SweepConfig,
    TileShape,
    load_arch_spec,
    load_layout_spec,
    run_sweep,
)
from tools.io_utils import write_dataframe_outputs
from tools.tpuv2_experiment import (
    DEFAULT_TPUV2_CFG,
    _extract_case_tags,
    _positive_int,
    _safe_float_label,
    enrich_metrics,
    parse_array,
    parse_float,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = RESULTS_DIR / "deep_model_tpuv2_experiment.parquet"
DEFAULT_CSV = RESULTS_DIR / "deep_model_tpuv2_experiment.csv"
DEFAULT_ENRICHED_CSV = RESULTS_DIR / "deep_model_tpuv2_experiment_enriched.csv"
DEFAULT_OUTPUT_ROOT = REPO / "outputs" / "deep_model_tpuv2_experiment"
DEFAULT_CACHE_ROOT = REPO / "outputs" / "cache" / "deep_model_tpuv2_experiment"
DEFAULT_PLOT_DIR = REPO / "results" / "figures" / "deep_model_tpuv2_experiment"


@dataclass(frozen=True)
class ModelGemm:
    preset: str
    model: str
    op: str
    m: int
    n: int
    k: int
    category: str
    description: str

    @property
    def mnk(self) -> MnkShape:
        return MnkShape(self.m, self.n, self.k)

    @property
    def slug(self) -> str:
        return _slug(f"{self.model}-{self.op}")


@dataclass(frozen=True)
class DeepCase:
    workload: ModelGemm
    group: str
    case_name: str
    mnk: MnkShape
    tile: TileShape
    array_h: int
    array_w: int
    bandwidth: int
    sram_scale: float
    cache_bw_scale: float
    ifmap_sram_kb: int
    filter_sram_kb: int
    ofmap_sram_kb: int
    arch_cfg: Path
    layout: Path
    dataflow: str



def _dense_transformer_gemms(
    *,
    preset: str,
    model: str,
    hidden: int,
    intermediate: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seqs: Iterable[int],
    category: str = "transformer_large",
    include_decode: bool = True,
) -> tuple[ModelGemm, ...]:
    """Build representative dense/hybrid Transformer GEMM proxy shapes.

    These are layer-level matrix shapes, not full model traces.  They are meant
    to stress the MatMul/tile pipeline with realistic projection dimensions.
    """

    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    rows: list[ModelGemm] = []
    for seq in seqs:
        tag = f"seq{seq}"
        rows.extend(
            [
                ModelGemm(preset, f"{model}_{tag}", "qkv", seq, qkv_dim, hidden, category, f"{model} {tag} fused QKV/GQA projection proxy"),
                ModelGemm(preset, f"{model}_{tag}", "o_proj", seq, hidden, q_dim, category, f"{model} {tag} attention output projection proxy"),
                ModelGemm(preset, f"{model}_{tag}", "mlp_gate_up", seq, 2 * intermediate, hidden, category, f"{model} {tag} fused MLP gate/up projection"),
                ModelGemm(preset, f"{model}_{tag}", "mlp_down", seq, hidden, intermediate, category, f"{model} {tag} MLP down projection"),
            ]
        )
    if include_decode:
        rows.extend(
            [
                ModelGemm(preset, f"{model}_decode1", "qkv", 1, qkv_dim, hidden, "transformer_decode", f"{model} single-token decode QKV/GQA projection proxy"),
                ModelGemm(preset, f"{model}_decode1", "o_proj", 1, hidden, q_dim, "transformer_decode", f"{model} single-token decode output projection proxy"),
                ModelGemm(preset, f"{model}_decode1", "mlp_gate_up", 1, 2 * intermediate, hidden, "transformer_decode", f"{model} single-token decode MLP gate/up"),
                ModelGemm(preset, f"{model}_decode1", "mlp_down", 1, hidden, intermediate, "transformer_decode", f"{model} single-token decode MLP down"),
            ]
        )
    return tuple(rows)


def _moe_transformer_gemms(
    *,
    preset: str,
    model: str,
    hidden: int,
    dense_intermediate: int | None,
    moe_intermediate: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    num_experts: int,
    experts_per_token: int,
    seqs: Iterable[int],
    category: str = "transformer_moe_large",
    include_decode: bool = True,
) -> tuple[ModelGemm, ...]:
    """Build representative MoE Transformer GEMM proxy shapes.

    The active expert paths are flattened as if the selected experts were
    evaluated as one larger effective GEMM.  This keeps the benchmark matrix-only
    while preserving the top-k MoE compute multiplier.
    """

    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    active_intermediate = moe_intermediate * experts_per_token
    rows: list[ModelGemm] = []
    for seq in seqs:
        tag = f"seq{seq}"
        rows.extend(
            [
                ModelGemm(preset, f"{model}_{tag}", "qkv", seq, qkv_dim, hidden, category, f"{model} {tag} fused QKV/GQA projection proxy"),
                ModelGemm(preset, f"{model}_{tag}", "o_proj", seq, hidden, q_dim, category, f"{model} {tag} attention output projection proxy"),
                ModelGemm(preset, f"{model}_{tag}", "router", seq, num_experts, hidden, "transformer_moe_router", f"{model} {tag} router projection"),
                ModelGemm(preset, f"{model}_{tag}", "moe_gate_up_active", seq, 2 * active_intermediate, hidden, category, f"{model} {tag} top-{experts_per_token} active expert gate/up proxy"),
                ModelGemm(preset, f"{model}_{tag}", "moe_down_active", seq, hidden, active_intermediate, category, f"{model} {tag} top-{experts_per_token} active expert down proxy"),
            ]
        )
        if dense_intermediate:
            rows.extend(
                [
                    ModelGemm(preset, f"{model}_{tag}", "dense_gate_up", seq, 2 * dense_intermediate, hidden, category, f"{model} {tag} dense fallback gate/up proxy"),
                    ModelGemm(preset, f"{model}_{tag}", "dense_down", seq, hidden, dense_intermediate, category, f"{model} {tag} dense fallback down proxy"),
                ]
            )
    if include_decode:
        rows.extend(
            [
                ModelGemm(preset, f"{model}_decode1", "qkv", 1, qkv_dim, hidden, "transformer_decode", f"{model} single-token decode QKV/GQA projection proxy"),
                ModelGemm(preset, f"{model}_decode1", "o_proj", 1, hidden, q_dim, "transformer_decode", f"{model} single-token decode output projection proxy"),
                ModelGemm(preset, f"{model}_decode1", "router", 1, num_experts, hidden, "transformer_decode", f"{model} single-token decode router"),
                ModelGemm(preset, f"{model}_decode1", "moe_gate_up_active", 1, 2 * active_intermediate, hidden, "transformer_decode", f"{model} single-token top-{experts_per_token} expert gate/up"),
                ModelGemm(preset, f"{model}_decode1", "moe_down_active", 1, hidden, active_intermediate, "transformer_decode", f"{model} single-token top-{experts_per_token} expert down"),
            ]
        )
    return tuple(rows)


def _load_custom_workloads(path: Path) -> tuple[ModelGemm, ...]:
    """Load extra model GEMM proxies from a JSON file.

    Supported JSON shapes:
      {"workloads": [{"preset": "custom", "model": "...", "op": "...", "m": 1, "n": 2, "k": 3, ...}]}
      [{"preset": "custom", "model": "...", "op": "...", "m": 1, "n": 2, "k": 3, ...}]
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("workloads", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("custom workload JSON must be a list or contain a 'workloads' list")
    rows: list[ModelGemm] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"custom workload #{i} must be an object")
        try:
            preset = str(item.get("preset", "custom"))
            model = str(item["model"])
            op = str(item["op"])
            m = int(item["m"])
            n = int(item["n"])
            k = int(item["k"])
        except KeyError as exc:
            raise ValueError(f"custom workload #{i} is missing required field {exc.args[0]!r}") from exc
        if min(m, n, k) <= 0:
            raise ValueError(f"custom workload #{i} has non-positive M/N/K")
        rows.append(
            ModelGemm(
                preset=preset,
                model=model,
                op=op,
                m=m,
                n=n,
                k=k,
                category=str(item.get("category", "custom")),
                description=str(item.get("description", f"custom GEMM {m}x{n}x{k}")),
            )
        )
    return tuple(rows)

# Representative GEMM shapes.  For Conv2D, M is output spatial positions,
# N is output channels, and K is input_channels * kernel_h * kernel_w.
MODEL_PRESETS: dict[str, tuple[ModelGemm, ...]] = {
    "llama7b_prefill_decode": (
        ModelGemm("llama7b_prefill_decode", "llama7b_prefill2048", "qkv", 2048, 12288, 4096, "transformer", "LLaMA-7B prefill QKV"),
        ModelGemm("llama7b_prefill_decode", "llama7b_prefill2048", "gate_up", 2048, 22016, 4096, "transformer", "LLaMA-7B prefill fused gate/up"),
        ModelGemm("llama7b_prefill_decode", "llama7b_decode1", "qkv", 1, 12288, 4096, "transformer_decode", "LLaMA-7B single-token decode QKV"),
        ModelGemm("llama7b_prefill_decode", "llama7b_decode1", "gate_up", 1, 22016, 4096, "transformer_decode", "LLaMA-7B single-token decode gate/up"),
    ),
    "bert_base": (
        ModelGemm("bert_base", "bert_base_seq128", "qkv", 128, 2304, 768, "transformer", "BERT-base seq128 QKV"),
        ModelGemm("bert_base", "bert_base_seq128", "ffn1", 128, 3072, 768, "transformer", "BERT-base seq128 FFN expand"),
        ModelGemm("bert_base", "bert_base_seq512", "qkv", 512, 2304, 768, "transformer", "BERT-base seq512 QKV"),
        ModelGemm("bert_base", "bert_base_seq512", "ffn1", 512, 3072, 768, "transformer", "BERT-base seq512 FFN expand"),
    ),
    "vit_base": (
        ModelGemm("vit_base", "vit_b16_224", "patch_embed", 196, 768, 768, "vision_transformer", "ViT-B/16 patch projection after flattening"),
        ModelGemm("vit_base", "vit_b16_224", "qkv", 197, 2304, 768, "vision_transformer", "ViT-B/16 QKV"),
        ModelGemm("vit_base", "vit_b16_224", "mlp1", 197, 3072, 768, "vision_transformer", "ViT-B/16 MLP expand"),
        ModelGemm("vit_base", "vit_b16_224", "mlp2", 197, 768, 3072, "vision_transformer", "ViT-B/16 MLP project"),
    ),
    "resnet50": (
        ModelGemm("resnet50", "resnet50", "conv1", 112 * 112, 64, 7 * 7 * 3, "cnn", "ResNet-50 conv1 im2col GEMM"),
        ModelGemm("resnet50", "resnet50", "res2_3x3", 56 * 56, 64, 3 * 3 * 64, "cnn", "ResNet-50 early 3x3 conv"),
        ModelGemm("resnet50", "resnet50", "res3_3x3", 28 * 28, 128, 3 * 3 * 128, "cnn", "ResNet-50 mid 3x3 conv"),
        ModelGemm("resnet50", "resnet50", "res4_3x3", 14 * 14, 256, 3 * 3 * 256, "cnn", "ResNet-50 late 3x3 conv"),
        ModelGemm("resnet50", "resnet50", "res5_3x3", 7 * 7, 512, 3 * 3 * 512, "cnn", "ResNet-50 final 3x3 conv"),
        ModelGemm("resnet50", "resnet50", "fc", 1, 1000, 2048, "classifier", "ResNet-50 classifier"),
    ),
    "mobilenet_v2": (
        ModelGemm("mobilenet_v2", "mobilenet_v2", "pw_expand", 56 * 56, 384, 64, "mobile_cnn", "MobileNetV2 pointwise expand"),
        ModelGemm("mobilenet_v2", "mobilenet_v2", "dw_3x3", 56 * 56, 64, 3 * 3, "mobile_cnn", "MobileNetV2 depthwise-like GEMM proxy"),
        ModelGemm("mobilenet_v2", "mobilenet_v2", "pw_project", 56 * 56, 64, 384, "mobile_cnn", "MobileNetV2 pointwise project"),
        ModelGemm("mobilenet_v2", "mobilenet_v2", "late_pw", 7 * 7, 1280, 320, "mobile_cnn", "MobileNetV2 late pointwise"),
    ),
    "recommendation_mlp": (
        ModelGemm("recommendation_mlp", "dlrm_mlp", "bottom1", 8192, 512, 128, "recommender", "DLRM-like bottom MLP layer"),
        ModelGemm("recommendation_mlp", "dlrm_mlp", "bottom2", 8192, 256, 512, "recommender", "DLRM-like bottom MLP layer"),
        ModelGemm("recommendation_mlp", "dlrm_mlp", "top1", 8192, 1024, 512, "recommender", "DLRM-like top MLP layer"),
        ModelGemm("recommendation_mlp", "dlrm_mlp", "top2", 8192, 1, 1024, "recommender", "DLRM-like final scoring GEMM"),
    ),
    # OpenAI gpt-oss model-card proxy shapes.  These are layer-level GEMMs,
    # not full model traces.  Both gpt-oss models use hidden=2880, 64 query
    # heads of dim 64, 8 KV heads, and top-4 MoE routing; the 20B/120B
    # distinction mainly changes expert count and layer count, so both are
    # useful for router/expert sweep sensitivity.
    "gpt_oss20b": _moe_transformer_gemms(
        preset="gpt_oss20b", model="gpt_oss20b", hidden=2880, dense_intermediate=2880,
        moe_intermediate=2880, q_heads=64, kv_heads=8, head_dim=64,
        num_experts=32, experts_per_token=4, seqs=(2048, 8192),
    ),
    "gpt_oss120b": _moe_transformer_gemms(
        preset="gpt_oss120b", model="gpt_oss120b", hidden=2880, dense_intermediate=2880,
        moe_intermediate=2880, q_heads=64, kv_heads=8, head_dim=64,
        num_experts=128, experts_per_token=4, seqs=(2048, 8192),
    ),
    # Qwen3.5-27B text-path proxy shapes from public config fields.
    # The model is hybrid, but this benchmark is MatMul-centric, so we keep
    # the major projection/MLP GEMMs as architecture stress cases.
    "qwen35_27b": _dense_transformer_gemms(
        preset="qwen35_27b", model="qwen35_27b", hidden=5120, intermediate=17408,
        q_heads=24, kv_heads=4, head_dim=256, seqs=(4096, 16384),
    ),
    # Qwen3-235B-A22B large MoE proxy.  This is useful when the goal is to
    # stress very large active-expert GEMMs without requiring actual weights.
    "qwen3_235b_a22b": _moe_transformer_gemms(
        preset="qwen3_235b_a22b", model="qwen3_235b_a22b", hidden=4096, dense_intermediate=12288,
        moe_intermediate=1536, q_heads=64, kv_heads=4, head_dim=128,
        num_experts=128, experts_per_token=8, seqs=(4096, 16384),
    ),
}

DEFAULT_PRESETS = ("llama7b_prefill_decode", "bert_base", "vit_base", "resnet50")
LARGE_MODEL_PRESETS = ("gpt_oss20b", "gpt_oss120b", "qwen35_27b", "qwen3_235b_a22b")


def _slug(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return s or "case"


def _scaled_sram(base: int, scale: float) -> int:
    return max(1, int(round(base * scale)))


def _tile_shape_for(*, mnk: MnkShape, array: tuple[int, int], mn_factor: float, tile_k: int) -> TileShape:
    h, w = array
    tm = max(1, min(mnk.m, int(round(h * mn_factor))))
    tn = max(1, min(mnk.n, int(round(w * mn_factor))))
    tk = max(1, min(mnk.k, tile_k))
    return TileShape(tm, tn, tk)


def _write_case_cfg(
    *,
    base_cfg: Path,
    workload: ModelGemm,
    group: str,
    array: tuple[int, int],
    bandwidth: int,
    sram_scale: float,
    cache_bw_scale: float,
    base_ifmap_sram_kb: int,
    base_filter_sram_kb: int,
    base_ofmap_sram_kb: int,
    dataflow: str,
    custom_layout: bool,
) -> tuple[Path, int, int, int]:
    ifmap_sram = _scaled_sram(base_ifmap_sram_kb, sram_scale)
    filter_sram = _scaled_sram(base_filter_sram_kb, sram_scale)
    ofmap_sram = _scaled_sram(base_ofmap_sram_kb, sram_scale)
    run_name = _slug(
        f"deep-{workload.slug}-{group}-a{array[0]}x{array[1]}-bw{bandwidth}-sram{_safe_float_label(sram_scale)}-cache{_safe_float_label(cache_bw_scale)}-{dataflow}"
    )
    bank_bw = max(1, int(round(bandwidth * cache_bw_scale)))
    cfg = write_arch_cfg(
        base_cfg=base_cfg,
        out_cfg=generated_cfg_path(run_name),
        params=ArchConfigParams(
            run_name=run_name,
            array_h=array[0],
            array_w=array[1],
            ifmap_sram_kb=ifmap_sram,
            filter_sram_kb=filter_sram,
            ofmap_sram_kb=ofmap_sram,
            bandwidth=bandwidth,
            dataflow=dataflow,
            # Do not automatically enable SCALE-Sim custom-layout mode for cache/bank
            # bandwidth sweeps.  The custom-layout parser is fragile for many
            # generated GEMM topologies and can fail independently of the actual
            # bandwidth experiment.  We still write the bank bandwidth fields so
            # experiments can opt into custom layouts explicitly via --custom-layout.
            ifmap_custom_layout=custom_layout,
            filter_custom_layout=custom_layout,
            ifmap_bank_bw=bank_bw,
            filter_bank_bw=bank_bw,
        ),
    )
    return cfg, ifmap_sram, filter_sram, ofmap_sram


def _make_case(
    *,
    base_cfg: Path,
    base_arch,
    workload: ModelGemm,
    group: str,
    array: tuple[int, int],
    mn_factor: float,
    tile_k: int,
    bandwidth: int,
    sram_scale: float,
    cache_bw_scale: float,
    layout: Path,
    dataflow: str,
    custom_layout: bool,
) -> DeepCase:
    tile = _tile_shape_for(mnk=workload.mnk, array=array, mn_factor=mn_factor, tile_k=tile_k)
    arch_cfg, ifmap_sram, filter_sram, ofmap_sram = _write_case_cfg(
        base_cfg=base_cfg,
        workload=workload,
        group=group,
        array=array,
        bandwidth=bandwidth,
        sram_scale=sram_scale,
        cache_bw_scale=cache_bw_scale,
        base_ifmap_sram_kb=base_arch.ifmap_sram_kb,
        base_filter_sram_kb=base_arch.filter_sram_kb,
        base_ofmap_sram_kb=base_arch.ofmap_sram_kb,
        dataflow=dataflow,
        custom_layout=custom_layout,
    )
    case_name = (
        f"model-{workload.slug}_{group}_a{array[0]}x{array[1]}"
        f"_tile-{_safe_float_label(mn_factor)}_{tile.label}"
        f"_bw{bandwidth}_sram{_safe_float_label(sram_scale)}_cache{_safe_float_label(cache_bw_scale)}_{dataflow}"
    )
    return DeepCase(
        workload=workload,
        group=group,
        case_name=case_name,
        mnk=workload.mnk,
        tile=tile,
        array_h=array[0],
        array_w=array[1],
        bandwidth=bandwidth,
        sram_scale=sram_scale,
        cache_bw_scale=cache_bw_scale,
        ifmap_sram_kb=ifmap_sram,
        filter_sram_kb=filter_sram,
        ofmap_sram_kb=ofmap_sram,
        arch_cfg=arch_cfg,
        layout=layout,
        dataflow=dataflow,
    )


def _expand_presets(presets: Iterable[str]) -> tuple[ModelGemm, ...]:
    selected: list[ModelGemm] = []
    seen = set()
    for preset in presets:
        if preset == "all":
            keys = tuple(MODEL_PRESETS.keys())
        elif preset == "large_models":
            keys = LARGE_MODEL_PRESETS
        else:
            keys = (preset,)
        for key in keys:
            if key not in MODEL_PRESETS:
                raise KeyError(f"unknown model preset {key!r}; available: {', '.join(sorted(MODEL_PRESETS))}, all, large_models")
            for w in MODEL_PRESETS[key]:
                ident = (w.model, w.op, w.m, w.n, w.k)
                if ident not in seen:
                    seen.add(ident)
                    selected.append(w)
    return tuple(selected)


def build_cases(
    *,
    workloads: Iterable[ModelGemm],
    base_cfg: Path,
    arrays: Iterable[tuple[int, int]],
    tile_mn_factors: Iterable[float],
    tile_ks: Iterable[int],
    bandwidths: Iterable[int],
    sram_scales: Iterable[float],
    cache_bw_scales: Iterable[float],
    layouts: Iterable[Path],
    dataflows: Iterable[str],
    mode: str,
    custom_layout: bool,
) -> tuple[DeepCase, ...]:
    base_arch = load_arch_spec(base_cfg)
    cases: list[DeepCase] = []
    seen = set()

    def append(**kwargs) -> None:
        case = _make_case(base_cfg=base_cfg, base_arch=base_arch, custom_layout=custom_layout, **kwargs)
        key = (
            case.workload.slug,
            case.group,
            case.mnk.label,
            case.tile.label,
            case.array_h,
            case.array_w,
            case.bandwidth,
            case.sram_scale,
            case.cache_bw_scale,
            case.layout,
            case.dataflow,
        )
        if key not in seen:
            seen.add(key)
            cases.append(case)

    for workload in workloads:
        for array in arrays:
            for layout in layouts:
                for dataflow in dataflows:
                    base_tile_k = min(workload.k, array[1])
                    if mode == "factorial":
                        for mn_factor in tile_mn_factors:
                            for tile_k in tile_ks:
                                for bandwidth in bandwidths:
                                    for sram_scale in sram_scales:
                                        for cache_bw_scale in cache_bw_scales:
                                            append(workload=workload, group="factorial", array=array, mn_factor=mn_factor, tile_k=tile_k, bandwidth=bandwidth, sram_scale=sram_scale, cache_bw_scale=cache_bw_scale, layout=layout, dataflow=dataflow)
                    else:
                        # Tile shape grid: the main experiment for finding tiling-efficiency metrics.
                        for mn_factor in tile_mn_factors:
                            for tile_k in tile_ks:
                                append(workload=workload, group="tile", array=array, mn_factor=mn_factor, tile_k=tile_k, bandwidth=base_arch.bandwidth, sram_scale=1.0, cache_bw_scale=1.0, layout=layout, dataflow=dataflow)
                        # Off-chip memory bandwidth sensitivity at a canonical tile.
                        for bandwidth in bandwidths:
                            append(workload=workload, group="bandwidth", array=array, mn_factor=1.0, tile_k=base_tile_k, bandwidth=bandwidth, sram_scale=1.0, cache_bw_scale=1.0, layout=layout, dataflow=dataflow)
                        # On-chip SRAM capacity sensitivity / feasibility check.
                        for sram_scale in sram_scales:
                            append(workload=workload, group="sram", array=array, mn_factor=1.0, tile_k=base_tile_k, bandwidth=base_arch.bandwidth, sram_scale=sram_scale, cache_bw_scale=1.0, layout=layout, dataflow=dataflow)
                        # Cache/SRAM bank bandwidth sensitivity.  This is separate from off-chip bandwidth.
                        for cache_bw_scale in cache_bw_scales:
                            append(workload=workload, group="cache", array=array, mn_factor=1.0, tile_k=base_tile_k, bandwidth=base_arch.bandwidth, sram_scale=1.0, cache_bw_scale=cache_bw_scale, layout=layout, dataflow=dataflow)
    return tuple(cases)


def _cases_to_configs(cases: Iterable[DeepCase]) -> tuple[SweepConfig, ...]:
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


def preview_cases(cases: Iterable[DeepCase]) -> pd.DataFrame:
    rows = []
    for idx, c in enumerate(cases, 1):
        rows.append(
            {
                "idx": idx,
                "preset": c.workload.preset,
                "model": c.workload.model,
                "op": c.workload.op,
                "group": c.group,
                "mnk": c.mnk.label,
                "array": f"{c.array_h}x{c.array_w}",
                "tile": c.tile.label,
                "bandwidth": c.bandwidth,
                "sram_scale": c.sram_scale,
                "cache_bw_scale": c.cache_bw_scale,
                "dataflow": c.dataflow,
            }
        )
    return pd.DataFrame(rows)


def _add_deep_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = _extract_case_tags(df)

    # Deep-model case names include a model slug before the sweep group, for
    # example:
    #   model-bert-base-seq128-qkv_tile_a128x128_tile-1_128x128x128_bw600_sram1_cache1_ws
    # The generic TPUv2 extractor only understands names that begin directly
    # with ``tile_``/``bandwidth_``/``sram_``..., so we must recover the tags
    # again here for plotting.
    case_str = out["case_name"].astype(str)
    deep = case_str.str.extract(
        r"^model-(?P<model_case>.+?)_"
        r"(?P<deep_group>tile|bandwidth|sram|cache|factorial)_"
        r"a(?P<array_from_case>\d+x\d+)_"
        r"tile-(?P<tile_relation>[^_]+)_(?P<tile_from_case>\d+x\d+x\d+)_"
        r"bw(?P<bandwidth_from_case>\d+)_"
        r"sram(?P<sram_label>[0-9p]+)_"
        r"cache(?P<cache_bw_label>[0-9p]+)_"
        r"(?P<dataflow_from_case>ws|os|is)$"
    )
    for col in deep.columns:
        if col in out.columns:
            out[col] = out[col].fillna(deep[col])
        else:
            out[col] = deep[col]

    # Numeric backfills used by make_deep_plots filtering.
    sram_vals = pd.to_numeric(out["sram_label"].astype(str).str.replace("p", ".", regex=False), errors="coerce")
    if "sram_scale_from_case" in out.columns:
        out["sram_scale_from_case"] = out["sram_scale_from_case"].fillna(sram_vals)
    else:
        out["sram_scale_from_case"] = sram_vals

    cache_vals = pd.to_numeric(out["cache_bw_label"].astype(str).str.replace("p", ".", regex=False), errors="coerce")
    if "cache_bw_scale_from_case" in out.columns:
        out["cache_bw_scale_from_case"] = out["cache_bw_scale_from_case"].fillna(cache_vals)
    else:
        out["cache_bw_scale_from_case"] = cache_vals

    array_fill = out["array_h"].astype("Int64").astype(str) + "x" + out["array_w"].astype("Int64").astype(str)
    if "array_from_case" in out.columns:
        out["array_from_case"] = out["array_from_case"].fillna(array_fill)
    else:
        out["array_from_case"] = array_fill

    model_case = out["model_case"].astype(str)
    family_patterns = (
        "llama7b", "gpt-oss20b", "gpt-oss120b", "qwen35-27b", "qwen3-235b-a22b",
        "bert-base", "bert", "vit", "resnet50", "mobilenet-v2", "mobilenet", "dlrm", "recommendation",
    )
    family_re = "|".join(re.escape(x) for x in family_patterns)
    model = model_case.str.extract(rf"(?P<model_family>{family_re})-(?P<model_op>.*)$")
    fallback = model_case.str.extract(r"(?P<model_family>[^-]+)-(?P<model_op>.*)$")
    model["model_family"] = model["model_family"].fillna(fallback["model_family"])
    model["model_op"] = model["model_op"].fillna(fallback["model_op"])
    out["model_family_from_case"] = model["model_family"].fillna(out["model_case"])
    out["model_op_from_case"] = model["model_op"].fillna(out["model_case"])
    out["workload_label"] = out["model_case"].fillna(out.get("workload"))
    out["series_label"] = out["array_from_case"].astype(str) + " " + out["model_op_from_case"].astype(str)

    # Tile working-set proxy for BF16/FP16 inputs and FP32 accumulation output.
    bytes_in = 2
    bytes_out = 4
    out["tile_ifmap_kb"] = out["tile_m"] * out["tile_k"] * bytes_in / 1024.0
    out["tile_filter_kb"] = out["tile_k"] * out["tile_n"] * bytes_in / 1024.0
    out["tile_ofmap_kb"] = out["tile_m"] * out["tile_n"] * bytes_out / 1024.0
    out["tile_working_set_kb"] = out["tile_ifmap_kb"] + out["tile_filter_kb"] + out["tile_ofmap_kb"]
    out["tile_fits_sram"] = (
        (out["tile_ifmap_kb"] <= out["ifmap_sram_kb"])
        & (out["tile_filter_kb"] <= out["filter_sram_kb"])
        & (out["tile_ofmap_kb"] <= out["ofmap_sram_kb"])
    )
    # A simple tile-level arithmetic-intensity proxy.  It is intentionally local
    # to a tile and should be used for comparing tile shapes, not for claiming
    # end-to-end model roofline numbers.
    traffic_bytes = (out["tile_m"] * out["tile_k"] + out["tile_k"] * out["tile_n"] + out["tile_m"] * out["tile_n"]) * bytes_in
    out["tile_ai_proxy_ops_per_byte"] = (2.0 * out["tile_m"] * out["tile_n"] * out["tile_k"]) / traffic_bytes.replace(0, pd.NA)
    # Be robust if this helper is called directly before enrich_metrics().
    if "array_area" not in out.columns:
        out["array_area"] = out["array_h"] * out["array_w"]
    if "macs" not in out.columns:
        out["macs"] = out["m"] * out["n"] * out["k"]
    if "tile_area" not in out.columns:
        out["tile_area"] = out["tile_m"] * out["tile_n"]
    if "tile_relation_ratio" not in out.columns:
        out["tile_relation_ratio"] = out["tile_area"] / out["array_area"].replace(0, pd.NA)
    if "macs_per_cycle" not in out.columns:
        out["macs_per_cycle"] = out["macs"] / out["total_cycles"].replace(0, pd.NA)
    if "macs_per_cycle_per_pe" not in out.columns:
        out["macs_per_cycle_per_pe"] = out["macs_per_cycle"] / out["array_area"].replace(0, pd.NA)
    if "memory_overhead_cycles" not in out.columns:
        out["memory_overhead_cycles"] = (out["total_cycles"] - out["compute_cycles"]).clip(lower=0)
    if "memory_overhead_ratio" not in out.columns:
        out["memory_overhead_ratio"] = out["memory_overhead_cycles"] / out["total_cycles"].replace(0, pd.NA)
    out["tile_efficiency_score"] = out["macs_per_cycle_per_pe"] * (1.0 - out["memory_overhead_ratio"].fillna(0.0))
    out["latency_efficiency_score"] = out["tile_efficiency_score"] / out["total_cycles"].replace(0, pd.NA)

    # Deep-model-specific baseline: same workload/array/layout/dataflow, valid SRAM,
    # canonical SRAM/cache scales, then prefer tile factor ~= 1 and max bandwidth.
    # If no exact canonical tile exists (e.g. M=1 decode or tiny classifier), fall
    # back to the closest tile_relation_ratio.  This avoids all-NaN speedups while
    # exposing whether the baseline is exact or approximate.
    if "total_cycles" in out.columns:
        group_cols = ["workload", "m", "n", "k", "array_h", "array_w", "layout", "dataflow"]
        out["baseline_available"] = False
        out["baseline_reason"] = "no matching baseline row"
        out["baseline_is_approx"] = pd.NA
        out["deep_baseline_total_cycles"] = pd.NA
        out["speedup_vs_deep_baseline"] = pd.NA
        base_mask = (
            (out["status"] == "ok")
            & (out["deep_group"].isin(["tile", "bandwidth", "sram", "cache", "factorial"]))
            & (out["sram_scale_from_case"] == 1.0)
            & (out["cache_bw_scale_from_case"] == 1.0)
            & (out["tile_fits_sram"].fillna(False))
        )
        if base_mask.any():
            candidates = out.loc[base_mask, group_cols + ["bandwidth", "tile_relation_ratio", "total_cycles"]].copy()
            max_bw = candidates.groupby(group_cols)["bandwidth"].transform("max")
            candidates = candidates[candidates["bandwidth"] == max_bw].copy()
            candidates["baseline_distance"] = (candidates["tile_relation_ratio"] - 1.0).abs()
            candidates["baseline_is_approx"] = candidates["baseline_distance"] > 0.05
            base = (
                candidates.sort_values(group_cols + ["baseline_distance", "total_cycles"])
                .drop_duplicates(group_cols)
                .rename(columns={"total_cycles": "deep_baseline_total_cycles"})
            )
            base["baseline_available"] = True
            base["baseline_reason"] = base["baseline_is_approx"].map(
                {False: "exact canonical tile baseline", True: "closest valid tile baseline"}
            )
            out = out.drop(
                columns=["deep_baseline_total_cycles", "baseline_available", "baseline_reason", "baseline_is_approx"],
                errors="ignore",
            ).merge(
                base[group_cols + ["deep_baseline_total_cycles", "baseline_available", "baseline_reason", "baseline_is_approx"]],
                on=group_cols,
                how="left",
            )
            out["baseline_available"] = out["baseline_available"].fillna(False)
            out["baseline_reason"] = out["baseline_reason"].fillna("no matching baseline row")
            out["speedup_vs_deep_baseline"] = out["deep_baseline_total_cycles"] / out["total_cycles"].replace(0, pd.NA)
            # Keep the generic columns meaningful for deep-model results too.
            deep_base_numeric = pd.to_numeric(out["deep_baseline_total_cycles"], errors="coerce")
            if "baseline_total_cycles" in out.columns:
                generic_base_numeric = pd.to_numeric(out["baseline_total_cycles"], errors="coerce")
                out["baseline_total_cycles"] = generic_base_numeric.where(generic_base_numeric.notna(), deep_base_numeric)
            else:
                out["baseline_total_cycles"] = deep_base_numeric
            deep_speedup_numeric = pd.to_numeric(out["speedup_vs_deep_baseline"], errors="coerce")
            if "speedup_vs_baseline" in out.columns:
                generic_speedup_numeric = pd.to_numeric(out["speedup_vs_baseline"], errors="coerce")
                out["speedup_vs_baseline"] = generic_speedup_numeric.where(generic_speedup_numeric.notna(), deep_speedup_numeric)
            else:
                out["speedup_vs_baseline"] = deep_speedup_numeric
    return out


def _save_line(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    out: Path,
    y_log: bool = False,
) -> bool:
    """Save a line plot without letting the legend cover the data.

    This is the legacy family-level plotter.  It intentionally keeps the old
    output filenames, but moves large legends outside the axes so the generated
    figures remain readable.  The newer split-by-array/sequence plots are
    generated by _save_pretty_family_plots().
    """

    import matplotlib.pyplot as plt
    if "status" in df.columns:
        ok = df[df["status"] == "ok"].copy()
    else:
        ok = df.copy()
    if ok.empty or x not in ok.columns or y not in ok.columns or group not in ok.columns:
        if out.exists():
            out.unlink()
        return False
    ok = ok.dropna(subset=[x, y, group])
    if ok.empty:
        if out.exists():
            out.unlink()
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = ok[group].dropna().astype(str).unique()
    legend_outside = len(labels) > 8
    fig_width = 13 if legend_outside else 10
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    plotted = 0
    for label, g in ok.sort_values(x).groupby(group, dropna=True):
        if g.empty:
            continue
        # Repeated x-values are common in factorial sweeps; average them so the
        # line represents the trend instead of drawing vertical zig-zags.
        gg = g.groupby(x, as_index=False)[y].mean().sort_values(x)
        ax.plot(gg[x], gg[y], marker="o", linewidth=2, markersize=5, label=str(label))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        if out.exists():
            out.unlink()
        return False
    if y_log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend_outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize="small",
            ncol=1,
        )
        fig.tight_layout(rect=[0, 0, 0.78, 1])
        fig.savefig(out, dpi=160)
    else:
        ax.legend(fontsize="small", frameon=False)
        fig.tight_layout()
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return True

def _save_bar(df: pd.DataFrame, *, x: str, y: str, group: str, title: str, ylabel: str, out: Path) -> bool:
    import matplotlib.pyplot as plt
    if "status" in df.columns:
        ok = df[df["status"] == "ok"].copy()
    else:
        ok = df.copy()
    if ok.empty or x not in ok.columns or y not in ok.columns or group not in ok.columns:
        if out.exists():
            out.unlink()
        return False
    ok = ok.dropna(subset=[x, y, group])
    if ok.empty:
        if out.exists():
            out.unlink()
        return False
    pivot = ok.pivot_table(index=x, columns=group, values=y, aggfunc="mean")
    if pivot.empty:
        if out.exists():
            out.unlink()
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return True


def _compact_op_label(value: object) -> str:
    """Shorten operation names for plot legends."""

    text = str(value)
    replacements = [
        ("moe-gate-up-active", "moe-gate"),
        ("moe_down_active", "moe-down"),
        ("moe-down-active", "moe-down"),
        ("moe_gate_up_active", "moe-gate"),
        ("dense-gate-up", "gate-up"),
        ("dense_gate_up", "gate-up"),
        ("dense-down", "down"),
        ("dense_down", "down"),
        ("mlp-gate-up", "gate-up"),
        ("mlp_gate_up", "gate-up"),
        ("mlp-down", "down"),
        ("mlp_down", "down"),
        ("o_proj", "o-proj"),
        ("ffn1", "ffn"),
        ("patch_embed", "patch"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _infer_seq_bucket(*values: object) -> str:
    """Extract decode/sequence bucket labels such as decode1, seq2048, seq8192."""

    for value in values:
        text = str(value)
        match = re.search(r"(decode\d+|seq\d+|prefill\d+)", text)
        if match:
            return match.group(1)
    return "unknown"


def _add_pretty_plot_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact labels used only for human-friendly plot grouping."""

    out = df.copy()
    if "array_from_case" in out.columns:
        out["pretty_array"] = out["array_from_case"].astype(str)
    elif {"array_h", "array_w"}.issubset(out.columns):
        out["pretty_array"] = out["array_h"].astype("Int64").astype(str) + "x" + out["array_w"].astype("Int64").astype(str)
    else:
        out["pretty_array"] = "array"

    if "model_op_from_case" in out.columns:
        op_source = out["model_op_from_case"].astype(str)
    elif "op" in out.columns:
        op_source = out["op"].astype(str)
    else:
        op_source = out.get("case_name", pd.Series("op", index=out.index)).astype(str)

    out["pretty_seq"] = [
        _infer_seq_bucket(row.get("model_op_from_case", ""), row.get("model_case", ""), row.get("workload", ""), row.get("case_name", ""))
        for _, row in out.iterrows()
    ]
    # Remove the seq prefix from legend labels because it is already encoded in
    # the filename/title after splitting the plot.
    op_without_seq = op_source.str.replace(r"^(decode\d+|seq\d+|prefill\d+)-?", "", regex=True)
    out["pretty_op"] = op_without_seq.map(_compact_op_label)
    return out


def _save_pretty_lineplot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    out: Path,
    y_log: bool = False,
) -> bool:
    """Save a compact plot with only operation names in the legend."""

    import matplotlib.pyplot as plt

    if df.empty or x not in df.columns or y not in df.columns or "pretty_op" not in df.columns:
        if out.exists():
            out.unlink()
        return False
    ok = df.dropna(subset=[x, y, "pretty_op"]).copy()
    if ok.empty:
        if out.exists():
            out.unlink()
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    plotted = 0
    for label, g in ok.sort_values(x).groupby("pretty_op", dropna=True):
        if g.empty:
            continue
        # Multiple sweep axes can produce repeated x-values for the same op.
        # Use mean y-value per x so each operation remains one clean trend line.
        gg = g.groupby(x, as_index=False)[y].mean().sort_values(x)
        if gg.empty:
            continue
        ax.plot(gg[x], gg[y], marker="o", linewidth=2, markersize=5, label=str(label))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        if out.exists():
            out.unlink()
        return False

    if y_log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="op",
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return True


def _save_pretty_family_metric_grid(
    df: pd.DataFrame,
    *,
    family: str,
    x: str,
    y: str,
    title_tail: str,
    ylabel: str,
    out: Path,
    y_log: bool = False,
) -> bool:
    """Save one compact small-multiple figure for a family/metric.

    Rows are array sizes, columns are sequence buckets, and the legend contains
    only short operation names.  This keeps the plots readable without producing
    dozens of separate files.
    """

    import matplotlib.pyplot as plt

    if df.empty or x not in df.columns or y not in df.columns:
        if out.exists():
            out.unlink()
        return False

    work = df.dropna(subset=[x, y, "pretty_array", "pretty_seq", "pretty_op"]).copy()
    if work.empty:
        if out.exists():
            out.unlink()
        return False

    arrays = sorted(work["pretty_array"].dropna().astype(str).unique(), key=lambda v: (len(v), v))
    seqs = sorted(
        work["pretty_seq"].dropna().astype(str).unique(),
        key=lambda v: (0 if v.startswith("decode") else 1, int(re.search(r"\d+", v).group(0)) if re.search(r"\d+", v) else 0, v),
    )
    if not arrays or not seqs:
        return False

    fig_w = max(8.0, 3.8 * len(seqs) + 2.4)
    fig_h = max(4.0, 2.8 * len(arrays) + 1.0)
    fig, axes = plt.subplots(len(arrays), len(seqs), figsize=(fig_w, fig_h), squeeze=False, sharex=False, sharey=False)

    handles_by_label: dict[str, object] = {}
    plotted = 0
    for r, array_label in enumerate(arrays):
        for c, seq_bucket in enumerate(seqs):
            ax = axes[r][c]
            g = work[(work["pretty_array"].astype(str) == array_label) & (work["pretty_seq"].astype(str) == seq_bucket)]
            if g.empty:
                ax.set_axis_off()
                continue
            for op_label, gg in g.sort_values(x).groupby("pretty_op", dropna=True):
                # Repeated x-values are common in factorial sweeps.  Average them
                # so the trend line remains readable instead of zig-zagging.
                line_df = gg.groupby(x, as_index=False)[y].mean().sort_values(x)
                if line_df.empty:
                    continue
                (line,) = ax.plot(line_df[x], line_df[y], marker="o", linewidth=1.8, markersize=4, label=str(op_label))
                handles_by_label.setdefault(str(op_label), line)
                plotted += 1
            if y_log:
                ax.set_yscale("log")
            ax.set_title(f"{array_label}, {seq_bucket}", fontsize=10)
            ax.grid(True, alpha=0.28)
            if r == len(arrays) - 1:
                ax.set_xlabel(x, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)

    if plotted == 0:
        plt.close(fig)
        if out.exists():
            out.unlink()
        return False

    fig.suptitle(f"{family}: {title_tail}", fontsize=15)
    labels = sorted(handles_by_label)
    handles = [handles_by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=9,
        title="op",
    )
    fig.subplots_adjust(left=0.07, right=0.84, top=0.88, bottom=0.12, wspace=0.28, hspace=0.38)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def _save_pretty_family_plots(family: str, fam: pd.DataFrame, plot_dir: Path) -> tuple[Path, ...]:
    """Generate readable small-multiple plots for one model family.

    Each output file is split into panels by array size and sequence bucket.
    This replaces the old single-axis plots whose legends grew too large for
    large-model sweeps such as GPT-OSS and Qwen.
    """

    generated: list[Path] = []
    safe_family = _slug(str(family))
    pretty_dir = plot_dir / "pretty"

    work = _add_pretty_plot_tags(fam)
    if "tile_fits_sram" in work.columns:
        feasible = work[work["tile_fits_sram"].fillna(False)].copy()
        if not feasible.empty:
            work = feasible

    specs = [
        (
            {"tile", "factorial"},
            "tile_relation_ratio",
            "tile_efficiency_score",
            "tile relation ratio vs efficiency score",
            "efficiency score",
            "tileratio_vs_eff_grid",
            False,
        ),
        (
            {"tile", "factorial"},
            "tile_relation_ratio",
            "total_cycles",
            "tile relation ratio vs total cycles",
            "total cycles",
            "tileratio_vs_cycles_grid",
            True,
        ),
        (
            {"bandwidth", "factorial"},
            "bandwidth",
            "memory_overhead_ratio",
            "off-chip bandwidth vs memory overhead ratio",
            "(total-compute)/total",
            "bandwidth_vs_memover_grid",
            False,
        ),
        (
            {"sram", "factorial"},
            "sram_scale_from_case",
            "tile_efficiency_score",
            "SRAM scale vs efficiency score",
            "efficiency score",
            "sram_vs_eff_grid",
            False,
        ),
        (
            {"cache", "factorial"},
            "cache_bw_scale_from_case",
            "memory_overhead_ratio",
            "cache/SRAM-bank bandwidth vs memory overhead ratio",
            "(total-compute)/total",
            "cachebw_vs_memover_grid",
            False,
        ),
    ]

    for groups, x, y, title_tail, ylabel, tag, y_log in specs:
        if x not in work.columns or y not in work.columns or "deep_group" not in work.columns:
            continue
        base = work[work["deep_group"].isin(groups)].copy()
        if base.empty:
            continue
        p = pretty_dir / f"{safe_family}_{tag}.png"
        if _save_pretty_family_metric_grid(base, family=str(family), x=x, y=y, title_tail=title_tail, ylabel=ylabel, out=p, y_log=y_log):
            generated.append(p)
    return tuple(generated)


def make_deep_plots(df: pd.DataFrame, plot_dir: Path = DEFAULT_PLOT_DIR) -> tuple[Path, ...]:
    generated: list[Path] = []
    if df.empty:
        return tuple(generated)
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        return tuple(generated)

    # Canonical baseline slice for per-op/model comparison.
    default_bw = ok["bandwidth"].mode().iloc[0] if not ok["bandwidth"].mode().empty else ok["bandwidth"].max()
    default_df = ok[(ok["deep_group"].isin(["tile", "bandwidth", "sram", "cache"])) & (ok["sram_scale_from_case"] == 1.0) & (ok["cache_bw_scale_from_case"] == 1.0)]
    if default_df.empty:
        default_df = ok

    p = plot_dir / "model_operation_total_cycles.png"
    if _save_bar(default_df, x="workload_label", y="total_cycles", group="array_from_case", title="Model/op vs total cycles", ylabel="total cycles", out=p):
        generated.append(p)

    p = plot_dir / "model_operation_efficiency_score.png"
    if _save_bar(default_df, x="workload_label", y="tile_efficiency_score", group="array_from_case", title="Model/op vs tile efficiency score", ylabel="MACs/cycle/PE × (1-memory overhead)", out=p):
        generated.append(p)

    for family, fam in ok.groupby("model_family_from_case", dropna=True):
        generated.extend(_save_pretty_family_plots(str(family), fam, plot_dir))

    ranking_cols = [
        "case_name", "workload", "model_family_from_case", "model_op_from_case", "array_h", "array_w",
        "deep_group", "tile", "tile_m", "tile_n", "tile_k", "bandwidth", "sram_scale_from_case", "cache_bw_scale_from_case",
        "tile_fits_sram", "tile_working_set_kb", "total_cycles", "macs_per_cycle_per_pe",
        "memory_overhead_ratio", "tile_ai_proxy_ops_per_byte", "tile_efficiency_score",
        "baseline_available", "baseline_is_approx", "baseline_reason", "deep_baseline_total_cycles",
        "speedup_vs_deep_baseline", "baseline_total_cycles", "speedup_vs_baseline",
        "logical_tiles", "simulated_tiles", "topology_mode", "cache_status",
    ]
    ranking = ok.sort_values(["tile_efficiency_score", "macs_per_cycle_per_pe"], ascending=False)[[c for c in ranking_cols if c in ok.columns]]
    p = plot_dir / "ranking.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(p, index=False)
    generated.append(p)

    valid = ok[ok["tile_fits_sram"].fillna(False)].copy() if "tile_fits_sram" in ok.columns else ok.copy()
    valid_ranking = valid.sort_values(["tile_efficiency_score", "macs_per_cycle_per_pe"], ascending=False)[[c for c in ranking_cols if c in valid.columns]]
    p = plot_dir / "ranking_valid_only.csv"
    valid_ranking.to_csv(p, index=False)
    generated.append(p)

    latency = valid.sort_values(["total_cycles", "tile_efficiency_score"], ascending=[True, False])[[c for c in ranking_cols if c in valid.columns]]
    p = plot_dir / "ranking_latency_valid_only.csv"
    latency.to_csv(p, index=False)
    generated.append(p)

    for group_name, group_df in ok.groupby("deep_group", dropna=True):
        safe_group = _slug(str(group_name))
        gp = group_df[group_df["tile_fits_sram"].fillna(False)] if "tile_fits_sram" in group_df.columns else group_df
        if gp.empty:
            continue
        p = plot_dir / f"ranking_{safe_group}_valid_only.csv"
        gp.sort_values(["tile_efficiency_score", "macs_per_cycle_per_pe"], ascending=False)[[c for c in ranking_cols if c in gp.columns]].to_csv(p, index=False)
        generated.append(p)

    summary = ok.groupby(["model_family_from_case", "array_from_case"], dropna=True).agg(
        cases=("case_name", "count"),
        best_efficiency_score=("tile_efficiency_score", "max"),
        mean_efficiency_score=("tile_efficiency_score", "mean"),
        min_total_cycles=("total_cycles", "min"),
        mean_memory_overhead_ratio=("memory_overhead_ratio", "mean"),
        feasible_tiles=("tile_fits_sram", "sum"),
        baseline_available_cases=("baseline_available", "sum"),
        exact_baseline_cases=("baseline_is_approx", lambda x: (~x.fillna(True)).sum()),
    ).reset_index()
    p = plot_dir / "model_summary.csv"
    summary.to_csv(p, index=False)
    generated.append(p)
    return tuple(generated)


def _parse_csv_triplet(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").replace("/", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected ifmap,filter,ofmap SRAM triplet, e.g. 2048,4096,2048")
    vals = tuple(int(p.strip()) for p in parts)
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("SRAM values must be positive")
    return vals


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run broad TPUv2 grouped-full sweeps across SRAM/cache/tile sizes and DL model GEMMs")
    ap.add_argument("--preset", choices=tuple(MODEL_PRESETS) + ("all", "large_models"), action="append", default=None)
    ap.add_argument("--custom-workload-json", type=Path, action="append", default=None, help="load additional GEMM workloads from JSON")
    ap.add_argument("--base-cfg", type=Path, default=DEFAULT_TPUV2_CFG)
    ap.add_argument("--array", type=parse_array, action="append", default=None)
    ap.add_argument("--tile-mn-factor", type=parse_float, action="append", default=None, help="tile M/N as a multiple of array height/width")
    ap.add_argument("--tile-k", type=_positive_int, action="append", default=None)
    ap.add_argument("--bandwidth", type=_positive_int, action="append", default=None, help="off-chip interface bandwidth values")
    ap.add_argument("--sram-scale", type=parse_float, action="append", default=None, help="scale TPUv2 ifmap/filter/ofmap SRAM sizes")
    ap.add_argument("--cache-bw-scale", type=parse_float, action="append", default=None, help="scale on-chip SRAM-bank/cache bandwidth relative to off-chip bandwidth")
    ap.add_argument("--layout", type=Path, action="append", default=None)
    ap.add_argument("--dataflow", choices=("ws", "os", "is"), action="append", default=None)
    ap.add_argument("--mode", choices=("one-factor", "factorial"), default="one-factor")
    ap.add_argument("--custom-layout", action="store_true")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--enriched-output", type=Path, default=DEFAULT_ENRICHED_CSV)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--topology-mode", choices=("grouped_full", "raw"), default="grouped_full")
    ap.add_argument("--skip-iree", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--jobs", type=_positive_int, default=1)
    ap.add_argument("--parallel-backend", choices=("thread", "process"), default="thread")
    ap.add_argument("--limit", type=_positive_int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--heartbeat-sec", type=int, default=30)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    presets = tuple(args.preset) if args.preset else DEFAULT_PRESETS
    workloads = _expand_presets(presets)
    if args.custom_workload_json:
        workloads = workloads + tuple(w for path in args.custom_workload_json for w in _load_custom_workloads(path))
    arrays = tuple(args.array) if args.array else ((128, 128), (256, 256))
    tile_mn_factors = tuple(args.tile_mn_factor) if args.tile_mn_factor else (0.5, 1.0, 2.0, 4.0)
    tile_ks = tuple(args.tile_k) if args.tile_k else (64, 128, 256)
    bandwidths = tuple(args.bandwidth) if args.bandwidth else (1200, 600, 300, 150)
    sram_scales = tuple(args.sram_scale) if args.sram_scale else (1.0, 0.25, 0.0625, 0.015625)
    cache_bw_scales = tuple(args.cache_bw_scale) if args.cache_bw_scale else (1.0, 0.5, 0.25, 0.125)
    layouts = tuple(args.layout) if args.layout else (DEFAULT_LAYOUT,)
    dataflows = tuple(args.dataflow) if args.dataflow else ("ws",)

    cases = build_cases(
        workloads=workloads,
        base_cfg=args.base_cfg,
        arrays=arrays,
        tile_mn_factors=tile_mn_factors,
        tile_ks=tile_ks,
        bandwidths=bandwidths,
        sram_scales=sram_scales,
        cache_bw_scales=cache_bw_scales,
        layouts=layouts,
        dataflows=dataflows,
        mode=args.mode,
        custom_layout=args.custom_layout,
    )
    if args.limit is not None:
        cases = cases[: args.limit]

    preview = preview_cases(cases)
    if not args.quiet:
        print("Deep-model TPUv2 experiment configuration:", flush=True)
        print(f"  presets={','.join(presets)} workloads={len(workloads)} cases={len(cases)} mode={args.mode}", flush=True)
        print(f"  arrays={arrays} tile_mn_factors={tile_mn_factors} tile_ks={tile_ks}", flush=True)
        print(f"  bandwidths={bandwidths} sram_scales={sram_scales} cache_bw_scales={cache_bw_scales}", flush=True)
        print(f"  jobs={args.jobs} backend={args.parallel_backend} topology_mode={args.topology_mode} skip_iree={args.skip_iree}", flush=True)
        print(preview.head(min(16, len(preview))).to_string(index=False), flush=True)
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
    # First add generic TPUv2 metrics, then parse deep-model tags and add
    # deep-model-specific scores.  _add_deep_tags depends on metrics such as
    # macs_per_cycle_per_pe and memory_overhead_ratio.
    df = _add_deep_tags(enrich_metrics(df))
    write_dataframe_outputs(df, args.output, args.csv_output)
    args.enriched_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.enriched_output, index=False)
    print(f"wrote {args.output} ({len(df)} rows x {len(df.columns)} cols)")
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.enriched_output}")
    print(f"status counts: {df['status'].value_counts(dropna=False).to_dict()}")
    if not args.skip_plots:
        generated = make_deep_plots(df, args.plot_dir)
        if generated:
            print("generated plots/tables:")
            for path in generated:
                print(f"  {path}")


if __name__ == "__main__":
    main()
