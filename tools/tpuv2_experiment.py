"""TPUv2-oriented array/tile/bandwidth/SRAM experiment runner.

This script builds SCALE-Sim config variants from ``SCALE-Sim/configs/tpuv2.cfg``,
runs the existing MatMul MLIR -> topology -> SCALE-Sim pipeline, stores CSV/Parquet
results, and emits comparison plots.

Default mode is ``one-factor``: vary one hardware/tile knob at a time around the
TPUv2-like baseline.  Use ``--mode factorial`` when you intentionally want every
array x tile x bandwidth x SRAM combination.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from tools.config_factory import ArchConfigParams, generated_cfg_path, write_arch_cfg
from tools.io_utils import write_dataframe_outputs
from tools.sweep_runner import (
    DEFAULT_LAYOUT,
    RESULTS_DIR,
    MnkShape,
    SweepConfig,
    TileShape,
    build_configs,
    load_arch_spec,
    load_layout_spec,
    parse_mnk,
    run_sweep,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TPUV2_CFG = REPO / "SCALE-Sim" / "configs" / "tpuv2.cfg"
DEFAULT_OUTPUT = RESULTS_DIR / "tpuv2_experiment.parquet"
DEFAULT_CSV = RESULTS_DIR / "tpuv2_experiment.csv"
DEFAULT_OUTPUT_ROOT = REPO / "outputs" / "tpuv2_experiment"
DEFAULT_PLOT_DIR = REPO / "results" / "figures" / "tpuv2_experiment"
DEFAULT_CACHE_ROOT = REPO / "outputs" / "cache" / "tpuv2_experiment"

TILE_RELATION_FACTORS = {
    "smaller": 0.5,
    "equal": 1.0,
    "larger": 2.0,
}


@dataclass(frozen=True)
class GeneratedCase:
    group: str
    case_name: str
    mnk: MnkShape
    tile: TileShape
    array_h: int
    array_w: int
    bandwidth: int
    sram_scale: float
    ifmap_sram_kb: int
    filter_sram_kb: int
    ofmap_sram_kb: int
    arch_cfg: Path
    layout: Path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive: {value}")
    return parsed


def parse_array(value: str) -> tuple[int, int]:
    raw = value.lower().replace(",", "x").split("x")
    if len(raw) != 2:
        raise argparse.ArgumentTypeError("array must be formatted like 128x128")
    try:
        h, w = (int(x) for x in raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"array contains a non-integer: {value}") from exc
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError(f"array dimensions must be positive: {value}")
    return h, w


def parse_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected float, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive: {value}")
    return parsed


def _safe_float_label(value: float) -> str:
    return (f"{value:g}").replace(".", "p")


def _scaled_sram(base: int, scale: float) -> int:
    return max(1, int(round(base * scale)))


def _tile_for_relation(*, mnk: MnkShape, array_h: int, array_w: int, relation: str, tile_k: int | None) -> TileShape:
    factor = TILE_RELATION_FACTORS[relation]
    m = max(1, int(round(array_h * factor)))
    n = max(1, int(round(array_w * factor)))
    k = tile_k if tile_k is not None else array_w
    return TileShape(m=min(mnk.m, m), n=min(mnk.n, n), k=min(mnk.k, max(1, k)))


def _write_tpuv2_case_cfg(
    *,
    base_cfg: Path,
    run_name: str,
    array_h: int,
    array_w: int,
    bandwidth: int,
    ifmap_sram_kb: int,
    filter_sram_kb: int,
    ofmap_sram_kb: int,
    dataflow: str,
    use_custom_layouts: bool,
) -> Path:
    return write_arch_cfg(
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


def _make_case(
    *,
    group: str,
    base_cfg: Path,
    base_ifmap_sram_kb: int,
    base_filter_sram_kb: int,
    base_ofmap_sram_kb: int,
    mnk: MnkShape,
    array: tuple[int, int],
    relation: str,
    bandwidth: int,
    sram_scale: float,
    tile_k: int | None,
    layout: Path,
    dataflow: str,
    use_custom_layouts: bool,
) -> GeneratedCase:
    array_h, array_w = array
    tile = _tile_for_relation(mnk=mnk, array_h=array_h, array_w=array_w, relation=relation, tile_k=tile_k)
    ifmap_sram_kb = _scaled_sram(base_ifmap_sram_kb, sram_scale)
    filter_sram_kb = _scaled_sram(base_filter_sram_kb, sram_scale)
    ofmap_sram_kb = _scaled_sram(base_ofmap_sram_kb, sram_scale)
    case_name = (
        f"{group}_a{array_h}x{array_w}_tile-{relation}_{tile.label}"
        f"_bw{bandwidth}_sram{_safe_float_label(sram_scale)}"
    )
    run_name = f"tpuv2_{case_name}_{dataflow}"
    arch_cfg = _write_tpuv2_case_cfg(
        base_cfg=base_cfg,
        run_name=run_name,
        array_h=array_h,
        array_w=array_w,
        bandwidth=bandwidth,
        ifmap_sram_kb=ifmap_sram_kb,
        filter_sram_kb=filter_sram_kb,
        ofmap_sram_kb=ofmap_sram_kb,
        dataflow=dataflow,
        use_custom_layouts=use_custom_layouts,
    )
    return GeneratedCase(
        group=group,
        case_name=case_name,
        mnk=mnk,
        tile=tile,
        array_h=array_h,
        array_w=array_w,
        bandwidth=bandwidth,
        sram_scale=sram_scale,
        ifmap_sram_kb=ifmap_sram_kb,
        filter_sram_kb=filter_sram_kb,
        ofmap_sram_kb=ofmap_sram_kb,
        arch_cfg=arch_cfg,
        layout=layout,
    )


def build_tpuv2_cases(
    *,
    mnks: Iterable[MnkShape],
    arrays: Iterable[tuple[int, int]],
    tile_relations: Iterable[str],
    bandwidths: Iterable[int],
    sram_scales: Iterable[float],
    base_cfg: Path = DEFAULT_TPUV2_CFG,
    layouts: Iterable[Path] = (DEFAULT_LAYOUT,),
    dataflow: str = "ws",
    tile_k: int | None = None,
    mode: str = "one-factor",
    use_custom_layouts: bool = False,
) -> tuple[GeneratedCase, ...]:
    """Generate TPUv2 experiment cases.

    ``one-factor`` varies tile relation, bandwidth, and SRAM scale independently
    around the baseline relation=equal, bandwidth=base TPUv2 bandwidth, sram=1.0.
    ``factorial`` produces every combination.
    """

    base_arch = load_arch_spec(base_cfg)
    base_bandwidth = base_arch.bandwidth
    base_ifmap = base_arch.ifmap_sram_kb
    base_filter = base_arch.filter_sram_kb
    base_ofmap = base_arch.ofmap_sram_kb
    layout_paths = tuple(layouts)
    cases: list[GeneratedCase] = []
    seen: set[tuple[str, str, str, int, int, int, float, str]] = set()

    def append(
        *,
        group: str,
        mnk: MnkShape,
        array: tuple[int, int],
        relation: str,
        bandwidth: int,
        sram_scale: float,
        layout: Path,
    ) -> None:
        case = _make_case(
            group=group,
            base_cfg=base_cfg,
            base_ifmap_sram_kb=base_ifmap,
            base_filter_sram_kb=base_filter,
            base_ofmap_sram_kb=base_ofmap,
            mnk=mnk,
            array=array,
            relation=relation,
            bandwidth=bandwidth,
            sram_scale=sram_scale,
            tile_k=tile_k,
            layout=layout,
            dataflow=dataflow,
            use_custom_layouts=use_custom_layouts,
        )
        key = (case.group, case.mnk.label, case.tile.label, case.array_h, case.array_w, case.bandwidth, case.sram_scale, str(case.layout))
        if key not in seen:
            seen.add(key)
            cases.append(case)

    for mnk in mnks:
        for array in arrays:
            for layout in layout_paths:
                if mode == "factorial":
                    for relation in tile_relations:
                        for bandwidth in bandwidths:
                            for sram_scale in sram_scales:
                                append(
                                    group="factorial",
                                    mnk=mnk,
                                    array=array,
                                    relation=relation,
                                    bandwidth=bandwidth,
                                    sram_scale=sram_scale,
                                    layout=layout,
                                )
                else:
                    for relation in tile_relations:
                        append(
                            group="tile",
                            mnk=mnk,
                            array=array,
                            relation=relation,
                            bandwidth=base_bandwidth,
                            sram_scale=1.0,
                            layout=layout,
                        )
                    for bandwidth in bandwidths:
                        append(
                            group="bandwidth",
                            mnk=mnk,
                            array=array,
                            relation="equal",
                            bandwidth=bandwidth,
                            sram_scale=1.0,
                            layout=layout,
                        )
                    for sram_scale in sram_scales:
                        append(
                            group="sram",
                            mnk=mnk,
                            array=array,
                            relation="equal",
                            bandwidth=base_bandwidth,
                            sram_scale=sram_scale,
                            layout=layout,
                        )
    return tuple(cases)


def cases_to_sweep_configs(cases: Iterable[GeneratedCase]) -> tuple[SweepConfig, ...]:
    configs = []
    for case in cases:
        arch = load_arch_spec(case.arch_cfg)
        layout = load_layout_spec(case.layout)
        configs.append(
            SweepConfig(
                mnk=case.mnk,
                tile_shape=case.tile,
                arch_spec=arch,
                layout_spec=layout,
                case_name=case.case_name,
            )
        )
    return tuple(configs)


def cases_preview(cases: Iterable[GeneratedCase]) -> pd.DataFrame:
    rows = []
    for idx, case in enumerate(cases, start=1):
        rows.append(
            {
                "idx": idx,
                "group": case.group,
                "case_name": case.case_name,
                "mnk": case.mnk.label,
                "array": f"{case.array_h}x{case.array_w}",
                "tile": case.tile.label,
                "bandwidth": case.bandwidth,
                "sram_scale": case.sram_scale,
                "ifmap_sram_kb": case.ifmap_sram_kb,
                "filter_sram_kb": case.filter_sram_kb,
                "ofmap_sram_kb": case.ofmap_sram_kb,
                "layout": case.layout.stem,
                "arch_cfg": str(case.arch_cfg.relative_to(REPO) if case.arch_cfg.is_relative_to(REPO) else case.arch_cfg),
            }
        )
    return pd.DataFrame(rows)


def enrich_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add comparison metrics that are useful for plotting and ranking."""

    out = df.copy()
    if out.empty:
        return out

    out["macs"] = out["m"] * out["n"] * out["k"]
    out["array_area"] = out["array_h"] * out["array_w"]
    out["tile_area"] = out["tile_m"] * out["tile_n"]
    out["tile_relation_ratio"] = out["tile_area"] / out["array_area"].replace(0, pd.NA)
    out["cycles_per_mac"] = out["total_cycles"] / out["macs"].replace(0, pd.NA)
    out["macs_per_cycle"] = out["macs"] / out["total_cycles"].replace(0, pd.NA)
    out["macs_per_cycle_per_pe"] = out["macs_per_cycle"] / out["array_area"].replace(0, pd.NA)
    out["cycles_per_mac_per_pe"] = out["cycles_per_mac"] * out["array_area"]
    if "memory_overhead_cycles" not in out.columns:
        out["memory_overhead_cycles"] = (out["total_cycles"] - out["compute_cycles"]).clip(lower=0)
    if "memory_overhead_ratio" not in out.columns:
        out["memory_overhead_ratio"] = out["memory_overhead_cycles"] / out["total_cycles"].replace(0, pd.NA)
    out["compute_cycles_per_mac"] = out["compute_cycles"] / out["macs"].replace(0, pd.NA)
    out["overhead_per_tile"] = out["memory_overhead_cycles"] / out["n_tiles"].replace(0, pd.NA)
    out["stall_per_tile"] = out["stall"] / out["n_tiles"].replace(0, pd.NA)

    baseline_mask = (
        (out["status"] == "ok")
        & (out["case_name"].str.contains("tile-equal", regex=False))
        & (out["sram_scale_from_case"] == 1.0 if "sram_scale_from_case" in out.columns else True)
    )
    if "group_from_case" in out.columns:
        baseline_mask &= out["group_from_case"].isin(["tile", "bandwidth", "sram", "factorial"])

    group_cols = ["workload", "m", "n", "k", "array_h", "array_w", "layout", "dataflow"]
    # Add a generic baseline when one is available.  Deep-model sweeps use
    # numeric tile factors rather than the older ``tile-equal`` naming scheme,
    # so the generic baseline may legitimately be empty.  Keep the columns
    # present anyway so downstream plot/ranking code can run on every result CSV.
    out["baseline_total_cycles"] = pd.NA
    out["speedup_vs_baseline"] = pd.NA
    try:
        candidate = baseline_mask & (out["bandwidth"] == out.groupby(group_cols)["bandwidth"].transform("max"))
        baseline = (
            out.loc[candidate, group_cols + ["total_cycles"]]
            .sort_values(group_cols + ["total_cycles"])
            .drop_duplicates(group_cols)
            .rename(columns={"total_cycles": "baseline_total_cycles"})
        )
        if not baseline.empty and "baseline_total_cycles" in baseline.columns:
            out = out.drop(columns=["baseline_total_cycles"], errors="ignore").merge(
                baseline[group_cols + ["baseline_total_cycles"]],
                on=group_cols,
                how="left",
            )
            out["speedup_vs_baseline"] = out["baseline_total_cycles"] / out["total_cycles"].replace(0, pd.NA)
    except Exception:
        # Baseline speedup is optional; never make plotting fail because a
        # particular experiment family does not contain the canonical baseline.
        out["baseline_total_cycles"] = pd.NA
        out["speedup_vs_baseline"] = pd.NA
    return out


def _extract_case_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    extracted = out["case_name"].str.extract(
        r"^(?P<group_from_case>[^_]+)_a(?P<array_from_case>\d+x\d+)_tile-(?P<tile_relation>[^_]+)_(?P<tile_from_case>\d+x\d+x\d+)_bw(?P<bandwidth_from_case>\d+)_sram(?P<sram_label>[0-9p]+)"
    )
    for col in extracted.columns:
        out[col] = extracted[col]
    out["sram_scale_from_case"] = out["sram_label"].str.replace("p", ".", regex=False).astype(float)

    # LLM case groups look like:
    #   llm-llama7b-prefill-2048-qkv_a128x128_tile-equal_...
    # Preserve the original generic tags, but also expose preset/op columns so
    # the plotting code can make workload-aware figures.
    llm = out["group_from_case"].astype(str).str.extract(
        r"^llm-(?P<llm_preset_from_case>.*)-(?P<llm_op_from_case>qkv|o-proj|gate-up|down)$"
    )
    out["llm_preset_from_case"] = llm["llm_preset_from_case"].str.replace("-", "_", regex=False)
    out["llm_op_from_case"] = llm["llm_op_from_case"].str.replace("-", "_", regex=False)
    return out


def _save_line_plot(df: pd.DataFrame, *, x: str, y: str, group: str, title: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"].copy()
    if ok.empty or x not in ok or y not in ok or group not in ok:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for label, g in ok.sort_values(x).groupby(group):
        plt.plot(g[x], g[y], marker="o", label=str(label))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_grouped_bar_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"].copy()
    if ok.empty or x not in ok or y not in ok or group not in ok:
        return
    pivot = ok.pivot_table(index=x, columns=group, values=y, aggfunc="mean")
    if pivot.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _mode_or_first(series: pd.Series, fallback=None):
    clean = series.dropna()
    if clean.empty:
        return fallback
    mode = clean.mode()
    if not mode.empty:
        return mode.iloc[0]
    return clean.iloc[0]


def _make_llm_plots(ok: pd.DataFrame, plot_dir: Path) -> list[Path]:
    """Generate LLM-specific plots.

    The normal TPUv2 plots depend on group labels named tile/bandwidth/sram.
    LLM cases instead use group labels such as llm-llama7b-prefill-2048-qkv,
    so the generic plot filters intentionally skip them. This helper derives
    the sweep axis from parsed case tags and creates workload-aware figures.
    """

    generated: list[Path] = []
    if "group_from_case" not in ok.columns:
        return generated
    llm = ok[ok["group_from_case"].astype(str).str.startswith("llm-")].copy()
    if llm.empty:
        return generated

    if "llm_preset_from_case" not in llm.columns or llm["llm_preset_from_case"].isna().all():
        parsed = llm["group_from_case"].astype(str).str.extract(
            r"^llm-(?P<llm_preset_from_case>.*)-(?P<llm_op_from_case>qkv|o-proj|gate-up|down)$"
        )
        llm["llm_preset_from_case"] = parsed["llm_preset_from_case"].str.replace("-", "_", regex=False)
        llm["llm_op_from_case"] = parsed["llm_op_from_case"].str.replace("-", "_", regex=False)

    llm["llm_series"] = llm["array_from_case"].astype(str) + " " + llm["llm_op_from_case"].astype(str)
    default_bw = _mode_or_first(llm.loc[llm["sram_scale_from_case"] == 1.0, "bandwidth"], fallback=_mode_or_first(llm["bandwidth"]))
    default_sram = 1.0

    for preset, preset_df in llm.groupby("llm_preset_from_case", dropna=True):
        safe_preset = str(preset).replace("/", "_").replace(" ", "_")
        default_df = preset_df[
            (preset_df["tile_relation"] == "equal")
            & (preset_df["bandwidth"] == default_bw)
            & (preset_df["sram_scale_from_case"] == default_sram)
        ]
        if default_df.empty:
            default_df = preset_df[preset_df["tile_relation"] == "equal"]

        p = plot_dir / f"llm_{safe_preset}_op_total_cycles.png"
        _save_grouped_bar_plot(
            default_df,
            x="llm_op_from_case",
            y="total_cycles",
            group="array_from_case",
            title=f"{preset}: operation vs total cycles",
            ylabel="total cycles",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

        p = plot_dir / f"llm_{safe_preset}_op_pe_normalized_throughput.png"
        _save_grouped_bar_plot(
            default_df,
            x="llm_op_from_case",
            y="macs_per_cycle_per_pe",
            group="array_from_case",
            title=f"{preset}: operation vs PE-normalized throughput",
            ylabel="MACs / cycle / PE",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

        tile_df = preset_df[
            (preset_df["bandwidth"] == default_bw)
            & (preset_df["sram_scale_from_case"] == default_sram)
        ]
        p = plot_dir / f"llm_{safe_preset}_tile_relation_total_cycles.png"
        _save_line_plot(
            tile_df,
            x="tile_relation_ratio",
            y="total_cycles",
            group="llm_series",
            title=f"{preset}: tile size relative to array vs total cycles",
            ylabel="total cycles",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

        p = plot_dir / f"llm_{safe_preset}_tile_relation_pe_normalized_throughput.png"
        _save_line_plot(
            tile_df,
            x="tile_relation_ratio",
            y="macs_per_cycle_per_pe",
            group="llm_series",
            title=f"{preset}: tile size relative to array vs PE-normalized throughput",
            ylabel="MACs / cycle / PE",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

        bw_df = preset_df[
            (preset_df["tile_relation"] == "equal")
            & (preset_df["sram_scale_from_case"] == default_sram)
        ]
        p = plot_dir / f"llm_{safe_preset}_bandwidth_total_cycles.png"
        _save_line_plot(
            bw_df,
            x="bandwidth",
            y="total_cycles",
            group="llm_series",
            title=f"{preset}: bandwidth vs total cycles",
            ylabel="total cycles",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

        p = plot_dir / f"llm_{safe_preset}_bandwidth_memory_overhead_ratio.png"
        _save_line_plot(
            bw_df,
            x="bandwidth",
            y="memory_overhead_ratio",
            group="llm_series",
            title=f"{preset}: bandwidth vs memory overhead ratio",
            ylabel="(total - compute) / total",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

        sram_df = preset_df[
            (preset_df["tile_relation"] == "equal")
            & (preset_df["bandwidth"] == default_bw)
        ]
        p = plot_dir / f"llm_{safe_preset}_sram_total_cycles.png"
        _save_line_plot(
            sram_df,
            x="sram_scale_from_case",
            y="total_cycles",
            group="llm_series",
            title=f"{preset}: SRAM scale vs total cycles",
            ylabel="total cycles",
            out_path=p,
        )
        if p.exists():
            generated.append(p)

    return generated


def make_plots(df: pd.DataFrame, plot_dir: Path = DEFAULT_PLOT_DIR) -> tuple[Path, ...]:
    """Create a small set of comparison plots and return the generated files."""

    generated: list[Path] = []
    if df.empty:
        return tuple(generated)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return tuple(generated)

    # LLM workloads use case groups like llm-llama7b-prefill-2048-qkv rather
    # than the generic tile/bandwidth/sram groups, so generate their workload-
    # aware figures before the standard TPUv2 plots.
    generated.extend(_make_llm_plots(ok, plot_dir))

    tile_df = ok[ok["group_from_case"].isin(["tile", "factorial"])]
    p = plot_dir / "tile_relation_total_cycles.png"
    _save_line_plot(
        tile_df,
        x="tile_relation_ratio",
        y="total_cycles",
        group="array_from_case",
        title="Tile size relative to array vs total cycles",
        ylabel="total cycles",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    p = plot_dir / "tile_relation_compute_util.png"
    _save_line_plot(
        tile_df,
        x="tile_relation_ratio",
        y="mean_compute_util_pct",
        group="array_from_case",
        title="Tile size relative to array vs compute utilization",
        ylabel="mean compute utilization (%)",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    p = plot_dir / "tile_relation_macs_per_cycle_per_pe.png"
    _save_line_plot(
        tile_df,
        x="tile_relation_ratio",
        y="macs_per_cycle_per_pe",
        group="array_from_case",
        title="Tile size relative to array vs PE-normalized throughput",
        ylabel="MACs / cycle / PE",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    bw_df = ok[ok["group_from_case"].isin(["bandwidth", "factorial"])]
    p = plot_dir / "bandwidth_total_cycles.png"
    _save_line_plot(
        bw_df,
        x="bandwidth",
        y="total_cycles",
        group="array_from_case",
        title="Memory bandwidth vs total cycles",
        ylabel="total cycles",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    p = plot_dir / "bandwidth_memory_overhead_cycles.png"
    _save_line_plot(
        bw_df,
        x="bandwidth",
        y="memory_overhead_cycles",
        group="array_from_case",
        title="Memory bandwidth vs memory overhead cycles",
        ylabel="total_cycles - compute_cycles",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    p = plot_dir / "bandwidth_memory_overhead_ratio.png"
    _save_line_plot(
        bw_df,
        x="bandwidth",
        y="memory_overhead_ratio",
        group="array_from_case",
        title="Memory bandwidth vs memory overhead ratio",
        ylabel="(total - compute) / total",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    p = plot_dir / "bandwidth_stall.png"
    _save_line_plot(
        bw_df,
        x="bandwidth",
        y="stall",
        group="array_from_case",
        title="Memory bandwidth vs SCALE-Sim stall cycles",
        ylabel="stall cycles",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    sram_df = ok[ok["group_from_case"].isin(["sram", "factorial"])]
    p = plot_dir / "sram_total_cycles.png"
    _save_line_plot(
        sram_df,
        x="sram_scale_from_case",
        y="total_cycles",
        group="array_from_case",
        title="SRAM scale vs total cycles",
        ylabel="total cycles",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    p = plot_dir / "sram_compute_util.png"
    _save_line_plot(
        sram_df,
        x="sram_scale_from_case",
        y="mean_compute_util_pct",
        group="array_from_case",
        title="SRAM scale vs compute utilization",
        ylabel="mean compute utilization (%)",
        out_path=p,
    )
    if p.exists():
        generated.append(p)

    # Ranking table by the most useful normalized metric.
    ranking = ok.sort_values(["macs_per_cycle", "mean_compute_util_pct"], ascending=False)[
        [
            "case_name",
            "workload",
            "array_h",
            "array_w",
            "tile",
            "bandwidth",
            "sram_scale_from_case",
            "total_cycles",
            "mean_compute_util_pct",
            "stall",
            "memory_overhead_cycles",
            "memory_overhead_ratio",
            "cycles_per_mac",
            "macs_per_cycle",
            "macs_per_cycle_per_pe",
            "speedup_vs_baseline",
        ]
    ]
    ranking_path = plot_dir / "ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    generated.append(ranking_path)
    return tuple(generated)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TPUv2 array/tile/bandwidth/SRAM experiments and plot results")
    ap.add_argument("--base-cfg", type=Path, default=DEFAULT_TPUV2_CFG)
    ap.add_argument("--mnk", type=parse_mnk, action="append", default=None, help="GEMM shape, e.g. 1024x1024x1024. May be repeated")
    ap.add_argument("--array", type=parse_array, action="append", default=None, help="array shape, e.g. 64x64. May be repeated")
    ap.add_argument("--tile-relation", choices=tuple(TILE_RELATION_FACTORS), action="append", default=None)
    ap.add_argument("--tile-k", type=_positive_int, default=None, help="fixed tile K. Default: array width")
    ap.add_argument("--bandwidth", type=_positive_int, action="append", default=None, help="memory bandwidth value. May be repeated")
    ap.add_argument("--sram-scale", type=parse_float, action="append", default=None, help="scale TPUv2 base SRAM by this factor. May be repeated")
    ap.add_argument("--layout", type=Path, action="append", default=None, help="SCALE-Sim layout CSV. May be repeated")
    ap.add_argument("--dataflow", default="ws", choices=("ws", "os", "is"))
    ap.add_argument("--mode", choices=("one-factor", "factorial"), default="one-factor")
    ap.add_argument("--custom-layout", action="store_true", help="enable SCALE-Sim custom layout flags in generated configs")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print generated cases without running IREE/SCALE-Sim")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--limit", type=_positive_int, default=None, help="run at most N cases after generation")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--jobs", type=_positive_int, default=1, help="parallel SCALE-Sim workers; keep this below available CPU cores/RAM limits")
    ap.add_argument("--parallel-backend", choices=("thread", "process"), default="thread", help="thread is safer in Docker; process may be faster but can use much more memory")
    ap.add_argument("--resume", action="store_true", help="reuse existing COMPUTE_REPORT.csv files and only run missing cases")
    ap.add_argument("--skip-iree", action="store_true", help="skip iree-opt; generate topology directly from MNK/tile spec for much faster architectural sweeps")
    ap.add_argument("--topology-mode", choices=("grouped_full", "raw"), default="grouped_full", help="TPUv2 default is grouped_full: derive the full topology, group identical tile rows, and weight aggregate metrics. Use raw only for small validation subsets.")
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT, help="global cache for topology and SCALE-Sim reports; case_name is intentionally ignored in cache keys")
    ap.add_argument("--no-cache", action="store_true", help="disable global cache and use per-case output directories only")
    ap.add_argument("--quiet", action="store_true", help="suppress progress logs from the sweep runner")
    ap.add_argument("--heartbeat-sec", type=int, default=30, help="print SIM-RUNNING heartbeat every N seconds while a SCALE-Sim case is running; use 0 to disable")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    mnks = tuple(args.mnk) if args.mnk else (MnkShape(1024, 1024, 1024),)
    arrays = tuple(args.array) if args.array else ((64, 64), (128, 128), (256, 256))
    tile_relations = tuple(args.tile_relation) if args.tile_relation else ("smaller", "equal", "larger")
    bandwidths = tuple(args.bandwidth) if args.bandwidth else (1200, 600, 300, 150, 75)
    sram_scales = tuple(args.sram_scale) if args.sram_scale else (0.25, 0.5, 1.0, 2.0)
    layouts = tuple(args.layout) if args.layout else (DEFAULT_LAYOUT,)

    cases = build_tpuv2_cases(
        mnks=mnks,
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
    if not cases:
        raise SystemExit("no TPUv2 experiment cases selected")

    preview = cases_preview(cases)
    if not args.quiet:
        print("TPUv2 experiment configuration:", flush=True)
        print(f"  cases={len(cases)} mode={args.mode} jobs={args.jobs} backend={args.parallel_backend} heartbeat_sec={args.heartbeat_sec}", flush=True)
        print(f"  resume={args.resume} skip_iree={args.skip_iree} topology_mode={args.topology_mode} clean={not args.no_clean and not args.resume}", flush=True)
        print(f"  output_root={args.output_root}", flush=True)
        print(f"  cache_root={None if args.no_cache else args.cache_root}", flush=True)
        print("  first cases:", flush=True)
        print(preview.head(min(8, len(preview))).to_string(index=False), flush=True)
    if args.dry_run:
        print(f"dry-run: generated {len(preview)} TPUv2 case(s); no IREE/SCALE-Sim execution")
        print(preview.to_string(index=False))
        return

    configs = cases_to_sweep_configs(cases)
    df = run_sweep(
        configs=configs,
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
    status_counts = df["status"].value_counts(dropna=False).to_dict()
    print(f"status counts: {status_counts}")

    if not args.skip_plots:
        generated = make_plots(df, args.plot_dir)
        if generated:
            print("generated plots/tables:")
            for path in generated:
                try:
                    print(f"  {path.relative_to(REPO)}")
                except ValueError:
                    print(f"  {path}")
        else:
            print("no plots generated because there were no ok rows")


if __name__ == "__main__":
    main()
