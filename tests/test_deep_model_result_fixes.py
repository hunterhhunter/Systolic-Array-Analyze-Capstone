from __future__ import annotations

import json

import pandas as pd

from tools.deep_model_tpuv2_experiment import _add_deep_tags, _expand_presets, _load_custom_workloads, make_deep_plots
from tools.tpuv2_experiment import enrich_metrics


def _row(case_name: str, total_cycles: int, tile_m: int = 1, tile_n: int = 256, tile_k: int = 256):
    return {
        "workload": "m1_n12288_k4096",
        "m": 1,
        "n": 12288,
        "k": 4096,
        "tile": f"{tile_m}x{tile_n}x{tile_k}",
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "case_name": case_name,
        "arch": "arch",
        "arch_cfg": "cfg",
        "array_h": 256,
        "array_w": 256,
        "bandwidth": 600,
        "ifmap_sram_kb": 4096,
        "filter_sram_kb": 4096,
        "ofmap_sram_kb": 4096,
        "dataflow": "ws",
        "layout": "layout.csv",
        "layout_path": "layout.csv",
        "ifmap_custom_layout": False,
        "filter_custom_layout": False,
        "input_mlir_path": "",
        "tiled_mlir_path": "",
        "topology_path": "",
        "sim_dir": "",
        "compute_report_path": "",
        "stdout_path": "",
        "stderr_path": "",
        "error_stage": pd.NA,
        "error_message": pd.NA,
        "n_tiles": 1,
        "compute_cycles": total_cycles - 10,
        "total_cycles": total_cycles,
        "reuse_aware_cycles": total_cycles,
        "reuse_aware_cycles_est": total_cycles,
        "reuse_model": "test",
        "reuse_model_calibrated": False,
        "reuse_fold_fraction": 0.5,
        "mean_overall_util_pct": 1.0,
        "mean_mapping_eff_pct": 1.0,
        "mean_compute_util_pct": 1.0,
        "stall": 0,
        "memory_overhead_cycles": 10,
        "memory_overhead_ratio": 10 / total_cycles,
        "logical_tiles": 1,
        "simulated_tiles": 1,
        "raw_topology_rows": 1,
        "topology_mode": "grouped_full",
        "cache_key": "key",
        "cache_status": "miss",
        "status": "ok",
    }


def test_large_model_presets_include_gpt_oss_and_qwen_shapes():
    workloads = _expand_presets(["large_models"])
    labels = {(w.preset, w.model, w.op, w.m, w.n, w.k) for w in workloads}
    assert any(p == "gpt_oss120b" and op == "router" and n == 128 for p, _, op, _, n, _ in labels)
    assert any(p == "qwen35_27b" and op == "qkv" and n == 8192 and k == 5120 for p, _, op, _, n, k in labels)
    assert any(p == "qwen3_235b_a22b" and op == "moe_gate_up_active" and n == 24576 for p, _, op, _, n, _ in labels)


def test_custom_workload_json_loader(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"workloads": [{"model": "custom_model", "op": "proj", "m": 7, "n": 11, "k": 13}]}), encoding="utf-8")
    rows = _load_custom_workloads(path)
    assert len(rows) == 1
    assert rows[0].model == "custom_model"
    assert rows[0].mnk.label == "m7_n11_k13"


def test_deep_baseline_falls_back_for_decode_like_m1_rows():
    df = pd.DataFrame([
        _row("model-llama7b-decode1-qkv_tile_a256x256_tile-1_1x256x256_bw600_sram1_cache1_ws", 1000),
        _row("model-llama7b-decode1-qkv_tile_a256x256_tile-2_1x512x256_bw600_sram1_cache1_ws", 900, tile_n=512),
    ])
    out = _add_deep_tags(enrich_metrics(df))
    assert out["baseline_available"].all()
    assert out["baseline_is_approx"].all()
    assert out["speedup_vs_baseline"].notna().all()
    assert out["baseline_reason"].eq("closest valid tile baseline").all()


def test_make_deep_plots_writes_valid_and_latency_rankings(tmp_path):
    df = pd.DataFrame([
        _row("model-qwen35-27b-seq4096-qkv_tile_a256x256_tile-1_256x256x256_bw600_sram1_cache1_ws", 1000, tile_m=256),
        _row("model-qwen35-27b-seq4096-qkv_sram_a256x256_tile-1_256x256x256_bw600_sram0p015625_cache1_ws", 900, tile_m=256),
    ])
    out = _add_deep_tags(enrich_metrics(df))
    generated = make_deep_plots(out, tmp_path)
    assert tmp_path.joinpath("ranking_valid_only.csv").exists()
    assert tmp_path.joinpath("ranking_latency_valid_only.csv").exists()
    assert tmp_path.joinpath("ranking_tile_valid_only.csv").exists()
    assert tmp_path.joinpath("model_summary.csv").exists()
    assert tmp_path.joinpath("ranking_valid_only.csv") in generated
