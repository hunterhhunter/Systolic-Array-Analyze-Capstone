"""Helpers for generating SCALE-Sim architecture configs for boundary sweeps.

The existing repo mostly uses hand-written ``SCALE-Sim/configs/*.cfg`` files.
This module makes boundary experiments reproducible by deriving small config
variants from a known-good base config.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CFG = REPO / "SCALE-Sim" / "configs" / "walking_8x8_ws.cfg"
GENERATED_CFG_DIR = REPO / "outputs" / "generated_configs"


@dataclass(frozen=True)
class ArchConfigParams:
    """Mutable architecture knobs that are useful for boundary tests."""

    run_name: str
    array_h: int = 8
    array_w: int = 8
    ifmap_sram_kb: int = 64
    filter_sram_kb: int = 64
    ofmap_sram_kb: int = 64
    bandwidth: int = 10
    dataflow: str = "ws"
    ifmap_custom_layout: bool = False
    filter_custom_layout: bool = False
    ifmap_bank_bw: int | None = None
    filter_bank_bw: int | None = None


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def write_arch_cfg(
    *,
    base_cfg: Path = DEFAULT_BASE_CFG,
    out_cfg: Path,
    params: ArchConfigParams,
) -> Path:
    """Write a SCALE-Sim config variant and return its absolute path."""

    base_cfg = repo_path(base_cfg)
    out_cfg = repo_path(out_cfg)
    cp = configparser.ConfigParser()
    cp.optionxform = str
    if not cp.read(base_cfg):
        raise FileNotFoundError(base_cfg)

    cp["general"]["run_name"] = params.run_name

    arch = cp["architecture_presets"]
    arch["ArrayHeight"] = str(params.array_h)
    arch["ArrayWidth"] = str(params.array_w)
    arch["IfmapSramSzkB"] = str(params.ifmap_sram_kb)
    arch["FilterSramSzkB"] = str(params.filter_sram_kb)
    arch["OfmapSramSzkB"] = str(params.ofmap_sram_kb)
    arch["Bandwidth"] = str(params.bandwidth)
    arch["Dataflow"] = params.dataflow

    if "layout" not in cp:
        cp["layout"] = {}
    layout = cp["layout"]
    layout["IfmapCustomLayout"] = str(params.ifmap_custom_layout)
    layout["FilterCustomLayout"] = str(params.filter_custom_layout)
    layout["IfmapSRAMBankBandwidth"] = str(params.ifmap_bank_bw or params.bandwidth)
    layout["FilterSRAMBankBandwidth"] = str(params.filter_bank_bw or params.bandwidth)

    # Keep bandwidth controlled by explicit config values during boundary tests.
    cp["run_presets"]["InterfaceBandwidth"] = "USER"

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with out_cfg.open("w", encoding="utf-8") as f:
        cp.write(f)
    return out_cfg


def generated_cfg_path(run_name: str) -> Path:
    return GENERATED_CFG_DIR / f"{run_name}.cfg"
