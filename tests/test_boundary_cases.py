"""Fast boundary-case generator tests."""

from __future__ import annotations

from tools.boundary_cases import build_boundary_configs, default_sram_tile, estimate_tile_sram_kb, tile_smaller_equal_larger
from tools.sweep_runner import MnkShape, TileShape


def test_tile_boundary_shapes_for_8x8_array():
    cases = dict(tile_smaller_equal_larger(8, 8, 8))
    assert cases["tile_smaller_than_array"] == TileShape(4, 4, 8)
    assert cases["tile_equal_to_array"] == TileShape(8, 8, 8)
    assert cases["tile_larger_than_array"] == TileShape(16, 16, 8)


def test_estimate_tile_sram_kb_rounds_up_to_at_least_one_kb():
    assert estimate_tile_sram_kb(TileShape(8, 8, 8)) == (1, 1, 1)


def test_default_sram_tile_is_larger_than_array_equal_tile():
    assert default_sram_tile(MnkShape(32, 32, 64), 8, 8, 8) == TileShape(32, 32, 32)
    assert estimate_tile_sram_kb(TileShape(32, 32, 32)) == (2, 2, 2)


def test_boundary_configs_include_tile_bandwidth_sram_cases(tmp_path, monkeypatch):
    # Redirect generated configs so the unit test does not write into repo outputs.
    import tools.boundary_cases as boundary_cases

    monkeypatch.setattr(boundary_cases, "generated_cfg_path", lambda run_name: tmp_path / f"{run_name}.cfg")
    configs = build_boundary_configs(mnk=MnkShape(32, 32, 64), array_h=8, array_w=8)
    case_names = {cfg.case_name for cfg in configs}

    assert {"tile_smaller_than_array", "tile_equal_to_array", "tile_larger_than_array"} <= case_names
    assert {"bandwidth_large", "bandwidth_300", "bandwidth_200", "bandwidth_100", "bandwidth_80", "bandwidth_60", "bandwidth_40", "bandwidth_30", "bandwidth_20", "bandwidth_10"} <= case_names
    assert {"sram_large", "sram_fit", "sram_tight"} <= case_names
    assert {cfg.tile for cfg in configs if cfg.case_name.startswith("sram_")} == {"32x32x32"}
    assert all(cfg.layout == "test" for cfg in configs)
    assert all(cfg.mnk == MnkShape(32, 32, 64) for cfg in configs)


def test_nondefault_layouts_do_not_enable_custom_layout_flags_by_default(tmp_path, monkeypatch):
    import tools.boundary_cases as boundary_cases

    monkeypatch.setattr(boundary_cases, "generated_cfg_path", lambda run_name: tmp_path / f"{run_name}.cfg")
    configs = build_boundary_configs(
        mnk=MnkShape(32, 32, 64),
        array_h=8,
        array_w=8,
        layouts=(boundary_cases.DEFAULT_LAYOUT.parent.parent / "GEMM_mnk" / "vit_s_MK_KN.csv",),
    )
    assert {cfg.arch_spec.ifmap_custom_layout for cfg in configs} == {False}
    assert {cfg.arch_spec.filter_custom_layout for cfg in configs} == {False}


def test_custom_layout_flags_are_explicit_opt_in(tmp_path, monkeypatch):
    import tools.boundary_cases as boundary_cases

    monkeypatch.setattr(boundary_cases, "generated_cfg_path", lambda run_name: tmp_path / f"{run_name}.cfg")
    configs = build_boundary_configs(
        mnk=MnkShape(32, 32, 64),
        array_h=8,
        array_w=8,
        layouts=(boundary_cases.DEFAULT_LAYOUT.parent.parent / "GEMM_mnk" / "vit_s_MK_KN.csv",),
        use_custom_layouts=True,
    )
    assert {cfg.arch_spec.ifmap_custom_layout for cfg in configs} == {True}
    assert {cfg.arch_spec.filter_custom_layout for cfg in configs} == {True}
