import pytest

pytestmark = pytest.mark.model

"""IREE-based ResNet/Conv tiling helpers."""

from pathlib import Path

from tools.iree_tile_resnet import (
    ConvTile,
    DEFAULT_TOPOLOGY,
    effective_mlir_ifmap_hw,
    emit_conv_topology,
    ofmap_hw,
    render_conv_module,
)
from tools.resnet_runner import parse_conv_topology


def _layer(name: str):
    return next(layer for layer in parse_conv_topology(DEFAULT_TOPOLOGY) if layer.layer_name == name)


def test_stride2_resnet_layer_uses_effective_padded_mlir_input():
    conv3 = _layer("Conv3_1a")
    assert ofmap_hw(conv3) == (28, 28)
    assert effective_mlir_ifmap_hw(conv3) == (57, 57)


def test_render_conv_module_tiles_output_h_w_and_oc():
    conv5 = _layer("Conv5_1a")
    text = render_conv_module([conv5], ConvTile(7, 7, 128))
    assert "linalg.conv_2d_nhwc_hwcf" in text
    assert "tensor<1x15x15x256xf32>" in text
    assert "tensor<3x3x256x512xf32>" in text
    assert "tile_sizes [0, 7, 7, 128, 0, 0, 0]" in text


def test_emit_conv_topology_expands_tiles():
    conv5 = _layer("Conv5_1a")
    text = emit_conv_topology([conv5], ConvTile(7, 7, 128))
    lines = text.splitlines()
    assert lines[0].startswith("Layer name, IFMAP Height")
    assert len(lines) == 5  # header + 1 OH tile * 1 OW tile * 4 OC tiles
    assert lines[1] == "Conv5_1a_oh000_ow000_oc0000, 15, 15, 3, 3, 256, 128, 2,"


def test_emit_conv_topology_handles_edge_tiles():
    conv3 = _layer("Conv3_1a")
    text = emit_conv_topology([conv3], ConvTile(8, 8, 96))
    lines = text.splitlines()
    assert len(lines) == 33  # header + ceil(28/8)^2 * ceil(128/96)
    assert lines[1] == "Conv3_1a_oh000_ow000_oc0000, 17, 17, 3, 3, 64, 96, 2,"
    assert lines[-1] == "Conv3_1a_oh024_ow024_oc0096, 9, 9, 3, 3, 64, 32, 2,"


def test_conv3_conv4_conv5_smoke_tile_count():
    layers = [_layer("Conv3_1a"), _layer("Conv4_1a"), _layer("Conv5_1a")]
    text = emit_conv_topology(layers, ConvTile(7, 7, 128))
    assert len(text.splitlines()) - 1 == 28
