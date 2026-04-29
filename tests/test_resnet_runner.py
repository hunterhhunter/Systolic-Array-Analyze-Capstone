"""ResNet baseline runner helpers."""

from pathlib import Path

from tools.resnet_runner import (
    DEFAULT_TOPOLOGY,
    equivalent_gemm,
    internal_folds,
    parse_conv_topology,
)


def test_resnet18_topology_has_21_rows_including_fc():
    layers = parse_conv_topology(DEFAULT_TOPOLOGY)
    assert len(layers) == 21
    assert layers[0].layer_name == "Conv1"
    assert layers[-1].layer_name == "FC"


def test_resnet_conv1_equivalent_gemm_shape():
    conv1 = parse_conv_topology(DEFAULT_TOPOLOGY)[0]
    assert equivalent_gemm(conv1) == (12100, 64, 147)


def test_resnet_conv1_ws_internal_folds_on_8x8_array():
    conv1 = parse_conv_topology(DEFAULT_TOPOLOGY)[0]
    s_r, s_c, t, row_fold, col_fold, folds = internal_folds(
        conv1, dataflow="ws", array_h=8, array_w=8
    )
    assert (s_r, s_c, t) == (147, 64, 12100)
    assert row_fold == 19
    assert col_fold == 8
    assert folds == 152


def test_tiny_3x3_conv_ws_internal_folds():
    tiny = parse_conv_topology(
        Path("SCALE-Sim/topologies/conv_nets/tiny_conv_test.csv")
    )[0]
    assert equivalent_gemm(tiny) == (4, 2, 12)
    assert internal_folds(tiny, dataflow="ws", array_h=3, array_w=3)[3:] == (
        4,
        1,
        4,
    )
