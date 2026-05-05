"""HuggingFace model-shape to linalg MLIR conversion helpers."""

from __future__ import annotations

import csv

import pytest

pytestmark = pytest.mark.model

from tools.hf_model_to_mlir import (
    LayerRecord,
    _conv_effective_input_hw,
    parse_input_shape,
    render_layer_mlir,
    write_summary_csv,
)


def test_parse_input_shape_accepts_commas_and_x_separator():
    assert parse_input_shape("1,3,224,224") == (1, 3, 224, 224)
    assert parse_input_shape("1x3x128x128") == (1, 3, 128, 128)


def test_parse_input_shape_rejects_bad_rank():
    with pytest.raises(ValueError, match="N,C,H,W"):
        parse_input_shape("1,3,224")


def test_conv_effective_input_hw_accounts_for_padding_shape():
    record = LayerRecord(
        index=0,
        name="resnet.encoder.stages.0.layers.0.layer.0.convolution",
        kind="conv2d",
        input_shape=(1, 64, 56, 56),
        output_shape=(1, 64, 56, 56),
        weight_shape=(64, 64, 3, 3),
        stride=(1, 1),
        dilation=(1, 1),
        padding=(1, 1),
    )
    assert _conv_effective_input_hw(record) == (58, 58)


def test_render_layer_mlir_emits_conv_and_linear_ops():
    records = [
        LayerRecord(
            index=0,
            name="embedder.embedder.convolution",
            kind="conv2d",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 64, 112, 112),
            weight_shape=(64, 3, 7, 7),
            stride=(2, 2),
            dilation=(1, 1),
            padding=(3, 3),
        ),
        LayerRecord(
            index=1,
            name="classifier.1",
            kind="linear",
            input_shape=(1, 512),
            output_shape=(1, 1000),
            weight_shape=(1000, 512),
        ),
    ]
    text = render_layer_mlir(records)
    assert "linalg.conv_2d_nhwc_hwcf" in text
    assert "tensor<1x229x229x3xf32>" in text
    assert "tensor<7x7x3x64xf32>" in text
    assert "linalg.matmul" in text
    assert "tensor<512x1000xf32>" in text


def test_render_layer_mlir_skips_grouped_conv():
    text = render_layer_mlir(
        [
            LayerRecord(
                index=0,
                name="depthwise",
                kind="conv2d",
                input_shape=(1, 32, 16, 16),
                output_shape=(1, 32, 16, 16),
                weight_shape=(32, 1, 3, 3),
                groups=32,
            )
        ]
    )
    assert "skipped grouped conv depthwise" in text
    assert "func.func" not in text


def test_write_summary_csv(tmp_path):
    out = tmp_path / "layers.csv"
    write_summary_csv(
        [
            LayerRecord(
                index=0,
                name="classifier",
                kind="linear",
                input_shape=(1, 512),
                output_shape=(1, 1000),
                weight_shape=(1000, 512),
            )
        ],
        out,
    )
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["name"] == "classifier"
    assert rows[0]["weight_shape"] == "1000x512"
