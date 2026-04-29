"""HF -> ONNX -> IREE MLIR conversion helper tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.hf_to_onnx_mlir import (
    _weights_kwargs,
    build_import_command,
    default_output_dir,
    parse_input_shape,
    resolve_importer,
    safe_name,
)


def test_parse_input_shape_accepts_common_separators():
    assert parse_input_shape("1,3,224,224") == (1, 3, 224, 224)
    assert parse_input_shape("1x3x128x128") == (1, 3, 128, 128)


def test_parse_input_shape_rejects_bad_rank():
    with pytest.raises(ValueError, match="N,C,H,W"):
        parse_input_shape("1,3,224")


def test_weights_format_maps_to_transformers_kwarg():
    assert _weights_kwargs("auto") == {}
    assert _weights_kwargs("safetensors") == {"use_safetensors": True}
    assert _weights_kwargs("pickle") == {"use_safetensors": False}


def test_safe_name_and_default_output_dir_are_stable():
    assert safe_name("microsoft/resnet-50") == "microsoft_resnet-50"
    assert default_output_dir("microsoft/resnet-50").as_posix().endswith(
        "models/hf_onnx_mlir/microsoft_resnet-50"
    )


def test_resolve_importer_accepts_explicit_path(tmp_path):
    importer = tmp_path / "iree-import-onnx"
    importer.write_text("#!/bin/sh\n")
    assert resolve_importer(importer) == importer


def test_build_import_command_includes_importer_flags():
    cmd = build_import_command(
        Path("/tmp/iree-import-onnx"),
        Path("model.onnx"),
        Path("imported.mlir"),
        no_verify=True,
        data_prop=False,
    )
    assert cmd == [
        "/tmp/iree-import-onnx",
        "model.onnx",
        "-o",
        "imported.mlir",
        "--no-verify",
        "--no-data-prop",
    ]
