"""Export a HuggingFace model to ONNX, then import ONNX to IREE MLIR.

This is the artifact path for real downloaded models:

    HuggingFace weights/config -> PyTorch model -> ONNX -> iree-import-onnx MLIR

The resulting MLIR is the importer output, not the shape-only linalg skeleton
used by tools.hf_model_to_mlir.py.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)
DEFAULT_IMPORTER = REPO / ".venv" / "bin" / "iree-import-onnx"


def repo_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text).strip("._-") or "model"


def parse_input_shape(text: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in re.split(r"[x,]", text) if part.strip()]
    if len(parts) != 4:
        raise ValueError("--input-shape must be N,C,H,W, e.g. 1,3,224,224")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError("--input-shape dimensions must be positive")
    return shape  # type: ignore[return-value]


def default_output_dir(model_id: str) -> Path:
    return REPO / "models" / "hf_onnx_mlir" / safe_name(model_id)


def resolve_importer(path: Path | None = None) -> Path:
    if path:
        resolved = repo_path(path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"iree-import-onnx not found: {resolved}")

    env_path = os.environ.get("IREE_IMPORT_ONNX")
    candidates = [
        Path(env_path).expanduser() if env_path else None,
        DEFAULT_IMPORTER,
        Path(shutil.which("iree-import-onnx") or ""),
    ]
    for candidate in candidates:
        if candidate and str(candidate) and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "iree-import-onnx not found. Install IREE's ONNX importer or set IREE_IMPORT_ONNX."
    )


def build_import_command(
    importer: Path,
    onnx_path: Path,
    mlir_path: Path,
    *,
    no_verify: bool = False,
    data_prop: bool = True,
) -> list[str]:
    cmd = [str(importer), str(onnx_path), "-o", str(mlir_path)]
    if no_verify:
        cmd.append("--no-verify")
    if not data_prop:
        cmd.append("--no-data-prop")
    return cmd


def _weights_kwargs(weights_format: str) -> dict[str, bool]:
    if weights_format == "safetensors":
        return {"use_safetensors": True}
    if weights_format == "pickle":
        return {"use_safetensors": False}
    return {}


def load_model(
    model_id: str,
    *,
    task: str = "auto",
    revision: str | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    from_tf: bool = False,
    from_flax: bool = False,
    weights_format: str = "auto",
) -> Any:
    try:
        from transformers import AutoModel, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "HF export requires transformers/torch. Install: "
            "uv pip install --python .venv/bin/python -r requirements-model.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "from_tf": from_tf,
        "from_flax": from_flax,
        **_weights_kwargs(weights_format),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if task == "image-classification":
        return AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
    if task == "auto-model":
        return AutoModel.from_pretrained(model_id, **kwargs)

    try:
        return AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
    except Exception:
        return AutoModel.from_pretrained(model_id, **kwargs)


def first_tensor(output: Any) -> Any:
    import torch

    if torch.is_tensor(output):
        return output
    for attr in ("logits", "last_hidden_state", "pooler_output"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if torch.is_tensor(value):
                return value
    if isinstance(output, dict):
        for value in output.values():
            try:
                return first_tensor(value)
            except TypeError:
                continue
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return first_tensor(value)
            except TypeError:
                continue
    raise TypeError(f"model output does not contain a tensor: {type(output)!r}")


def export_hf_to_onnx(
    model: Any,
    onnx_path: Path,
    *,
    input_shape: tuple[int, int, int, int],
    opset: int,
    input_name: str,
    output_name: str,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires torch. Install requirements-model.txt."
        ) from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(input_shape, dtype=torch.float32)

    class ExportWrapper(torch.nn.Module):
        """Normalize HuggingFace ModelOutput to one tensor for ONNX export."""

        def __init__(self, wrapped_model: Any):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, pixel_values: Any) -> Any:
            try:
                output = self.wrapped_model(pixel_values=pixel_values)
            except TypeError:
                output = self.wrapped_model(pixel_values)
            return first_tensor(output)

    with torch.no_grad():
        torch.onnx.export(
            ExportWrapper(model),
            (dummy,),
            str(onnx_path),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def import_onnx_to_mlir(
    onnx_path: Path,
    mlir_path: Path,
    *,
    importer: Path | None = None,
    no_verify: bool = False,
    data_prop: bool = True,
) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "iree-import-onnx requires onnx. Install: "
            "uv pip install --python .venv/bin/python -r requirements-model.txt"
        ) from exc

    resolved = resolve_importer(importer)
    mlir_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_import_command(
            resolved,
            onnx_path,
            mlir_path,
            no_verify=no_verify,
            data_prop=data_prop,
        ),
        check=True,
    )


def convert_hf_to_onnx_mlir(
    model_id: str,
    *,
    output_dir: Path | None = None,
    onnx_output: Path | None = None,
    mlir_output: Path | None = None,
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE,
    task: str = "auto",
    revision: str | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    from_tf: bool = False,
    from_flax: bool = False,
    weights_format: str = "auto",
    opset: int = 17,
    input_name: str = "pixel_values",
    output_name: str = "output",
    importer: Path | None = None,
    skip_import: bool = False,
    no_verify: bool = False,
    data_prop: bool = True,
) -> tuple[Path, Path | None]:
    output_dir = repo_path(output_dir) if output_dir else default_output_dir(model_id)
    onnx_output = repo_path(onnx_output) if onnx_output else output_dir / "model.onnx"
    mlir_output = repo_path(mlir_output) if mlir_output else output_dir / "imported.mlir"
    cache_dir = repo_path(cache_dir) if cache_dir else None

    model = load_model(
        model_id,
        task=task,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        from_tf=from_tf,
        from_flax=from_flax,
        weights_format=weights_format,
    )
    export_hf_to_onnx(
        model,
        onnx_output,
        input_shape=input_shape,
        opset=opset,
        input_name=input_name,
        output_name=output_name,
    )
    if skip_import:
        return onnx_output, None

    import_onnx_to_mlir(
        onnx_output,
        mlir_output,
        importer=importer,
        no_verify=no_verify,
        data_prop=data_prop,
    )
    return onnx_output, mlir_output


def main() -> None:
    ap = argparse.ArgumentParser(description="HuggingFace model -> ONNX -> IREE imported MLIR")
    ap.add_argument("--model-id", required=True, help="e.g. microsoft/resnet-50")
    ap.add_argument("--task", choices=("auto", "image-classification", "auto-model"), default="auto")
    ap.add_argument("--revision")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--from-tf", action="store_true")
    ap.add_argument("--from-flax", action="store_true")
    ap.add_argument(
        "--weights-format",
        choices=("auto", "safetensors", "pickle"),
        default="auto",
        help="auto lets transformers choose; safetensors/pickle force use_safetensors",
    )
    ap.add_argument("--input-shape", default="1,3,224,224", help="dummy input N,C,H,W")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--input-name", default="pixel_values")
    ap.add_argument("--output-name", default="output")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--onnx-output", type=Path)
    ap.add_argument("--mlir-output", type=Path)
    ap.add_argument("--importer", type=Path, help="path to iree-import-onnx")
    ap.add_argument("--skip-import", action="store_true", help="only export ONNX")
    ap.add_argument("--no-verify", action="store_true", help="pass --no-verify to iree-import-onnx")
    ap.add_argument("--no-data-prop", action="store_true", help="pass --no-data-prop to iree-import-onnx")
    args = ap.parse_args()

    onnx_path, mlir_path = convert_hf_to_onnx_mlir(
        args.model_id,
        output_dir=args.output_dir,
        onnx_output=args.onnx_output,
        mlir_output=args.mlir_output,
        input_shape=parse_input_shape(args.input_shape),
        task=args.task,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        from_tf=args.from_tf,
        from_flax=args.from_flax,
        weights_format=args.weights_format,
        opset=args.opset,
        input_name=args.input_name,
        output_name=args.output_name,
        importer=args.importer,
        skip_import=args.skip_import,
        no_verify=args.no_verify,
        data_prop=not args.no_data_prop,
    )
    print(f"wrote {onnx_path}")
    if mlir_path:
        print(f"wrote {mlir_path}")
    else:
        print("skipped iree-import-onnx (--skip-import)")


if __name__ == "__main__":
    main()
