# npu-capston Makefile — Phase 3 뼈대.
# Phase 3 후반(mlir2scalesim.py, pytest)에 test/demo 타겟이 실체 동작.

VENV ?= .venv
PY   := $(VENV)/bin/python
UV   := uv
IREE_OPT ?= $(VENV)/bin/iree-opt
export IREE_OPT
MODEL_MLIR ?= mlir/inputs/matmul_small.mlir
KIND ?= matmul
RUN_NAME ?= $(basename $(notdir $(MODEL_MLIR)))_$(KIND)
RUN_DIR ?= outputs/experiments/$(RUN_NAME)
TILE_ARGS ?= --tile-m 8 --tile-n 8 --tile-k 8
ARCH_CFG ?= SCALE-Sim/configs/tpuv2.cfg
RESULTS ?= results/$(RUN_NAME).parquet
MODEL_INPUT ?= $(MODEL_MLIR)
MODEL_TILE_ARGS ?= --preset matmul --tile-m 8 --tile-n 8 --tile-k 8
HF_MODEL_ID ?= microsoft/resnet-50
HF_CONVERT_ARGS ?= --weights-format safetensors --input-shape 1,3,224,224
MNK_M ?= 32
MNK_N ?= 32
MNK_K ?= 64
TILE_M ?= 8
TILE_N ?= 8
TILE_K ?= 8
SWEEP_MNKS ?= 32x32x64
SWEEP_TILES ?= 8x8x8 16x16x16
SWEEP_ARCH_CFGS ?= SCALE-Sim/configs/walking_8x8_ws.cfg
REPORT ?= outputs/walking_skeleton_full/walking_8x8_ws/COMPUTE_REPORT.csv

.PHONY: env env-model freeze test hf-onnx-mlir tile emit-topology run-exp experiment help
.PHONY: demo tile-mnk tile-input tile-resnet sweep aggregate reuse resnet resnet-smoke

help:
	@echo "Targets:"
	@echo "  env     - Create .venv (uv) and install runtime + dev deps + SCALE-Sim (-e)"
	@echo "  env-model - Install optional torch/transformers/onnx deps for HF conversion"
	@echo "  test    - Run pytest on tests/"
	@echo "  hf-onnx-mlir - Export HF_MODEL_ID to ONNX, then import ONNX to IREE MLIR"
	@echo "  tile    - IREE-tile MODEL_MLIR=<linalg.mlir> KIND=matmul|conv2d"
	@echo "  emit-topology - Emit SCALE-Sim topology.csv from RUN_DIR/tile_manifest.json"
	@echo "  run-exp - Run SCALE-Sim for RUN_DIR/topology.csv and summarize results"
	@echo "  experiment - Run tile -> emit-topology -> run-exp"

env:
	test -d $(PY) || $(UV) venv $(VENV) --python 3.12
	$(UV) pip install --python $(PY) -r requirements-dev.txt
	$(UV) pip install --python $(PY) -e ./SCALE-Sim

env-model:
	$(UV) pip install --python $(PY) -r requirements-model.txt

freeze:
	$(UV) pip freeze --python $(PY) > requirements.lock
	@echo "Wrote requirements.lock — diff against requirements.txt to spot drift."

test:
	$(PY) -m pytest -v tests/

hf-onnx-mlir:
	$(PY) -m tools.hf_to_onnx_mlir \
		--model-id $(HF_MODEL_ID) \
		$(HF_CONVERT_ARGS)

tile:
	$(PY) -m tools.tile_mlir \
		--input $(MODEL_MLIR) \
		--kind $(KIND) \
		--run-dir $(RUN_DIR) \
		$(TILE_ARGS)

emit-topology:
	$(PY) -m tools.emit_scalesim_topology \
		--manifest $(RUN_DIR)/tile_manifest.json \
		--output $(RUN_DIR)/topology.csv

run-exp:
	$(PY) -m tools.run_experiment \
		--run-dir $(RUN_DIR) \
		--kind $(KIND) \
		--arch-cfg $(ARCH_CFG) \
		--results $(RESULTS)

experiment: tile emit-topology run-exp

# Legacy/internal convenience targets kept for regression and older notes.
demo:
	$(PY) -m tools.demo

tile-mnk:
	$(PY) -m tools.iree_tile_mnk \
		--m $(MNK_M) --n $(MNK_N) --k $(MNK_K) \
		--tile-m $(TILE_M) --tile-n $(TILE_N) --tile-k $(TILE_K) \
		--emit-topology

tile-input:
	$(PY) -m tools.iree_tile_input --input $(MODEL_INPUT) $(MODEL_TILE_ARGS)

tile-resnet:
	$(PY) -m tools.iree_tile_resnet --smoke --run-scalesim

sweep:
	$(PY) -m tools.sweep_runner \
		$(foreach mnk,$(SWEEP_MNKS),--mnk $(mnk)) \
		--tiles $(SWEEP_TILES) \
		--arch-cfg $(SWEEP_ARCH_CFGS) \
		--csv-output results/results.csv

aggregate:
	$(PY) -m tools.aggregator $(REPORT) --output results/aggregate.csv

reuse:
	$(PY) -m tools.reuse_runner $(REPORT) \
		--output results/reuse_tiles.csv \
		--summary-output results/reuse_summary.csv

resnet:
	$(PY) -m tools.resnet_runner --verbose

resnet-smoke:
	$(PY) -m tools.resnet_runner --smoke \
		--results results/resnet18_tpuv2_smoke.parquet \
		--output-root outputs/resnet18_tpuv2_smoke \
		--verbose
