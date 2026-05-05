# npu-capstone Makefile — Phase 3 뼈대.
# Phase 3 후반(mlir2scalesim.py, pytest)에 test/demo 타겟이 실체 동작.

VENV ?= .venv
PY ?= $(VENV)/bin/python
VENV_PY ?= $(VENV)/bin/python
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
BOUNDARY_MNK ?= 32x32x64
BOUNDARY_ARRAY ?= 8x8
BOUNDARY_TILE_K ?= 8
BOUNDARY_LAYOUTS ?= SCALE-Sim/layouts/GEMM_mnk/vit_s_MK_KN.csv SCALE-Sim/layouts/GEMM_mnk/vit_s_KM_NK.csv
BOUNDARY_BANDWIDTHS ?= 600 300 200 100 80 60 40 30 20 10
BOUNDARY_SRAM_TILE ?=
BOUNDARY_RESULTS ?= results/boundary_sweep.parquet
BOUNDARY_CSV ?= results/boundary_sweep.csv
BOUNDARY_ONLY ?=
BOUNDARY_LIMIT ?=
BOUNDARY_DRY_RUN ?=
BOUNDARY_FAIL_FAST ?=
TPUV2_MNKS ?= 1024x1024x1024
TPUV2_ARRAYS ?= 64x64 128x128 256x256
TPUV2_TILE_RELATIONS ?= smaller equal larger
TPUV2_BANDWIDTHS ?= 1200 600 300 150 75
TPUV2_SRAM_SCALES ?= 0.25 0.5 1.0 2.0
TPUV2_LAYOUTS ?= SCALE-Sim/layouts/conv_nets/test.csv
TPUV2_MODE ?= one-factor
TPUV2_TILE_K ?=
TPUV2_LIMIT ?=
TPUV2_DRY_RUN ?=
TPUV2_SKIP_PLOTS ?=
TPUV2_FAIL_FAST ?=
TPUV2_JOBS ?= 1
TPUV2_BACKEND ?= thread
TPUV2_RESUME ?=
TPUV2_SKIP_IREE ?=
TPUV2_QUIET ?=
TPUV2_TOPOLOGY_MODE ?= grouped_full
TPUV2_CACHE_ROOT ?= outputs/cache/tpuv2_experiment
TPUV2_NO_CACHE ?=
TPUV2_HEARTBEAT_SEC ?= 30
TPUV2_RESULTS ?= results/tpuv2_experiment.parquet
TPUV2_CSV ?= results/tpuv2_experiment.csv
TPUV2_OUTPUT_ROOT ?= outputs/tpuv2_experiment
TPUV2_PLOT_DIR ?= results/figures/tpuv2_experiment
LLM_PRESETS ?= llama7b_prefill_2048
LLM_ARRAYS ?= 128x128 256x256
LLM_TILE_RELATIONS ?= equal larger
LLM_BANDWIDTHS ?= 1200 600 300 150
LLM_SRAM_SCALES ?= 1.0 0.25 0.0625 0.015625
LLM_LAYOUTS ?= SCALE-Sim/layouts/conv_nets/test.csv
LLM_MODE ?= one-factor
LLM_TILE_K ?=
LLM_LIMIT ?=
LLM_DRY_RUN ?=
LLM_SKIP_PLOTS ?=
LLM_FAIL_FAST ?=
LLM_JOBS ?= 1
LLM_BACKEND ?= thread
LLM_RESUME ?=
LLM_SKIP_IREE ?=
LLM_QUIET ?=
LLM_TOPOLOGY_MODE ?= grouped_full
LLM_CACHE_ROOT ?= outputs/cache/llm_tpuv2_experiment
LLM_NO_CACHE ?=
LLM_HEARTBEAT_SEC ?= 30
LLM_RESULTS ?= results/llm_tpuv2_experiment.parquet
LLM_CSV ?= results/llm_tpuv2_experiment.csv
LLM_OUTPUT_ROOT ?= outputs/llm_tpuv2_experiment
LLM_PLOT_DIR ?= results/figures/llm_tpuv2_experiment

DEEP_PRESETS ?= llama7b_prefill_decode bert_base vit_base resnet50
DEEP_ARRAYS ?= 128x128 256x256
DEEP_TILE_MN_FACTORS ?= 0.5 1.0 2.0 4.0
DEEP_TILE_KS ?= 64 128 256
DEEP_BANDWIDTHS ?= 1200 600 300 150
DEEP_SRAM_SCALES ?= 1.0 0.25 0.0625 0.015625
DEEP_CACHE_BW_SCALES ?= 1.0 0.5 0.25 0.125
DEEP_LAYOUTS ?= SCALE-Sim/layouts/conv_nets/test.csv
DEEP_DATAFLOWS ?= ws
DEEP_CUSTOM_WORKLOAD_JSON ?=
DEEP_MODE ?= one-factor
DEEP_LIMIT ?=
DEEP_DRY_RUN ?=
DEEP_SKIP_PLOTS ?=
DEEP_FAIL_FAST ?=
DEEP_JOBS ?= 1
DEEP_BACKEND ?= thread
DEEP_RESUME ?=
DEEP_SKIP_IREE ?=
DEEP_QUIET ?=
DEEP_TOPOLOGY_MODE ?= grouped_full
DEEP_CACHE_ROOT ?= outputs/cache/deep_model_tpuv2_experiment
DEEP_NO_CACHE ?=
DEEP_HEARTBEAT_SEC ?= 30
DEEP_RESULTS ?= results/deep_model_tpuv2_experiment.parquet
DEEP_CSV ?= results/deep_model_tpuv2_experiment.csv
DEEP_ENRICHED_CSV ?= results/deep_model_tpuv2_experiment_enriched.csv
DEEP_OUTPUT_ROOT ?= outputs/deep_model_tpuv2_experiment
DEEP_PLOT_DIR ?= results/figures/deep_model_tpuv2_experiment

.PHONY: env env-model freeze test test-unit test-e2e hf-onnx-mlir tile emit-topology run-exp experiment help
.PHONY: demo tile-mnk tile-input tile-resnet sweep boundary-sweep tpuv2-sweep tpuv2-plot llm-tpuv2-sweep llm-tpuv2-plot deep-model-tpuv2-sweep deep-model-tpuv2-plot plot-all aggregate reuse resnet resnet-smoke

help:
	@echo "Targets:"
	@echo "  env     - Create .venv (uv) and install runtime + dev deps + SCALE-Sim (-e)"
	@echo "  env-model - Install optional torch/transformers/onnx deps for HF conversion"
	@echo "  test    - Run fast/non-e2e pytest tests"
	@echo "  test-e2e - Run end-to-end pytest tests that need IREE/SCALE-Sim"
	@echo "  hf-onnx-mlir - Export HF_MODEL_ID to ONNX, then import ONNX to IREE MLIR"
	@echo "  tile    - IREE-tile MODEL_MLIR=<linalg.mlir> KIND=matmul|conv2d"
	@echo "  emit-topology - Emit SCALE-Sim topology.csv from RUN_DIR/tile_manifest.json"
	@echo "  run-exp - Run SCALE-Sim for RUN_DIR/topology.csv and summarize results"
	@echo "  experiment - Run tile -> emit-topology -> run-exp"
	@echo "  boundary-sweep - Run array/bandwidth/SRAM/layout boundary sweep"
	@echo "  tpuv2-sweep - Run TPUv2 array/tile/bandwidth/SRAM sweep and plots"
	@echo "  tpuv2-plot - Rebuild TPUv2 comparison plots from saved results"
	@echo "  llm-tpuv2-sweep - Run TPUv2 grouped-full sweep on LLM-like GEMM workloads"
	@echo "  llm-tpuv2-plot - Rebuild LLM TPUv2 comparison plots"
	@echo "  deep-model-tpuv2-sweep - Run broad SRAM/cache/tile sweeps across LLM/CNN/ViT/BERT workloads"
	@echo "  deep-model-tpuv2-plot - Rebuild broad deep-model TPUv2 plots"
	@echo "  plot-all - Rebuild TPUv2, LLM, and deep-model plots"

env:
	test -x $(PY) || $(UV) venv $(VENV) --python 3.12
	$(UV) pip install --python $(VENV_PY) -r requirements-dev.txt
	$(UV) pip install --python $(VENV_PY) -e ./SCALE-Sim

env-model:
	$(UV) pip install --python $(VENV_PY) -r requirements-model.txt

freeze:
	$(UV) pip freeze --python $(VENV_PY) > requirements.lock
	@echo "Wrote requirements.lock — diff against requirements.txt to spot drift."

test: test-unit

test-unit:
	$(PY) -m pytest -v tests/ -m "not e2e and not slow and not model"

test-e2e:
	$(PY) -m pytest -v tests/ -m "e2e or slow"

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

boundary-sweep:
	$(PY) -m tools.boundary_sweep \
		--mnk $(BOUNDARY_MNK) \
		--array $(BOUNDARY_ARRAY) \
		--tile-k $(BOUNDARY_TILE_K) \
		$(foreach bw,$(BOUNDARY_BANDWIDTHS),--bandwidth $(bw)) \
		$(foreach layout,$(BOUNDARY_LAYOUTS),--layout $(layout)) \
		$(if $(BOUNDARY_SRAM_TILE),--sram-tile $(BOUNDARY_SRAM_TILE),) \
		$(foreach group,$(BOUNDARY_ONLY),--only $(group)) \
		$(if $(BOUNDARY_LIMIT),--limit $(BOUNDARY_LIMIT),) \
		$(if $(BOUNDARY_DRY_RUN),--dry-run,) \
		$(if $(BOUNDARY_FAIL_FAST),--fail-fast,) \
		--output $(BOUNDARY_RESULTS) \
		--csv-output $(BOUNDARY_CSV)


tpuv2-sweep:
	$(PY) -m tools.tpuv2_experiment \
		$(foreach mnk,$(TPUV2_MNKS),--mnk $(mnk)) \
		$(foreach array,$(TPUV2_ARRAYS),--array $(array)) \
		$(foreach rel,$(TPUV2_TILE_RELATIONS),--tile-relation $(rel)) \
		$(foreach bw,$(TPUV2_BANDWIDTHS),--bandwidth $(bw)) \
		$(foreach scale,$(TPUV2_SRAM_SCALES),--sram-scale $(scale)) \
		$(foreach layout,$(TPUV2_LAYOUTS),--layout $(layout)) \
		--mode $(TPUV2_MODE) \
		$(if $(TPUV2_TILE_K),--tile-k $(TPUV2_TILE_K),) \
		$(if $(TPUV2_LIMIT),--limit $(TPUV2_LIMIT),) \
		$(if $(TPUV2_DRY_RUN),--dry-run,) \
		$(if $(TPUV2_SKIP_PLOTS),--skip-plots,) \
		$(if $(TPUV2_FAIL_FAST),--fail-fast,) \
		--jobs $(TPUV2_JOBS) \
		--parallel-backend $(TPUV2_BACKEND) \
		$(if $(TPUV2_RESUME),--resume,) \
		$(if $(TPUV2_SKIP_IREE),--skip-iree,) \
		--topology-mode $(TPUV2_TOPOLOGY_MODE) \
		$(if $(TPUV2_NO_CACHE),--no-cache,--cache-root $(TPUV2_CACHE_ROOT)) \
		$(if $(TPUV2_QUIET),--quiet,) \
		--heartbeat-sec $(TPUV2_HEARTBEAT_SEC) \
		--output $(TPUV2_RESULTS) \
		--csv-output $(TPUV2_CSV) \
		--output-root $(TPUV2_OUTPUT_ROOT) \
		--plot-dir $(TPUV2_PLOT_DIR)

tpuv2-plot:
	$(PY) -m tools.plot_tpuv2_results \
		--input $(TPUV2_CSV) \
		--plot-dir $(TPUV2_PLOT_DIR) \
		--enriched-output results/tpuv2_experiment_enriched.csv

llm-tpuv2-sweep:
	$(PY) -m tools.llm_tpuv2_experiment \
		$(foreach preset,$(LLM_PRESETS),--preset $(preset)) \
		$(foreach array,$(LLM_ARRAYS),--array $(array)) \
		$(foreach rel,$(LLM_TILE_RELATIONS),--tile-relation $(rel)) \
		$(foreach bw,$(LLM_BANDWIDTHS),--bandwidth $(bw)) \
		$(foreach scale,$(LLM_SRAM_SCALES),--sram-scale $(scale)) \
		$(foreach layout,$(LLM_LAYOUTS),--layout $(layout)) \
		--mode $(LLM_MODE) \
		$(if $(LLM_TILE_K),--tile-k $(LLM_TILE_K),) \
		$(if $(LLM_LIMIT),--limit $(LLM_LIMIT),) \
		$(if $(LLM_DRY_RUN),--dry-run,) \
		$(if $(LLM_SKIP_PLOTS),--skip-plots,) \
		$(if $(LLM_FAIL_FAST),--fail-fast,) \
		--jobs $(LLM_JOBS) \
		--parallel-backend $(LLM_BACKEND) \
		$(if $(LLM_RESUME),--resume,) \
		$(if $(LLM_SKIP_IREE),--skip-iree,) \
		--topology-mode $(LLM_TOPOLOGY_MODE) \
		$(if $(LLM_NO_CACHE),--no-cache,--cache-root $(LLM_CACHE_ROOT)) \
		$(if $(LLM_QUIET),--quiet,) \
		--heartbeat-sec $(LLM_HEARTBEAT_SEC) \
		--output $(LLM_RESULTS) \
		--csv-output $(LLM_CSV) \
		--output-root $(LLM_OUTPUT_ROOT) \
		--plot-dir $(LLM_PLOT_DIR)

llm-tpuv2-plot:
	$(PY) -m tools.plot_tpuv2_results \
		--input $(LLM_CSV) \
		--plot-dir $(LLM_PLOT_DIR) \
		--enriched-output results/llm_tpuv2_experiment_enriched.csv


deep-model-tpuv2-sweep:
	$(PY) -m tools.deep_model_tpuv2_experiment \
		$(foreach preset,$(DEEP_PRESETS),--preset $(preset)) \
		$(foreach array,$(DEEP_ARRAYS),--array $(array)) \
		$(foreach factor,$(DEEP_TILE_MN_FACTORS),--tile-mn-factor $(factor)) \
		$(foreach tk,$(DEEP_TILE_KS),--tile-k $(tk)) \
		$(foreach bw,$(DEEP_BANDWIDTHS),--bandwidth $(bw)) \
		$(foreach scale,$(DEEP_SRAM_SCALES),--sram-scale $(scale)) \
		$(foreach scale,$(DEEP_CACHE_BW_SCALES),--cache-bw-scale $(scale)) \
		$(foreach layout,$(DEEP_LAYOUTS),--layout $(layout)) \
		$(foreach dataflow,$(DEEP_DATAFLOWS),--dataflow $(dataflow)) \
		$(foreach json,$(DEEP_CUSTOM_WORKLOAD_JSON),--custom-workload-json $(json)) \
		--mode $(DEEP_MODE) \
		$(if $(DEEP_LIMIT),--limit $(DEEP_LIMIT),) \
		$(if $(DEEP_DRY_RUN),--dry-run,) \
		$(if $(DEEP_SKIP_PLOTS),--skip-plots,) \
		$(if $(DEEP_FAIL_FAST),--fail-fast,) \
		--jobs $(DEEP_JOBS) \
		--parallel-backend $(DEEP_BACKEND) \
		$(if $(DEEP_RESUME),--resume,) \
		$(if $(DEEP_SKIP_IREE),--skip-iree,) \
		--topology-mode $(DEEP_TOPOLOGY_MODE) \
		$(if $(DEEP_NO_CACHE),--no-cache,--cache-root $(DEEP_CACHE_ROOT)) \
		$(if $(DEEP_QUIET),--quiet,) \
		--heartbeat-sec $(DEEP_HEARTBEAT_SEC) \
		--output $(DEEP_RESULTS) \
		--csv-output $(DEEP_CSV) \
		--enriched-output $(DEEP_ENRICHED_CSV) \
		--output-root $(DEEP_OUTPUT_ROOT) \
		--plot-dir $(DEEP_PLOT_DIR)

deep-model-tpuv2-plot:
	$(PY) -m tools.plot_deep_model_results \
		--input $(DEEP_CSV) \
		--plot-dir $(DEEP_PLOT_DIR) \
		--enriched-output $(DEEP_ENRICHED_CSV)

plot-all: tpuv2-plot llm-tpuv2-plot deep-model-tpuv2-plot

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
