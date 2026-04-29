# NPU Capstone Experiment Pipeline

이 저장소는 모델이 MLIR/linalg 단계까지 내려온 뒤의 실험을 자동화한다.

```text
linalg MLIR -> IREE tiling -> SCALE-Sim topology -> SCALE-Sim run -> result summary
```

모델 다운로드, 학습, 정확도 검증, full framework export는 메인 실험 범위가 아니다. 핵심은 tile size와 하드웨어 설정이 systolic array utilization/cycle에 어떤 영향을 주는지 빠르게 비교하는 것이다.

## 공유 범위

GitHub에 공유할 때 포함할 것:

- `tools/`: tiling, topology 변환, SCALE-Sim 실행, metric 집계 스크립트
- `tests/`: smoke/unit test
- `mlir/inputs/`: 작은 예제 MLIR
- `SCALE-Sim/`: 이 프로젝트에서 쓰는 vendored 수정본
- `SCALE-Sim/configs/tpuv2.cfg`: 기본 HW config

공유하지 않을 것:

- `.venv/`
- `outputs/`
- `results/`
- `*.parquet`
- 모델 weight: `*.bin`, `*.safetensors`, `*.h5`, `*.ckpt`
- 큰 중간 산출물: `*.onnx`, HF cache

`SCALE-Sim/`은 upstream `scalesim-project/SCALE-Sim`의 MIT-licensed 코드를 이 프로젝트에 맞게 수정해 포함한 vendored copy다. 원본 라이선스는 `SCALE-Sim/LICENSE`에 유지한다.

`outputs/`와 `results/`는 실행 시 재생성되는 산출물이다. Parquet은 실험 중 pandas/pyarrow로 읽기 편해서 생성하지만 GitHub 공유 대상에서는 제외한다.

## 설치

Python 3.12와 `uv`를 사용한다.

```bash
make env
```

`make env`는 루트 `.venv`를 만들고, 이 저장소 안의 `SCALE-Sim/`을 editable 모드로 설치한다. SCALE-Sim은 upstream 패키지가 아니라 여기 포함된 수정본을 기준으로 실행해야 한다.

IREE transform 실행에는 `iree-opt`가 필요하다. `make env`에서 설치하는 `iree-compiler` pip package가 `.venv/bin/iree-opt`를 제공하므로 보통 별도 IREE source build는 필요 없다.

```bash
.venv/bin/iree-opt --version
```

팀원 PC에서 다른 IREE build를 쓰고 싶을 때만 직접 경로를 지정한다.

```bash
export IREE_OPT=/path/to/custom/iree-opt
```

설치 확인:

```bash
make test
```

## 전체 파이프라인

이 프로젝트의 기본 단위는 이미 MLIR/linalg까지 내려온 모델 조각이다. 전체 흐름은 다음과 같다.

```text
모델 또는 linalg MLIR
  -> IREE transform tiling
  -> tile_manifest.json
  -> SCALE-Sim topology.csv
  -> SCALE-Sim 실행
  -> COMPUTE_REPORT.csv
  -> results/*.parquet, results/*.csv 요약
```

`outputs/`에는 중간 산출물과 SCALE-Sim 원본 report가 생기고, `results/`에는 비교하기 쉬운 요약 테이블이 생긴다. 두 디렉토리는 재생성 가능하므로 GitHub에는 올리지 않는다.

## 빠른 실행

설치가 끝난 뒤 파이프라인이 정상 동작하는지 가장 빨리 확인하는 명령이다.

```bash
make demo
```

기본 MatMul sweep을 실행한다.

입력과 설정:

- workload: `32x32x64` MatMul
- tile: `8x8x8`, `16x16x16`
- hardware config: `SCALE-Sim/configs/walking_8x8_ws.cfg`

생성 파일:

```text
outputs/demo/
results/demo.parquet
results/demo.csv
```

여러 tile 크기를 비교하는 sweep은 다음처럼 실행한다.

```bash
make sweep \
  SWEEP_MNKS="32x32x64" \
  SWEEP_TILES="8x8x8 16x16x16" \
  SWEEP_ARCH_CFGS=SCALE-Sim/configs/walking_8x8_ws.cfg
```

생성 파일:

```text
outputs/small_sweep/
results/results.parquet
results/results.csv
```

## linalg MLIR -> SCALE-Sim 실행

### 1. IREE Tiling

입력 MLIR에 transform schedule을 붙이고 `iree-opt`로 tiled MLIR을 만든다.

```bash
make tile \
  MODEL_MLIR=mlir/inputs/matmul_small.mlir \
  KIND=matmul \
  RUN_DIR=outputs/experiments/matmul_8x8x8 \
  TILE_ARGS="--tile-m 8 --tile-n 8 --tile-k 8"
```

생성 파일:

```text
outputs/experiments/matmul_8x8x8/input_with_schedule.mlir
outputs/experiments/matmul_8x8x8/tiled.mlir
outputs/experiments/matmul_8x8x8/tile_manifest.json
```

### 2. SCALE-Sim Topology 변환

tile manifest를 SCALE-Sim이 읽을 수 있는 topology CSV로 변환한다.

```bash
make emit-topology RUN_DIR=outputs/experiments/matmul_8x8x8
```

생성 파일:

```text
outputs/experiments/matmul_8x8x8/topology.csv
```

edge tile은 남은 실제 크기만큼 작은 row로 emit한다. 그래서 tile size가 array와 딱 맞지 않을 때 mapping efficiency 변화가 결과에 드러난다.

### 3. SCALE-Sim 실행과 결과 집계

기본 HW는 TPUv2 config다.

```bash
make run-exp \
  RUN_DIR=outputs/experiments/matmul_8x8x8 \
  KIND=matmul \
  ARCH_CFG=SCALE-Sim/configs/tpuv2.cfg \
  RESULTS=results/matmul_8x8x8.parquet
```

생성 파일:

- `outputs/experiments/matmul_8x8x8/sim/<scale-run-name>/COMPUTE_REPORT.csv`
- `results/matmul_8x8x8.parquet`
- `results/matmul_8x8x8.csv`

`results/*.parquet`은 pandas/pyarrow로 분석하기 좋은 형식이고, `results/*.csv`는 바로 열어보기 좋은 공유용 요약이다. 둘 다 재생성 가능하므로 commit하지 않는다.

### 한 번에 실행

위 세 단계를 한 번에 실행하려면 `make experiment`를 사용한다.

```bash
make experiment \
  MODEL_MLIR=mlir/inputs/matmul_small.mlir \
  KIND=matmul \
  RUN_DIR=outputs/experiments/matmul_8x8x8 \
  TILE_ARGS="--tile-m 8 --tile-n 8 --tile-k 8" \
  ARCH_CFG=SCALE-Sim/configs/tpuv2.cfg \
  RESULTS=results/matmul_8x8x8.parquet
```

Conv2D linalg MLIR을 넣을 때는 `KIND=conv2d`와 Conv tile 옵션을 사용한다.

```bash
make experiment \
  MODEL_MLIR=path/to/conv_linalg.mlir \
  KIND=conv2d \
  RUN_DIR=outputs/experiments/conv_8x8x96 \
  TILE_ARGS="--tile-oh 8 --tile-ow 8 --tile-oc 96" \
  ARCH_CFG=SCALE-Sim/configs/tpuv2.cfg \
  RESULTS=results/conv_8x8x96.parquet
```

## SCALE-Sim topology 직접 실행

이미 SCALE-Sim topology CSV가 있으면 IREE tiling 단계를 건너뛰고 바로 SCALE-Sim을 실행할 수 있다.

```bash
.venv/bin/python -m tools.run_experiment \
  --topology SCALE-Sim/topologies/conv_nets/tiny_conv_test.csv \
  --kind conv2d \
  --arch-cfg SCALE-Sim/configs/tiny_3x3.cfg \
  --output-root outputs/tiny_conv \
  --results results/tiny_conv.parquet \
  --name tiny_conv
```

ResNet-18처럼 SCALE-Sim에 이미 들어있는 topology를 기준으로 baseline을 돌릴 때는 convenience target을 쓸 수 있다.

```bash
make resnet-smoke
```

생성 파일:

```text
outputs/resnet18_tpuv2_smoke/
results/resnet18_tpuv2_smoke.parquet
results/resnet18_tpuv2_smoke.csv
```

## 지원 입력

공개 v1 경로에서 안정적으로 지원하는 op:

- `linalg.matmul`
- `linalg.conv_2d_nhwc_hwcf`

전제:

- static shape
- f32 tensor
- Conv2D는 NHWC input, HWCF filter
- SCALE-Sim topology 기준 Conv stride는 h/w가 같아야 함
- dilation/group conv는 현재 일반 실험 경로에서 제외

HuggingFace/ONNX helper로 만든 `imported.mlir`처럼 `torch.operator "onnx.Conv"`가 들어있는 파일도 Conv shape를 읽어 `linalg_bridge.mlir`을 생성할 수 있다. 이 bridge는 SCALE-Sim topology 실험용 shape-only MLIR이다. 즉, ResNet의 residual add, activation, pooling까지 수치적으로 동일하게 재현하는 full lowering은 아니고, Conv layer의 tile/cycle 실험을 위한 입력이다.

## Optional: 모델에서 MLIR 생성

메인 공유 흐름은 linalg MLIR을 직접 받는 것을 권장한다. 그래도 HuggingFace 모델에서 ONNX와 IREE imported MLIR을 만들고 싶으면 optional dependency를 설치한다.

```bash
make env-model

make hf-onnx-mlir \
  HF_MODEL_ID=microsoft/resnet-50 \
  HF_CONVERT_ARGS="--weights-format safetensors --input-shape 1,3,224,224"
```

생성 위치:

```text
models/hf_onnx_mlir/microsoft_resnet-50/
  model.onnx
  imported.mlir
```

`model.onnx`는 공유 대상에서 제외한다. `imported.mlir`도 크기가 크면 공유하지 말고, 필요한 경우 linalg까지 lowering한 작은 artifact만 별도로 올린다.

이 단계는 모델을 실험 입력으로 준비하는 optional helper다. 핵심 실험 파이프라인은 `linalg MLIR -> make experiment -> results/*.csv`이다.

## 주요 Metric

결과 해석에서 우선 보는 값:

- `n_tiles`
- `compute_cycles`
- `total_cycles`
- `reuse_aware_cycles`
- `mean_mapping_eff_pct`

`total_cycles`만 보고 결론을 내리면 안 된다. Tile-as-layer 방식에서는 각 tile마다 prefetch overhead가 반복 계산될 수 있어서 total cycle이 실제보다 커 보일 수 있다. 그래서 compute-only 관점의 `compute_cycles`, prefetch bias를 완화한 `reuse_aware_cycles`, array utilization을 보는 `mean_mapping_eff_pct`를 같이 비교한다.

## 자주 쓰는 명령 요약

환경 구성:

```bash
make env
make test
```

MatMul 단일 실험:

```bash
make experiment
```

MatMul sweep:

```bash
make sweep
```

SCALE-Sim report만 다시 요약:

```bash
make aggregate REPORT=outputs/tiny_conv/tiny_conv_3x3/COMPUTE_REPORT.csv
```

Reuse-aware 보정 CSV 생성:

```bash
make reuse REPORT=outputs/tiny_conv/tiny_conv_3x3/COMPUTE_REPORT.csv
```

Full ResNet 계열 run은 topology row가 많아서 오래 걸릴 수 있다. 먼저 `make demo`, `make resnet-smoke`, 작은 layer topology로 smoke를 확인한 뒤 전체 sweep을 돌리는 것을 권장한다.

## 스크립트 구조

- `tools/tile_mlir.py`: 공개 tiling entrypoint
- `tools/emit_scalesim_topology.py`: tile manifest -> SCALE-Sim topology CSV
- `tools/run_experiment.py`: SCALE-Sim 실행 + summary CSV/parquet 생성
- `tools/aggregator.py`: SCALE-Sim compute report 집계
- `tools/reuse_model.py`: tile별 reuse-aware cycle 계산
- `tools/hf_to_onnx_mlir.py`: optional HF -> ONNX -> IREE imported MLIR helper

Legacy/internal script는 남겨두되, 팀원이 새 실험을 시작할 때는 위 세 단계 `tile`, `emit-topology`, `run-exp`를 기준으로 보면 된다.

## GitHub 공유 체크리스트

- `.venv/`, `outputs/`, `*.parquet`, 큰 model artifact가 올라가지 않는지 확인한다.
- `make test`가 통과하는 상태에서 공유한다.