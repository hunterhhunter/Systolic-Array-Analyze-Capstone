# Walking Skeleton Baseline: linalg.matmul(32x64 * 64x32) + [8,8,8] 타일링

생성일: 2026-04-21
목적: IREE Transform Dialect → tiled MLIR → 수작업 변환 → SCALE-Sim 까지 **1회 전체 관통** 을 기록.
다음 단계: `tools/tile_extractor.py` + `tools/mlir_to_scalesim.py` 자동화.

---

## 1. 입력

- **Payload** ([mlir/inputs/matmul_small.mlir](../../mlir/inputs/matmul_small.mlir)): `linalg.matmul` with `A: 32x64`, `B: 64x32`, `C: 32x32`, f32.
- **Schedule** ([mlir/schedules/tile_matmul_8x8x8.mlir](../../mlir/schedules/tile_matmul_8x8x8.mlir)): `transform.structured.tile_using_for tile_sizes [8, 8, 8]`.
- **Combined file** ([mlir/inputs/matmul_small_with_schedule.mlir](../../mlir/inputs/matmul_small_with_schedule.mlir)): payload + schedule in one module (transform-interpreter가 외부 파일 옵션 대신 nested named_sequence를 찾도록).

## 2. 변환 커맨드

```bash
IREE_OPT=/home/swlab-youngjin/iree-build/tools/iree-opt
$IREE_OPT mlir/inputs/matmul_small_with_schedule.mlir \
  --pass-pipeline="builtin.module(transform-interpreter)" \
  -o mlir/tiled_outputs/matmul_tiled.mlir
```

외부 schedule 파일을 별도로 주는 `--transform-file-name=` 옵션 형식은 `--pass-pipeline` 문법 안에서 인식되지 않았다. **Phase 3 자동화 단계에서 해결할 과제**. 현재는 한 파일 병합으로 우회.

## 3. Tiled IR 분석

[matmul_tiled.mlir](../../mlir/tiled_outputs/matmul_tiled.mlir) 구조:

```
scf.for M (0..32 step 8)   // 4 iterations
  scf.for N (0..32 step 8)   // 4 iterations
    scf.for K (0..64 step 8)   // 8 iterations
      tensor.extract_slice A[M, K] [8, 8] [1, 1]
      tensor.extract_slice B[K, N] [8, 8] [1, 1]
      tensor.extract_slice C[M, N] [8, 8] [1, 1]
      linalg.matmul ins(<8x8>, <8x8>) outs(<8x8>)
      tensor.insert_slice ... into C[M, N]
```

타일 인스턴스:
- 루프 반복 총합 = 4 × 4 × 8 = **128 inner matmul ops**
- 각 inner matmul의 GEMM 차원 = **(M=8, N=8, K=8)**
- K 루프는 동일 C[M,N] 타일에 accumulation. SCALE-Sim은 layer 간 reuse를 모델링하지 않으므로 각 8×8×8 instance를 독립 layer로 처리.

## 4. SCALE-Sim 입력 (수작업 변환)

**Config**: [SCALE-Sim/configs/walking_8x8_ws.cfg](../../SCALE-Sim/configs/walking_8x8_ws.cfg)
- ArrayHeight = ArrayWidth = 8
- SRAM 64KB 각각
- Dataflow: Weight Stationary

**Topology (GEMM 모드)**:
- [matmul_tiled_8x8x8.csv](../../SCALE-Sim/topologies/walking_skeleton/matmul_tiled_8x8x8.csv) — 단일 타일 smoke test
- [matmul_tiled_8x8x8_full.csv](../../SCALE-Sim/topologies/walking_skeleton/matmul_tiled_8x8x8_full.csv) — 128개 동일 타일

CSV 포맷 (input_type=gemm):
```
Layer Name, M, N, K,
Tile_000, 8, 8, 8,
...
```

## 5. 실행 결과

커맨드:
```bash
cd SCALE-Sim
python scalesim/scale.py \
  -c configs/walking_8x8_ws.cfg \
  -t topologies/walking_skeleton/matmul_tiled_8x8x8_full.csv \
  -p ../outputs/walking_skeleton_full/ \
  -i gemm
```

**Per-tile** ([COMPUTE_REPORT.csv](../../outputs/walking_skeleton_full/walking_8x8_ws/COMPUTE_REPORT.csv) 각 행):

| Metric | Value |
|---|---|
| Total Cycles (incl. prefetch) | 102 |
| Total Cycles (compute) | 29 |
| Stall Cycles | 0 |
| Overall Utilization | 27.59% |
| Mapping Efficiency | 100.00% |
| Compute Utilization | 21.62% |

**128 타일 합산** (`aggregator` 손 스크립트):

| Metric | Value |
|---|---|
| Sum Total Cycles (incl. prefetch) | **13,056** |
| Sum Total Cycles (compute) | **3,712** |
| Sum Stall Cycles | 0 |

## 6. 검증과 해석

**이상적 lower bound** (어떤 mapping loss도 없는 경우):
- 전체 matmul: M=32, N=32, K=64 → MACs = 32 × 32 × 64 = **65,536**
- 8×8 array peak throughput = 64 MACs/cycle
- Ideal cycles = 65,536 / 64 = **1,024**

**관찰**:
- 측정 = 3,712 cycles → **이상적 대비 3.6배**.
- 각 타일마다 `Total Cycles(incl prefetch) - Compute` = 102 − 29 = **73 cycles 프리페치 오버헤드**. 타일이 많을수록 이 오버헤드가 누적.
- 타일 크기가 array와 정확히 일치(8×8×8 vs 8×8 array)해서 Mapping Efficiency는 **100%**. 그러나 Overall Util은 27%밖에 안 됨 — **prefetch/fill 파이프라인 때문**.
- 이게 바로 우리가 측정하고 싶은 "**타일링 결정의 영향**": 같은 matmul이라도 다른 타일 크기(예: [16,16,16])로 했을 때 per-tile 오버헤드가 줄지, 전체 사이클이 줄지 Phase 4 sweep에서 확인할 포인트.

## 7. 열린 이슈 / Phase 3 TODO

1. **`--transform-file-name` 옵션 문법**: pass-pipeline 내 서브옵션 vs 상위 CLI 플래그 구분 필요. IREE `transform-interpreter` pass가 이 옵션을 외부 CLI 플래그 방식으로 노출하는지 소스 확인.
2. **Parser 설계 결정**: tiled IR의 각 inner matmul을 자동 추출할 때, 어떤 키를 사용할 것인가?
   - (a) `tensor.extract_slice` 의 `[8, 8, 8]` 크기
   - (b) `scf.for` 의 `step` 값
   - (c) inner `linalg.matmul` 의 `tensor<8x8xf32>` shape
   세 가지가 모두 일관되므로 가장 견고한 (c)를 제1소스, (b)를 cross-check로 제안.
3. **K-fold 처리**: 현재 128개 독립 layer로 SCALE-Sim에 던졌지만, 실제 하드웨어에서는 같은 (i,j) 출력 타일에 대한 K 루프가 accumulator를 유지할 것. 이 점은 Limitations에 명시 필요.
4. **다른 타일 크기**: [16,16,16], [4,4,4] 와 비교해 봐야 "타일 크기의 영향" 이라는 논지가 선다. Phase 3 자동화와 동시에 [4], [8], [16], [32] 최소 4점 sweep 설계.

## 8. 산출 아티팩트 체크리스트

- [x] `mlir/inputs/matmul_small.mlir`
- [x] `mlir/schedules/tile_matmul_8x8x8.mlir`
- [x] `mlir/inputs/matmul_small_with_schedule.mlir`
- [x] `mlir/tiled_outputs/matmul_tiled.mlir`
- [x] `SCALE-Sim/configs/walking_8x8_ws.cfg`
- [x] `SCALE-Sim/topologies/walking_skeleton/matmul_tiled_8x8x8.csv`
- [x] `SCALE-Sim/topologies/walking_skeleton/matmul_tiled_8x8x8_full.csv`
- [x] `outputs/walking_skeleton_full/walking_8x8_ws/COMPUTE_REPORT.csv`
- [x] 이 문서
