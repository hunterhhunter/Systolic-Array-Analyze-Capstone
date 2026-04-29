// Walking Skeleton 변형: 동일 payload + [16,16,16] schedule (Phase 3 demo의 두 번째 데이터 포인트).
// Payload: 32x64 * 64x32 matmul (8x8x8 schedule과 공유)
// Schedule: [M=16, N=16, K=16] tile_using_for → 2 × 2 × 4 = 16 inner matmul
// 8x8 WS array에 over-fit (mapping_eff 측정 대상): per-tile 측정으로 타일 크기 효과 비교.

module attributes { transform.with_named_sequence } {

  func.func @matmul_small(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x32xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>)
        outs(%filled : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
