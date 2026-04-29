// Walking Skeleton: linalg.matmul + Transform Dialect schedule 통합 파일
// Payload: 32x64 * 64x32 matmul
// Schedule: [M=8, N=8, K=8] tile_using_for
// 사용: iree-opt input.mlir --pass-pipeline="builtin.module(transform-interpreter)"

module attributes { transform.with_named_sequence } {

  // --- Payload ---
  func.func @matmul_small(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x32xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>)
        outs(%filled : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }

  // --- Transform schedule ---
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [8, 8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
