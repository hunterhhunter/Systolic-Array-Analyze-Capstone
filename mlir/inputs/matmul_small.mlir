// Walking Skeleton input: 32x64 * 64x32 linalg.matmul
// Dimensions: M=32, K=64, N=32  (A: MxK, B: KxN, C: MxN)
// This is the smallest matmul where [8,8,8] tiling produces multiple tiles on every axis.

func.func @matmul_small(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32x32xf32>
  %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul
      ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>)
      outs(%filled : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
