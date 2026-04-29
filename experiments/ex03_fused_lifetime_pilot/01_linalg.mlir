// Fused MatMul + row-wise RMSNorm pilot input.
// M = N = K = 128, f32.
//
// Sub-tensor lifetime structure (intentional):
//   %mm  = linalg.matmul (A, B)             // MxN accumulator
//   %sq  = generic mm * mm                  // MxN element-wise square
//   %rs  = generic row-reduce(%sq)          // M-vector mean square
//   %out = generic mm / sqrt(rs/N + eps) * gamma  // MxN normalized output
//
// %mm is consumed by both %sq (stage 2) and %out (stage 4) — its lifetime
// spans across the row reduction (stage 3). This is the structural feature
// we want to make visible in the bufferized IR.

#map_mn = affine_map<(i, j) -> (i, j)>
#map_m  = affine_map<(i, j) -> (i)>
#map_n  = affine_map<(i, j) -> (j)>

func.func @fused_mm_rmsnorm(
    %A: tensor<128x128xf32>,
    %B: tensor<128x128xf32>,
    %gamma: tensor<128xf32>) -> tensor<128x128xf32> {
  %zero  = arith.constant 0.0 : f32
  %eps   = arith.constant 1.0e-5 : f32
  %inv_n = arith.constant 7.8125e-3 : f32   // 1.0 / 128.0

  // Stage 1: MM accumulator
  %e0 = tensor.empty() : tensor<128x128xf32>
  %f0 = linalg.fill ins(%zero : f32) outs(%e0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul
      ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%f0  : tensor<128x128xf32>) -> tensor<128x128xf32>

  // Stage 2: element-wise square (mm * mm)
  %e1 = tensor.empty() : tensor<128x128xf32>
  %sq = linalg.generic {
      indexing_maps = [#map_mn, #map_mn],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm : tensor<128x128xf32>) outs(%e1 : tensor<128x128xf32>) {
    ^bb0(%x: f32, %_y: f32):
      %p = arith.mulf %x, %x : f32
      linalg.yield %p : f32
  } -> tensor<128x128xf32>

  // Stage 3: row-sum reduction → M-vector
  %e2 = tensor.empty() : tensor<128xf32>
  %fz = linalg.fill ins(%zero : f32) outs(%e2 : tensor<128xf32>) -> tensor<128xf32>
  %rs = linalg.generic {
      indexing_maps = [#map_mn, #map_m],
      iterator_types = ["parallel", "reduction"]
    } ins(%sq : tensor<128x128xf32>) outs(%fz : tensor<128xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
  } -> tensor<128xf32>

  // Stage 4: normalized output = mm / sqrt(rs/N + eps) * gamma
  %e3 = tensor.empty() : tensor<128x128xf32>
  %out = linalg.generic {
      indexing_maps = [#map_mn, #map_m, #map_n, #map_mn],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm, %rs, %gamma : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)
      outs(%e3 : tensor<128x128xf32>) {
    ^bb0(%x: f32, %r: f32, %g: f32, %_y: f32):
      %mean  = arith.mulf %r, %inv_n : f32
      %v     = arith.addf %mean, %eps : f32
      %inv   = math.rsqrt %v : f32
      %scale = arith.mulf %inv, %g : f32
      %y     = arith.mulf %x, %scale : f32
      linalg.yield %y : f32
  } -> tensor<128x128xf32>

  return %out : tensor<128x128xf32>
}
