#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @fused_mm_rmsnorm(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.812500e-03 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %3 = tensor.empty() : tensor<128xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %in : f32
      %9 = arith.addf %8, %out : f32
      linalg.yield %9 : f32
    } -> tensor<128xf32>
    %6 = tensor.empty() : tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %5, %arg2 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%6 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in_2, %cst_1 : f32
      %9 = arith.addf %8, %cst_0 : f32
      %10 = math.rsqrt %9 : f32
      %11 = arith.mulf %10, %in_3 : f32
      %12 = arith.mulf %in, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<128x128xf32>
    return %7 : tensor<128x128xf32>
  }
}

