#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @fused_mm_rmsnorm(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.812500e-03 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_4 = arith.constant 128 : index
    %c128_5 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c32_6 = arith.constant 32 : index
    %c32_7 = arith.constant 32 : index
    %2 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %1) -> (tensor<128x128xf32>) {
      %10 = scf.for %arg5 = %c0_2 to %c128_4 step %c32_6 iter_args(%arg6 = %arg4) -> (tensor<128x128xf32>) {
        %11 = scf.for %arg7 = %c0_3 to %c128_5 step %c32_7 iter_args(%arg8 = %arg6) -> (tensor<128x128xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %extracted_slice_8 = tensor.extract_slice %arg1[%arg7, %arg5] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %extracted_slice_9 = tensor.extract_slice %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %12 = linalg.matmul ins(%extracted_slice, %extracted_slice_8 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_9 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %inserted_slice = tensor.insert_slice %12 into %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice : tensor<128x128xf32>
        }
        scf.yield %11 : tensor<128x128xf32>
      }
      scf.yield %10 : tensor<128x128xf32>
    }
    %3 = tensor.empty() : tensor<128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.mulf %in, %in : f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    %5 = tensor.empty() : tensor<128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<128x128xf32>) outs(%6 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %in, %out : f32
      linalg.yield %10 : f32
    } -> tensor<128xf32>
    %8 = tensor.empty() : tensor<128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %7, %arg2 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_8: f32, %in_9: f32, %out: f32):
      %10 = arith.mulf %in_8, %cst_1 : f32
      %11 = arith.addf %10, %cst_0 : f32
      %12 = math.rsqrt %11 : f32
      %13 = arith.mulf %12, %in_9 : f32
      %14 = arith.mulf %in, %13 : f32
      linalg.yield %14 : f32
    } -> tensor<128x128xf32>
    return %9 : tensor<128x128xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

