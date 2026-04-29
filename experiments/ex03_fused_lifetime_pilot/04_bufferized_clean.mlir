#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @fused_mm_rmsnorm(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128xf32>) -> memref<128x128xf32> {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.812500e-03 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<128x128xf32>)
    scf.for %arg3 = %c0 to %c128 step %c32 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        scf.for %arg5 = %c0 to %c128 step %c32 {
          %subview = memref.subview %arg0[%arg3, %arg5] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          %subview_5 = memref.subview %arg1[%arg5, %arg4] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          %subview_6 = memref.subview %alloc[%arg3, %arg4] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          linalg.matmul ins(%subview, %subview_5 : memref<32x32xf32, strided<[128, 1], offset: ?>>, memref<32x32xf32, strided<[128, 1], offset: ?>>) outs(%subview_6 : memref<32x32xf32, strided<[128, 1], offset: ?>>)
          memref.copy %subview_6, %subview_6 : memref<32x32xf32, strided<[128, 1], offset: ?>> to memref<32x32xf32, strided<[128, 1], offset: ?>>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc : memref<128x128xf32>) outs(%alloc_2 : memref<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.mulf %in, %in : f32
      linalg.yield %0 : f32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<128xf32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc_2 : memref<128x128xf32>) outs(%alloc_3 : memref<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_3, %arg2 : memref<128x128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_4 : memref<128x128xf32>) {
    ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
      %0 = arith.mulf %in_5, %cst_1 : f32
      %1 = arith.addf %0, %cst_0 : f32
      %2 = math.rsqrt %1 : f32
      %3 = arith.mulf %2, %in_6 : f32
      %4 = arith.mulf %in, %3 : f32
      linalg.yield %4 : f32
    }
    bufferization.dealloc (%alloc_2 : memref<128x128xf32>) if (%true)
    bufferization.dealloc (%alloc_3 : memref<128xf32>) if (%true)
    bufferization.dealloc (%alloc : memref<128x128xf32>) if (%true)
    return %alloc_4 : memref<128x128xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

