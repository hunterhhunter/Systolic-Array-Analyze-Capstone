#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @fused_mm_rmsnorm(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128xf32>) -> memref<128x128xf32> {
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.812500e-03 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<128x128xf32>)
    linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>) outs(%alloc : memref<128x128xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<128xf32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<128x128xf32>) outs(%alloc_2 : memref<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.mulf %in, %in : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_2, %arg2 : memref<128x128xf32>, memref<128xf32>, memref<128xf32>) outs(%alloc_3 : memref<128x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
      %0 = arith.mulf %in_4, %cst_1 : f32
      %1 = arith.addf %0, %cst_0 : f32
      %2 = math.rsqrt %1 : f32
      %3 = arith.mulf %2, %in_5 : f32
      %4 = arith.mulf %in, %3 : f32
      linalg.yield %4 : f32
    }
    bufferization.dealloc (%alloc : memref<128x128xf32>) if (%true)
    bufferization.dealloc (%alloc_2 : memref<128xf32>) if (%true)
    return %alloc_3 : memref<128x128xf32>
  }
}

