module attributes {transform.with_named_sequence} {
  func.func @fused_mm_rmsnorm(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128xf32>) -> memref<128x128xf32> {
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.812500e-03 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    scf.for %arg3 = %c0 to %c128 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        memref.store %cst, %alloc[%arg3, %arg4] : memref<128x128xf32>
      }
    }
    scf.for %arg3 = %c0 to %c128 step %c32 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        scf.for %arg5 = %c0 to %c128 step %c32 {
          %subview = memref.subview %arg0[%arg3, %arg5] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          %subview_5 = memref.subview %arg1[%arg5, %arg4] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          %subview_6 = memref.subview %alloc[%arg3, %arg4] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
          scf.for %arg6 = %c0 to %c32 step %c1 {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              scf.for %arg8 = %c0 to %c32 step %c1 {
                %0 = memref.load %subview[%arg6, %arg8] : memref<32x32xf32, strided<[128, 1], offset: ?>>
                %1 = memref.load %subview_5[%arg8, %arg7] : memref<32x32xf32, strided<[128, 1], offset: ?>>
                %2 = memref.load %subview_6[%arg6, %arg7] : memref<32x32xf32, strided<[128, 1], offset: ?>>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                memref.store %4, %subview_6[%arg6, %arg7] : memref<32x32xf32, strided<[128, 1], offset: ?>>
              }
            }
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    scf.for %arg3 = %c0 to %c128 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        %0 = memref.load %alloc[%arg3, %arg4] : memref<128x128xf32>
        %1 = arith.mulf %0, %0 : f32
        memref.store %1, %alloc_2[%arg3, %arg4] : memref<128x128xf32>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    scf.for %arg3 = %c0 to %c128 step %c1 {
      memref.store %cst, %alloc_3[%arg3] : memref<128xf32>
    }
    scf.for %arg3 = %c0 to %c128 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        %0 = memref.load %alloc_2[%arg3, %arg4] : memref<128x128xf32>
        %1 = memref.load %alloc_3[%arg3] : memref<128xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %alloc_3[%arg3] : memref<128xf32>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    scf.for %arg3 = %c0 to %c128 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        %0 = memref.load %alloc[%arg3, %arg4] : memref<128x128xf32>
        %1 = memref.load %alloc_3[%arg3] : memref<128xf32>
        %2 = memref.load %arg2[%arg4] : memref<128xf32>
        %3 = arith.mulf %1, %cst_1 : f32
        %4 = arith.addf %3, %cst_0 : f32
        %5 = math.rsqrt %4 : f32
        %6 = arith.mulf %5, %2 : f32
        %7 = arith.mulf %0, %6 : f32
        memref.store %7, %alloc_4[%arg3, %arg4] : memref<128x128xf32>
      }
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

