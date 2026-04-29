// Transform Dialect schedule: linalg.matmul을 [M=8, N=8, K=8] 로 타일링한다.
// 입력 matmul의 shape이 32x64 * 64x32 (M=32, N=32, K=64) 이므로,
// 타일링 결과 루프 반복 수: M/8=4, N/8=4, K/8=8 → 총 inner matmul instance 128개 (= 4*4*8).
// 단, 각 (i,j) 출력 타일에 대한 K 루프는 accumulation이므로
// SCALE-Sim 관점에서는 (M_out_tile × N_out_tile × K_partial)을 하나의 matmul로 볼지,
// K-split된 여러 개로 볼지 결정 필요 → 다음 단계에서 IR 보고 결정.

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // Step 1. matmul op 매칭
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Step 2. [M, N, K] = [8, 8, 8] 타일링
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [8, 8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
