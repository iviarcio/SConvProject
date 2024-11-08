module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op) {
    %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %res, %loops:4 = transform.structured.sconv %conv
      : (!transform.op<"linalg.conv_2d_nchw_fchw">)
      -> (!transform.op<"linalg.generic">, !transform.any_op, 
          !transform.any_op, !transform.any_op, !transform.any_op)
  
    transform.yield
  }
}
