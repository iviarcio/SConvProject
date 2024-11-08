//===-- SConv.td - Transform dialect SConv -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the SConv.
//
//===----------------------------------------------------------------------===//

#ifndef SCONV
#define SCONV

include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

// Define the sconv operation. By convention, prefix its name with the name of the 
// extension, "structured.". The full op name will be further prefixed with "transform.".
def SConvOp : Op<Transform_Dialect, "structured.sconv",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        ReportTrackingListenerFailuresOpTrait]> {

  // Provide a brief and a full description of SConvOp. It is recommended describing 
  // the effects on the operands and how the operation processes various failure modes.
  let summary = "Improving Direct Convolution through Tensor Slicing, VB Packing.";
  let description = [{
    SConv is a direct-convolution algorithm based on an MLIR/LLVM code-generation
    toolchain that can be integrated into machine-learning compilers.
    This algorithm introduces: (a) Convolution Slicing Analysis (CSA) — a convolution-
    specific 3D cache-blocking analysis pass that focuses on tile reuse over the cache
    hierarchy; (b) Convolution Slicing Optimization (CSO) — a code-generation pass that
    uses CSA to generate a tiled direct-convolution macro-kernel; and (c) Vector-Based
    Packing (VBP) — an architecture-specific optimized input-tensor packing solution
    based on vector-register shift instructions for convolutions with unitary stride.
  }];

  // The argument include the handle to the payload operation.
  // The handle must implement TransformHandleTypeInterface.   
  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs TransformHandleTypeInterface:$transformed,
                      Variadic<TransformHandleTypeInterface>:$loops);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type(operands, results)
  }];

}

#endif // SCONV
