//===-- SConv.h - Transform dialect Extension SConv --------------*- c++ -*-===//
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

#ifndef SCONV_H
#define SCONV_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

namespace mlir {
class CallOpInterface;
class RewriterBase;

namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace transform {
class AnyOpType;
class AnyValueType;
class OperationType;
class TransformHandleTypeInterface;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;
} // namespace mlir

#define GET_OP_CLASSES
#include "SConv.h.inc"

// Registers our Transform dialect extension.
void registerSConv(::mlir::DialectRegistry &registry);

#endif // SCONV_H
