//===-- SConv.cpp - Transform dialect Extension SConv ----------------------===//
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

#include "SConv.h"
#include "CSA.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

#define DEBUG_TYPE "sconv-transform"
#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "]: ")

#define GET_OP_CLASSES
#include "SConv.cpp.inc"

// Define the SConv transform dialect. This uses the CRTP idiom to identify extensions.
class SConv
    : public transform::TransformDialectExtension<SConv> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SConv)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void SConv::init() {
  // As an transform extension dialect, we must declare all dependent dialects.
  // These dialects will be loaded along with the extension and, therefore,
  // along with the Transform dialect. The dependent dialects contain the
  // attributes or types used by transform operations.
  declareDependentDialect<linalg::LinalgDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  declareGeneratedDialect<affine::AffineDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<index::IndexDialect>();
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<tensor::TensorDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "SConv.cpp.inc"
      >();
}

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](const APInt &element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  if (isa<IntegerType>(x.getType()))
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, accType, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, accType, /*isUnsignedCast=*/false);
  if (isa<IntegerType>(accType))
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
}

// Swap the inner loops when schedule is Input Stationary
static LogicalResult
swapInductionVars(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                  SmallVector<Operation *> tiledOps, SmallVector<Operation *> &loopOps) {

  if (res.schd != IS) return success();

  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[0]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[1]);
  if (!outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  // Create the new outerLoop with innerLoop args, except InitArgs
  rewriter.setInsertionPoint(outerLoop);
  auto newOuterLoop = rewriter.create<scf::ForOp>(
      outerLoop.getLoc(), innerLoop.getLowerBound(), innerLoop.getUpperBound(),
      innerLoop.getStep(), outerLoop.getInitArgs());

  // Create the new innerLoop with outerLoop args, except InitArgs
  rewriter.setInsertionPointToStart(newOuterLoop.getBody());
  auto newInnerLoop = rewriter.create<scf::ForOp>(
      innerLoop.getLoc(), outerLoop.getLowerBound(), outerLoop.getUpperBound(),
      outerLoop.getStep(), newOuterLoop.getRegionIterArgs());

  // Create mappings for swapping induction variables
  IRMapping mapping;
  mapping.map(outerLoop.getInductionVar(), newInnerLoop.getInductionVar());
  mapping.map(innerLoop.getInductionVar(), newOuterLoop.getInductionVar());

  // Map the iterative arguments of the innerLoop and outerLoop
    for (auto [oldArg, newArg] : llvm::zip(outerLoop.getRegionIterArgs(), newOuterLoop.getRegionIterArgs())) {
    mapping.map(oldArg, newArg);
  }
  for (auto [oldArg, newArg] : llvm::zip(innerLoop.getRegionIterArgs(), newInnerLoop.getRegionIterArgs())) {
    mapping.map(oldArg, newArg);
  }

  // Clone the inner loop's body into the new innerLoop
  rewriter.setInsertionPointToStart(newInnerLoop.getBody());
  for (Operation &op : *innerLoop.getBody()) {
    if (!isa<scf::YieldOp>(&op)) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(), clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }
  }

  // Lookup the tensor.insert_slice op in the innerLoop Body
  tensor::InsertSliceOp insertSliceOp = nullptr;
  for (Operation &op : newInnerLoop.getBody()->getOperations()) {
    if (auto candidate = dyn_cast<tensor::InsertSliceOp>(&op)) {
      if (insertSliceOp)
        return transformOp->emitError("Multiple tensor.insert_slice operations found in the innerLoop body");
      insertSliceOp = candidate;
    }
  }
  if (!insertSliceOp)
    return transformOp->emitError("No tensor.insert_slice operation found in the innerLoop body");

  Value yieldedFilter = insertSliceOp.getResult();
  Value yieldedInput = mapping.lookup(innerLoop.getRegionIterArgs().front());

  Type yieldedFilterType = yieldedFilter.getType();
  Type yieldedInputType = yieldedInput.getType();

  // Valida os tipos do filter e input antes de criar o scf.yield
  if (!yieldedFilterType || !yieldedFilterType.isa<ShapedType>()) {
    return transformOp->emitError("Invalid or missing FilterType in scf.yield");
  }
  if (!yieldedInputType || !yieldedInputType.isa<ShapedType>()) {
    return transformOp->emitError("Invalid or missing InputType in scf.yield");
  }

  // Insere o yield com os valores do input e filter
  rewriter.create<scf::YieldOp>(innerLoop.getLoc(), ValueRange{yieldedInput, yieldedFilter});

  // Clone the outer loop's body into the new outerLoop
  rewriter.setInsertionPointToStart(newOuterLoop.getBody());
  for (Operation &op : *outerLoop.getBody()) {
    if (&op != innerLoop.getOperation() && !isa<scf::YieldOp>(&op)) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(), clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }
  }

  // Add a yield operation for the new outerLoop
  rewriter.setInsertionPointToEnd(newOuterLoop.getBody());
  Value yieldValueOuter = newInnerLoop.getResults().front();
  rewriter.create<scf::YieldOp>(outerLoop.getLoc(), yieldValueOuter);

  llvm::errs() << "\nDump after clonning outerLoop:\n";
  newOuterLoop.dump();

  // Replace all uses of the original loops' results
  for (auto [oldRes, newRes] : llvm::zip(innerLoop.getResults(), newInnerLoop.getResults())) {
    oldRes.replaceAllUsesWith(newRes);
  }
  for (auto [oldRes, newRes] : llvm::zip(outerLoop.getResults(), newOuterLoop.getResults())) {
    oldRes.replaceAllUsesWith(newRes);
  }

  rewriter.eraseOp(innerLoop);
  rewriter.eraseOp(outerLoop);

  // Update the uKernel to the new one in the innerLoop Body
  tiledOps[0] = *newInnerLoop.getBody()->getOps<linalg::GenericOp>().begin();

  // Update the loops
  loopOps[0] = newOuterLoop.getOperation();
  loopOps[1] = newInnerLoop.getOperation();

  llvm::errs() << "\nDump after replacements:\n";
  newOuterLoop.dump();

  return success();
}

// Apply a tiling transformation to a modified payload ops and store both the
// tiled operation  (uKernel) as well as the created tile loops.
static LogicalResult
applyTileTo(RewriterBase &rewriter, Operation *transformOp, Operation *target,
            CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, 
            transform::TransformResults &transformResults) {

  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  // Assign tile sizes
  // Input Stationary: N, NF * K2, NWIN * K3, NC, FH, FW  
  // Weight Stationary: N, NF * K3, NWIN * K2, NC, FH, FW  
  int64_t nFTiles = csa.mK_.num_filters * (res.schd == IS ? res.k2 : res.k3);
  int64_t nWinTiles = csa.mK_.nwindows * (res.schd == IS ? res.k3 : res.k2);
  SmallVector<int64_t, 6> tileSize = {1, nFTiles, nWinTiles, res.tile_c, 0, 0};
  SmallVector<OpFoldResult> tileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), tileSize);

  // Order:
  // Input Stationary: N, NC, NWIN, NF
  // Weight Stationary: N, NC, NF, NWIN
  int64_t outer = res.schd == IS ? 2 : 1;
  int64_t inner = res.schd == IS ? 1 : 2;
  SmallVector<int64_t, 4> tileInterchange = {0, 3, outer, inner};

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOfr).setInterchange(tileInterchange);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, tilingOptions);
  if (failed(tiledResults))
    return transformOp->emitError("failed the outermost tile operation");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(tilingInterfaceOp, tiledResults->replacements);

  // Perform the tiling in the inner convolution
  auto innerOp = tiledResults->tiledOps.front();

  SmallVector<int64_t, 6> innerTileSize = {0, csa.mK_.num_filters, csa.mK_.nwindows, 0, 0, 0};
  SmallVector<OpFoldResult> innerTileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), innerTileSize);

  int64_t innerFOrder = res.schd == IS ? 1 : 0;
  int64_t innerSOrder = res.schd == IS ? 0 : 1;
  SmallVector<int64_t, 2> innerInterchange = {innerFOrder, innerSOrder};

  auto innerTilingInterfaceOp = dyn_cast<TilingInterface>(innerOp);
  if (!innerTilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");
  scf::SCFTilingOptions innerTilingOptions;
  innerTilingOptions.setTileSizes(innerTileSizesOfr).setInterchange(innerInterchange);
  innerTilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(innerOp);
  FailureOr<scf::SCFTilingResult> innerTiledResults =
      scf::tileUsingSCF(rewriter, innerTilingInterfaceOp, innerTilingOptions);
  if (failed(innerTiledResults))
    return transformOp->emitError("failed the innermost tile operation");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(innerTilingInterfaceOp, innerTiledResults->replacements);

  // Report back the relevant handles to the transform op.
  tiledOps.push_back(innerTiledResults->tiledOps.front());
  for (Operation *loop : innerTiledResults->loops)
    loopOps.push_back(loop);
  for (Operation *loop : tiledResults->loops)
    loopOps.push_back(loop);

  // Swap the innner loops in the case of Input Stationary
  LogicalResult result0 = swapInductionVars(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result0)) return transformOp->emitError("failed to swap indvar Ops");

  transformResults.set(transformOp->getOpResult(0), tiledOps);
  for (auto [index, loop] : llvm::enumerate(loopOps))
    transformResults.set(transformOp->getOpResult(index + 1), {loop});

  return success();
}

///
/// Implementation of SConv::apply transform dialect operation.
///
DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Get context and convOp params
  MLIRContext *context = rewriter.getContext();
  auto targetOps = state.getPayloadOps(getTarget());

  assert(llvm::hasSingleElement(targetOps) && "expected a single target op");

  auto convOp = dyn_cast_or_null<linalg::Conv2DNchwFchwOp>(*targetOps.begin());
  if (!convOp)  
    return emitSilenceableError() << "expected a Conv2DNchwFchwOp for transformation";

  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();

  SmallVector<Value> inputs = convOp.getDpsInputs();
  ValueRange outputs = convOp.getDpsInits();
  Value input = inputs[0];
  Value filter = inputs[1];
  Value output = outputs[0];

  auto inputType = cast<ShapedType>(input.getType());
  auto filterType = cast<ShapedType>(filter.getType());
  auto outputType = cast<ShapedType>(output.getType());

  if (!filterType.hasStaticShape())
    return emitSilenceableError() << "expected a static shape for the filter";

  if (!inputType.hasStaticShape())
    return emitSilenceableError() << "expected a static shape for the input";

  // Does not support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return emitSilenceableError() << "expected all ones for dilations";

  auto inputShape = inputType.getShape();
  auto filterShape = filterType.getShape();
  auto outputShape = outputType.getShape();
  int64_t n = outputShape[0];
  int64_t ic = inputShape[1];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];
  int64_t oc = outputShape[1];
  int64_t oh = outputShape[2];
  int64_t ow = outputShape[3];

  // Create the Collapse shape to be inserted at begining
  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType = RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  // Create the affine maps, iterator types and output tensor shape
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction, reduction, reduction};

  // Get strides
  auto hstride = convOp.getStrides().getValues<int64_t>()[0];
  auto wstride = convOp.getStrides().getValues<int64_t>()[1];
  SmallVector<int64_t, 2> strides = {hstride, wstride};

  AffineExpr d0, d1, d2, d3, d4, d5;
  bindDims(context, d0, d1, d2, d3, d4, d5);
  auto lhsMap = AffineMap::get(6, 0, {d0, d3, d2.floorDiv(oh) * hstride + d4, d2 % oh * wstride + d5}, context);
  auto rhsMap = AffineMap::get(6, 0, {d1, d3, d4, d5}, context);
  auto resultMap = AffineMap::get(6, 0, {d0, d1, d2}, context);

  // Create the new genericOp that replaces the named convolution
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc,
      reshapedOutputType,
      inputs,
      ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap},
      newOpIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul = createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });

  // Create the Expanded Shape
  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(loc, outputType, genericOp.getResults().front(), outputReassocIndices);

  // replace convOp with (reshapedOutput + genericOp + reshapedResult)
  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  // Call the CSA Analysis
  ConvInfo csaConv = {ic, oh, ow, fh, fw, oc, 4};
  CSA csa = createCSAPass(csaConv);
  CSAStrategy res = csa();
  
  /* Just for test */
  res.schd = IS; res.k2 = 8; res.k3 = 2; res.tile_c = 16;
  /* Comment the code above to use the CSA Analysis */

  // Apply the tile in the genericOp based on the CSA Analysis
  LogicalResult result = applyTileTo(rewriter, getOperation(), genericOp, csa, res, strides, results);

  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
}

void transform::SConvOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
