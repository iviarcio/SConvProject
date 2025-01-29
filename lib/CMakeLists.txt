# Outside examples, this should be `add_mlir_library`.
add_mlir_library(
  # Library called SConv.
  SConvDialect

  # Built from the following source files.
  SConv.cpp
  CSA.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  SConvDialectIncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PRIVATE
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)
