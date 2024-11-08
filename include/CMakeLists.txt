# Tell Tablegen to use SConv.td as input.
set(LLVM_TARGET_DEFINITIONS SConv.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(SConv.h.inc -gen-op-decls)
mlir_tablegen(SConv.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(SConvDialectIncGen)

# Don't forget to generate the documentation, this will produce a
# SConvDialect.md under docs/Dialects
add_mlir_doc(SConv SConvDialect docs/Dialects/ -gen-op-doc)
