cmake_minimum_required(VERSION 3.13.4)
project(SConvDialect LANGUAGES CXX)
add_custom_target(SConv)

# Defina o caminho para o LLVM e MLIR
set(LLVM_DIR $HOME/work/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR $HOME/work/llvm-project/build/lib/cmake/mlir)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Incluir diretórios de cabeçalhos
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_dependencies(SConv transform-opt)
add_llvm_executable(transform-opt
  transform-opt/transform-opt.cpp)

target_link_libraries(transform-opt
  PRIVATE
  MLIRIR
  MLIRMlirOptMain
  MLIRSideEffectInterfaces
  MLIRDestinationStyleOpInterface
  SConvDialect
)
