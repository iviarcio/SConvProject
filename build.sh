# mkdir build && cd build
cmake -G Ninja .. \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_LLD=ON \
   -DMLIR_DIR=$HOME/work/llvm-project/build/lib/cmake/mlir \
   -DLLVM_DIR=$HOME/work/llvm-project/build/lib/cmake/llvm
cmake --build . --target transform-opt
