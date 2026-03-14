cmake_minimum_required(VERSION 3.20)
project(nn-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# Find MLIR (and LLVM, which MLIR depends on)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# --- TableGen: generate C++ from .td files ---

set(LLVM_TARGET_DEFINITIONS NNDialect.td)
mlir_tablegen(NNDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NNDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(NNDialectIncGen)

set(LLVM_TARGET_DEFINITIONS NNOps.td)
mlir_tablegen(NNOps.h.inc -gen-op-decls)
mlir_tablegen(NNOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NNOpsIncGen)

# --- Build nn-opt ---

add_llvm_executable(nn-opt
  nn-opt.cpp
  NNDialect.cpp
  NNOps.cpp
  ReluCombine.cpp
)

add_dependencies(nn-opt NNDialectIncGen NNOpsIncGen)

target_link_libraries(nn-opt PRIVATE
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)
