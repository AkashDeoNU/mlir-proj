cmake_minimum_required(VERSION 3.20)
project(nn-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# --- TableGen: generate C++ from .td files ---

set(LLVM_TARGET_DEFINITIONS include/NNDialect.td)
mlir_tablegen(include/NNDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/NNDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(NNDialectIncGen)

set(LLVM_TARGET_DEFINITIONS include/NNOps.td)
mlir_tablegen(include/NNOps.h.inc -gen-op-decls)
mlir_tablegen(include/NNOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NNOpsIncGen)

# --- Build nn-opt ---

add_llvm_executable(nn-opt
  src/nn-opt.cpp
  src/NNDialect.cpp
  src/NNOps.cpp
  src/ReluCombine.cpp
  src/NNLower.cpp
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
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRLinalgTransforms
  MLIRBufferizationTransforms
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
)
