#ifndef NN_LOWER_TO_LLVM_H
#define NN_LOWER_TO_LLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace nn {

// One-shot conversion that takes Func + SCF + CF + Arith + MemRef all the
// way to the LLVM dialect in a single applyFullConversion call, using a
// shared LLVMTypeConverter so memref/func/cf agree on ABI.
struct NNLowerToLLVMPass
    : mlir::PassWrapper<NNLowerToLLVMPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;
};

} // namespace nn

#endif // NN_LOWER_TO_LLVM_H
