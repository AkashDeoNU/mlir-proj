#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "NNOps.h"

namespace nn {
  struct NNToLinAlgLoweringPass : mlir::PassWrapper<NNToLinAlgLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation();
};
} // nn
