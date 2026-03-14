#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "NNDialect.h"
#include "NNOps.h"
#include "NNLower.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: nn-opt <input.mlir>\n";
    return 1;
  }

  // 1. Create the context and register dialects
  mlir::MLIRContext context;
  context.loadDialect<nn::NNDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::cf::ControlFlowDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();

  // 2. Parse the input file into a module
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // 3. Create a pass manager and add passes
  mlir::PassManager pm(&context);

  // Canonicalize first (folds relu(relu(x)) -> relu(x))
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  // Lower nn ops to linalg/arith
  pm.addPass(std::make_unique<nn::NNToLinAlgLoweringPass>());

  // Bufferize (tensor -> memref)
  pm.addPass(mlir::bufferization::createOneShotBufferizePass());

  // Lower linalg to loops
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // Lower scf to cf (loops to branches)
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFToControlFlowPass());

  // Lower everything to LLVM dialect
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // 4. Run the pass manager
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  // 5. Print the transformed IR
  module->print(llvm::outs());
  return 0;
}
