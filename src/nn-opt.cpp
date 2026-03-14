#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "NNDialect.h"
#include "NNOps.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: nn-opt <input.mlir>\n";
    return 1;
  }

  // 1. Create the context and register dialects
  mlir::MLIRContext context;
  context.loadDialect<nn::NNDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  // 2. Parse the input file into a module
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // 3. Create a pass manager and add the canonicalize pass
  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  // 4. Run the pass manager
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  // 5. Print the transformed IR
  module->print(llvm::outs());
  return 0;
}
